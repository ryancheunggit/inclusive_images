#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com
from CONSTANTS import *
from nn_util import *
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import glorot_uniform
from keras.models import Model

# --- [SE]ResNe[X]t Blocks ---
def identity_block(X, f, filters, stage, block, gp_cardinality = 0, se_ratio = 0):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid',
        name = conv_name_base + '2a', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same',
        name = conv_name_base + '2b', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    if gp_cardinality > 0:
        X = grouped_convolution(X, F2 // gp_cardinality, gp_cardinality = gp_cardinality, strides=1)

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid',
        name = conv_name_base + '2c', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    if se_ratio > 0:
        X = squeeze_excite_block(X, F3, stage, block, se_ratio = se_ratio)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

def convolutional_block(X, f, filters, stage, block, strides = 2, gp_cardinality=32, se_ratio=16):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (strides,strides), padding = 'valid',
        name = conv_name_base + '2a', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same',
        name = conv_name_base + '2b', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    if gp_cardinality > 0:
        X = grouped_convolution(X, F2 // gp_cardinality, gp_cardinality = gp_cardinality, strides=1)

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid',
        name = conv_name_base + '2c', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    if se_ratio > 0:
        X = squeeze_excite_block(X, F3, stage, block, se_ratio = se_ratio)

    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (strides, strides), padding = 'valid',
        name = conv_name_base + '1', kernel_initializer = glorot_uniform())(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

# --- SE Block ---
def squeeze_excite_block(X, filters, stage, block, se_ratio=16):
    ''' FOR SE-ResNet https://arxiv.org/pdf/1709.01507.pdf '''
    se_name_base = 'se' + str(stage) + block

    SE = GlobalAveragePooling2D(name= str(stage) + block + '_gap')(X)
    SE = Reshape([1, 1, filters])(SE)
    SE = Dense(units = filters // se_ratio, activation='relu', use_bias=False,
        name = se_name_base + '_sqz', kernel_initializer='he_normal')(SE)
    SE = Dense(units = filters, activation='sigmoid', use_bias=False,
        name = se_name_base + '_exc', kernel_initializer='he_normal')(SE)
    SE = Reshape([1, 1, filters])(SE)
    X = Multiply()([X, SE])
    return X

# --- X Block ---
def grouped_convolution(X, grouped_channels, gp_cardinality, strides):
    ''' FOR ResNeXt https://arxiv.org/pdf/1611.05431.pdf '''
    X_in = X
    group = []
    for c in range(gp_cardinality):
        X = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels])(X_in)
        X = Conv2D(filters = grouped_channels, kernel_size = (3, 3), strides=(strides, strides),
            padding='same', use_bias=False, kernel_initializer='he_normal')(X)

        group.append(X)
    X_merged = Concatenate(axis=-1)(group)
    X = BatchNormalization(axis=-1)(X_merged)
    X = Activation('relu')(X)
    return X


def ResNet(
        input_shape=IMAGE_SIZE,
        classes=NUM_CLASSES,
        filter_sizes=[64, 128, 256, 512],
        num_channels=[256, 512, 1024, 2048],
        num_layers=[3, 4, 6, 3],
        se_ratio=0,
        gp_cardinality=0,
        feature='global_max_pooling',
        deeper_fc=False
    ):
    """

    Build (SE)ResNe(X)t models

    ResNet paper: https://arxiv.org/abs/1512.03385.pdf
    SE-ResNet paper: https://arxiv.org/pdf/1709.01507.pdf
    ResNeXt paper: https://arxiv.org/pdf/1611.05431.pdf
    """
    total_layers=2 + 3* sum(num_layers)
    model_name='Res'
    if se_ratio > 0:
        model_name='SE-' + model_name
    if gp_cardinality > 0:
        model_name=model_name + 'NeXt'
    else:
        model_name=model_name + 'Net'
    model_name += str(total_layers)

    X_input=Input(input_shape)
    X=ZeroPadding2D((3, 3))(X_input)

    # conv1_x
    X=Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
        name='conv1', kernel_initializer=glorot_uniform())(X)
    X=BatchNormalization(axis=3, name='bn_conv1')(X)
    X=Activation('relu')(X)
    X=MaxPooling2D((3, 3), strides=(2, 2))(X)

    # conv2_x
    filters=(filter_sizes[0], filter_sizes[0], num_channels[0])
    X=convolutional_block(X, f=3, filters=filters, stage=2, block='1', strides=1,
        gp_cardinality=gp_cardinality, se_ratio=se_ratio)
    for block_name in map(str, range(98, 97 + num_layers[0] )):
        X=identity_block(X, 3, filters, stage=2, block=block_name,
            gp_cardinality=gp_cardinality, se_ratio=se_ratio)

    # conv3_x
    filters=(filter_sizes[1], filter_sizes[1], num_channels[1])
    X=convolutional_block(X, f=3, filters=filters, stage=3, block='1', strides=2,
        gp_cardinality=gp_cardinality, se_ratio=se_ratio)
    for block_name in map(str, range(98, 97 + num_layers[1])):
        X=identity_block(X, 3, filters, stage=3, block=block_name,
            gp_cardinality=gp_cardinality, se_ratio=se_ratio)

    # conv4_x
    filters=(filter_sizes[2], filter_sizes[2], num_channels[2])
    X=convolutional_block(X, f=3, filters=filters, stage=4, block='1', strides=2,
        gp_cardinality=gp_cardinality, se_ratio=se_ratio)
    for block_name in map(str, range(98, 97 + num_layers[2])):
        X=identity_block(X, 3, filters, stage=4, block=block_name,
            gp_cardinality=gp_cardinality, se_ratio=se_ratio)

    # conv5_x
    filters=(filter_sizes[3], filter_sizes[3], num_channels[3])
    X=convolutional_block(X, f=3, filters=filters, stage=5, block='1', strides=2,
        gp_cardinality=gp_cardinality, se_ratio=se_ratio)
    for block_name in map(str, range(98, 97 + num_layers[3])):
        X=identity_block(X, 3, filters, stage=5, block=block_name,
            gp_cardinality=gp_cardinality, se_ratio=se_ratio)

    if feature == 'global_max_pooling':
        X=GlobalMaxPooling2D()(X)
    elif feature == 'global_avg_pooling':
        X=GlobalAveragePooling2D()(X)
    elif feature == 'concat':
        gmp=GlobalMaxPooling2D()(X)
        gap=GlobalAveragePooling2D()(X)
        X=Concatenate()([gmp, gap])
    else:
        X=Flatten()(X)

    if deeper_fc:
        X=Dense(units=classes, activation='linear', name='fc_1',  kernel_initializer=glorot_uniform())(X)
        X=LeakyReLU(alpha=0.01)(X)
    X=Dense(units=classes, activation='sigmoid', name='out_lbl', kernel_initializer=glorot_uniform())(X)
    model=Model(inputs=X_input, outputs=X, name=model_name)
    return model

def ResNet50(input_shape, classes, se_ratio=0, gp_cardinality=0, feature='global_max_pooling', deeper_fc=False):
    return ResNet(input_shape, classes, num_layers=[3, 4, 6, 3], se_ratio=se_ratio, gp_cardinality=gp_cardinality, feature=feature, deeper_fc=deeper_fc)
def ResNet101(input_shape, classes, se_ratio=0, gp_cardinality=0, feature='global_max_pooling', deeper_fc=False):
    return ResNet(input_shape, classes, num_layers=[3, 4, 23, 3], se_ratio=se_ratio, gp_cardinality=gp_cardinality, feature=feature, deeper_fc=deeper_fc)
def ResNet152(input_shape, classes, se_ratio=0, gp_cardinality=0, feature='global_max_pooling', deeper_fc=False):
    return ResNet(input_shape, classes, num_layers=[3, 8, 36, 3], se_ratio=se_ratio, gp_cardinality=gp_cardinality, feature=feature, deeper_fc=deeper_fc)
