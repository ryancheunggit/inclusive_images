#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com
import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from CONSTANTS import *
from keras.utils import Sequence
from keras.layers import *
from keras.initializers import glorot_uniform
from keras import backend as K
from keras import optimizers
from keras.optimizers import Optimizer
from keras.callbacks import Callback
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# --- Learning rate multiplier ---
def lr_multiplier(optimizer, ratio=0.1):
    print('mutiply learning rate by : {}'.format(ratio))
    K.set_value(optimizer.lr, ratio*K.eval(optimizer.lr))
    return optimizer


# --- Callback ---
class ModelCheckpointOriginal(Callback):
    ''' save the original model at end of per period epochs when fitting parallel model '''
    def __init__(self, model, filepath, verbose=1):
         self.model_to_save = model
         self.filepath = filepath
         self.verbose = verbose
    def on_epoch_end(self, epoch, logs=None):
        print('saving model')
        if self.verbose > 0:
            print('verbose')
        self.model_to_save.save(self.filepath + '_epoch_%d.h5' % epoch)


# --- Batch Generator ---
class BatchGenerator(Sequence):
    ''' create generators for training model or use model to generate prediction '''
    allowed_tasks = ['train_clf', 'train_emb', 'train_dual', 'train_tri', 'predict']
    def __init__(self,
            image_ids, labels, data_dir, batch_size=32, image_size=(224, 224, 3), num_classes=NUM_CLASSES, task='train',
            random_batches=False, augument_config=None, label_embedding=None
        ):
        """
        Crreate a data generator for training model / generating prediction

        Parameters
        ----------
        image_ids : list
            List of image_ids
        labels : matrix
            Matrix of shape # examples x # labels, values are 0 or 1, indicates an object exist in picture or not
        data_dir : string
            The path to image files, used together with image_ids to load images from disk
        batch_size : int
            Number of images per batch
        image_size : tuple
            (Height, Width, Channel) of image
        num_classes : int
            The number of classes, determines the classifier output layer shape
        task : string
            The task for the generator, x will be the same in all options, y will be different
                train - y is 0,1 encoded labels
                train_emb - y is sum of label embeddings
                train_dual - y is [y_train, y_train_emb]
                train_tri - y is [y_train, y_train_emb, y_train.sum(axis=1)]
                predict - y is random nonsense
        random_batches : bool
            Whether to draw random samples to form batch or to sequencially draw images based on the order in image_ids
        augument_config : dict
            # TODO: to be implemented
            Configuration to pass to image augumentor
        label_embedding : matrix
            Matrix of label embedding

        Returns
        -------
        Instance of BatchGenerator
            a generator object can be used in model.fit_generator or model.predict_generator for keras models
        """
        assert task in self.allowed_tasks, 'invalid task category, should be within {}'.format(allowed_tasks)
        if task in self.allowed_tasks[1:3]:
            assert label_embedding is not None, 'need label_embedding matrix for the task: {}'.format(task)
        self.total_images = len(image_ids)
        self.image_ids = image_ids
        self.labels = labels
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_batches = int(np.ceil(self.total_images / self.batch_size))
        self.task = task
        self.random_batches = random_batches
        self.augument_config = augument_config
        if task == 'predict':
            self.augument_config = None
        self.label_embedding = label_embedding
        self.input_shape = (batch_size, *image_size)
        self.output_shape = (batch_size, num_classes)

    @staticmethod
    def _read_and_transform_image(path_to_image, image_size, augument_config):
        """
        Load and transform image

        Parameters
        path_to_image : string
            path to image file
        image_size : tuple
            (Height, Width, Channel) of image
        augument_config : dict
            Configuration to pass to image augumentor
        ----------
        Returns
        np.ndarray
            numpy array with shape of image_size
        """
        image = cv2.imread(path_to_image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if augument_config is not None:
            image = ImageDataGenerator().apply_transform(image, augument_config)
        image = cv2.resize(image, image_size[:2])
        return image / 255

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        """ Produce a batch of (X, y) data """
        X = np.empty(shape=self.input_shape)
        y = np.empty(shape=self.output_shape)

        lower = idx * self.batch_size
        upper = (idx + 1) * self.batch_size
        indices = list(range(lower, upper))
        if upper > self.total_images:
            indices = list(range(lower, self.total_images)) + list(range(0, upper - self.total_images))

        if self.random_batches:
            np.random.seed() # SUPER IMPORTANT!
            indices = indices = np.random.choice(range(self.total_images), size = self.batch_size, replace=False)

        X = np.array([
            self._read_and_transform_image(
                path_to_image = os.path.join(self.data_dir, self.image_ids[i] + '.jpg'),
                image_size = self.image_size,
                augument_config = self.augument_config
            )
            for i in indices
        ])
        if self.task == 'train_clf':
            y = self.labels[indices].toarray()
        if self.task == 'train_emb':
            y = self.labels[indices].toarray()
            y = y.dot(self.label_embedding)
        if self.task == 'train_dual':
            y = self.labels[indices].toarray()
            y_emb = y.dot(self.label_embedding)
            y = [y, y_emb]
        if self.task == 'train_tri':
            y = self.labels[indices].toarray()
            y_card = y.sum(axis = 1)
            y = [y, y_emb, y_card]
        return X, y


# --- Loss ---
def focal_loss(gamma=2., alpha=.25):
    """
    Focal Loss for Dense Object Detection
    The idea is to balance cross entropy loss to put less weight on 'easy' example and more weight on 'hard' ones
    Reference: https://arxiv.org/pdf/1708.02002.pdf

    binary_crossentropy is:
    CE(p,y) = -log(p) if y == 1 else -log(1-p)

    Let pt = p if y == 1 else 1-p
    Then CE(p, y) simplifies to CE(pt) = -log(pt)

    alpha-balanced CE loss is
        CE(pt) = - alpha * log(pt)

    focal loss(FL) is:
        FL(pt) = - (1 - pt)^gamma * log(pt)
    where gamma is a tunable hyper-parameter

    alpha-balanced FL loss is:
        FL(pt) = - alpha(1 - pt)^gamma * log(pt)

    Based on paper, alpha-balanced FL out perform vanilla FL loss slightly

    default hyperparameter gamma = 2 and alpha = .25 were taken from paper page 5

    Parameters
    ----------
    alpha : float
        balancing factor for positive / negative examples
    gamma : float
        focusing parameter for hard / easy examples

    Returns
    -------
    Function
        A function to calculate focal loss
    """
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        loss = -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
        return loss
    return focal_loss_fixed


def f2_bce_loss(y_true, y_pred):
    tp_loss = K.sum(y_true * (1 - K.binary_crossentropy(y_pred, y_true)), axis=-1)
    fp_loss = K.sum((1 - y_true) * K.binary_crossentropy(y_pred, y_true), axis=-1)
    return - K.mean(5 * tp_loss / ((4 * K.sum(y_true, axis = -1)) + tp_loss + fp_loss))


def f2_loss(y_true, y_pred):
    epsilon = 1e-6
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    y_true = K.clip(y_true, K.epsilon(), 1.0 - K.epsilon())
    true_and_pred = y_true * y_pred
    ttp_sum = K.sum(true_and_pred, axis=1)
    tpred_sum = K.sum(y_pred, axis=1)
    ttrue_sum = K.sum(y_true, axis=1)
    tprecision = ttp_sum / tpred_sum
    trecall = ttp_sum / ttrue_sum
    tf_score = (5 * tprecision * trecall + K.epsilon()) / (4 * tprecision + trecall + K.epsilon())
    return -K.mean(tf_score)


# --- Metrics ---
def binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)


def fbeta(beta=2, threshold=0.1):
    """
    attemp to replace the fixed f2 score function with a more general fbeta metric
    # TODO: Check implementation! Would it break if threshold > 0.5 ?
    """
    def fbeta_fixed(y_true, y_pred):
        shift = 0.5 - threshold
        y_pred = K.clip(y_pred, 0, 1)
        y_pred_bin = K.round(y_pred + shift)
        tp = K.sum(K.round(y_true * y_pred_bin), axis=1) + K.epsilon()
        fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)
        fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fbeta_score = (1+beta**2)*((precision*recall)/(beta**2*precision+recall+K.epsilon()))
        return K.mean(fbeta_score)
    return fbeta_fixed


def f2(y_true, y_pred, beta=2):
    """
    Mean samples fbeta score function with beta == 2, which is what the competition uses

    Fbeta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)

    With mean samples fbeta score, we calculate fbeta for each example and finally calculate a overall mean
    """
    agreement = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=1)
    true_positive = K.sum(K.round(K.clip(y_true, 0, 1)), axis=1)  # incase we use confidence of machine label as truth
    pred_positive = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=1)
    recall = agreement / (true_positive + K.epsilon())
    precision = agreement / (pred_positive + K.epsilon())
    f2score = (1+beta**2)*((precision*recall)/(beta**2*precision+recall+K.epsilon()))
    return K.mean(f2score)


# --- NN Architecture Hack ---
def pay_attention(model):
    """
    Attch an attention layer on a debse output layer
    Reference : https://github.com/philipperemy/keras-attention-mechanism
    """
    X = model.output
    Att_probs = Dense(units = K.int_shape(X)[1], activation = 'softmax', name='Att_prob')(X)
    Att_out = Multiply(name='Att_out')([X, Att_probs])
    new_model = Model(inputs=model.layers[0].input, outputs=Att_out)
    return new_model


def insert_last_dropout(model, dropout_rate):
    ''' Attch an dropout layer before final output layer '''
    out = model.layers.pop()
    X = model.layers[-1].output
    X = Dropout(rate=dropout_rate)(X)
    X = Dense(units=out.units, activation = out.activation, name =out.name, kernel_initializer=out.kernel_initializer)(X)
    new_model = Model(inputs=model.layers[0].input, outputs=X)
    new_model.layers[-1].set_weights(out.get_weights())
    return new_model


def reuse_weights(to_model, from_model, skip_output_layer_weights=True, set_trainable=False):
    """
    reuse weights from a trained model as initialization for a new model
    # TODO: better layer compatability checks
    """
    layer_lookup = {layer.name: layer for layer in from_model.layers}
    for layer in tqdm(to_model.layers):
        if 'out' in layer.name and skip_output_layer_weights:
            continue
        if layer.name in layer_lookup:
            weights = layer_lookup[layer.name].get_weights()
            if weights:
                try:
                    layer.set_weights(weights)
                    layer.trainable = set_trainable
                except:
                    print('layer {} might not be compatable between two models'.format(layer.name))
    return to_model


def defrost_model(model):
    """ simply set all layer trainable """
    for layer in model.layers:
        layer.trainable=True
    return model


def retask_to_emb(model, embedding_dim=4096, dropout_rate=0.2):
    """
    Take in a vanilla resnet model,
    swap the classifier layer with a regression layer prediction the sum of label embedding
    and dropout between glp and the above two
    """
    out=model.layers.pop()
    X=model.layers[-1].output
    X=Dropout(rate=dropout_rate)(X)
    Emb_out=Dense(units=embedding_dim, activation='linear', name='out_emb', kernel_initializer=glorot_uniform())(X)
    new_model=Model(inputs=model.layers[0].input, outputs=[Emb_out])
    return new_model


def retask_to_dual(model, embedding_dim=4096, dropout_rate=0.2):
    """
    Take in a vanilla resnet model,
    throw out the classifier layer,
    add two output layers,
        one for label
        one for sum of label embedding
    and dropout between glp and the above two
    """
    out=model.layers.pop()
    X=model.layers[-1].output
    X=Dropout(rate=dropout_rate)(X)
    Label_out=Dense(units=out.units, activation='sigmoid', name='out_lbl', kernel_initializer=glorot_uniform())(X)
    Emb_out=Dense(units=embedding_dim, activation='linear', name='out_emb', kernel_initializer=glorot_uniform())(X)
    new_model=Model(inputs=model.layers[0].input, outputs=[Label_out, Emb_out])
    return new_model


def retask_to_tri(model, embedding_dim=4096, dropout_rate=0.2):
    """
    Take in a vanilla resnet model,
    throw out the classifier layer,
    add two output layers,
        one for label
        one for sum of label embedding
        one for the cardinality of labels per image
    and dropout between glp and the above three
    """
    out=model.layers.pop()
    X=model.layers[-1].output
    X=Dropout(rate=dropout_rate)(X)
    Emb_out=Dense(units=embedding_dim, activation='linear', name='out_emb', kernel_initializer=glorot_uniform())(X)
    Label_out=Dense(units=out.units, activation='sigmoid', name='out_lbl', kernel_initializer=glorot_uniform())(X)
    Card_out=Dense(units=1, activation='linear', name='out_card', kernel_initializer=glorot_uniform())(X)
    new_model=Model(inputs=model.layers[0].input, outputs=[Label_out, Emb_out, Card_out])
    return new_model


def strip_aux_task(model, aux_out=('out_emb', 'out_card'), add_lbl_out=True):
    """
    strip out output layers by layer name, optionally add label clf layer
    """
    for layer in model.layers:
        if layer.name in aux_out:
            print('---- strip out aux output layer {}'.format(layer.name))
            model.layers.remove(layer)

    if add_lbl_out:
        print('---- adding multilabel classifier layer')
        out = model.layers[-1].output
        Lbl_out = Dense(units=NUM_CLASSES, activation='sigmoid', name='out_lbl', kernel_initializer=glorot_uniform())(out)
        model=Model(inputs=model.layers[0].input, outputs=Lbl_out)
    else:
        model = Model(inputs=model.layers[0].input, outputs=model.layers[-1].output)
    return model


def freeze_except_last(model, last=1):
    ''' freeze layers upuntil the last x layers, used to fine tune the final classifier layer '''
    num_layers = len(model.layers)
    for idx, layer in enumerate(model.layers[:-last]):
        layer.trainable = False
    return Model(inputs=model.layers[0].input, outputs = model.layers[-1].output)


# --- Model Diagnosis ----
def get_model_memory_usage(batch_size, model):
    """
    Calculate the memory requirement of a model given a batch_size
    Refernce : https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
    # TODO: Does not seem accurate
    """
    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem
    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])
    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


# --- Custom Optimizer ---
class AggBatchSGD(Optimizer):
    '''
    vanilla SGD which aggregates gradients from multiple mini batches and then perfom an update

    # TODO: Can we get a implementation of this with momentum and nesterov?
    # Larger batch_size is really desirable.
    # See idea from : https://arxiv.org/abs/1711.00489
    '''
    def __init__(self, lr=0.01, batches_per_update=1, **kwargs):
        super(AggBatchSGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.batches_per_update = batches_per_update

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        shapes = [K.int_shape(p) for p in params]
        sum_grads = [K.zeros(shape) for shape in shapes]
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        self.weights = [self.iterations] + sum_grads
        for p, g, sg in zip(params, grads, sum_grads):
            new_p = p - self.lr * sg / float(self.batches_per_update)
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            cond = K.equal(self.iterations % self.batches_per_update, 0)
            self.updates.append(K.switch(cond, K.update(p, new_p), p))
            self.updates.append(K.switch(cond, K.update(sg, g), K.update(sg, sg+g)))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)), 'batches_per_update': self.batches_per_update}
        base_config = super(AggBatchSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
