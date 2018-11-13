#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com
import os
import argparse
from pprint import pprint
from CONSTANTS import *
from util import *
from nn_util import *
from nn_model import *
from keras.models import load_model
from keras import backend as K
from keras.optimizers import SGD, Adam, RMSprop
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

print('---- MY PID IS: {} KILL ME TO RELEASE GPU IF INTERRUPT PROGRAM MIDWAY'.format(os.getpid()))

loss_functions = {
    'focal_loss': focal_loss(),
    'binary_crossentropy': 'binary_crossentropy',
    'cosine_proximity': 'cosine_proximity',
    'mean_squared_error': 'mean_squared_error',
    'f2_bce_loss': f2_bce_loss,
    'f2_loss': f2_loss
}

monit_metrics = {
    'f2': f2,
    'fbeta': fbeta(beta=2, threshold=0.1),
    'binary_crossentropy': 'binary_crossentropy',
    'cosine_proximity': 'cosine_proximity',
    'mean_squared_error': 'mean_squared_error',
}

task_keys = {
    '_clf': ['out_lbl'],
    '_emb': ['out_emb'],
    '_dual': ['out_lbl', 'out_emb'],
    '_tri': ['out_lbl', 'out_emb', 'out_card']
}

optimizers = {
    'Adam': Adam(lr=0.01),
    'Adam_s': Adam(lr=0.001),
    'SGD': SGD(lr=0.01, momentum=0.9, nesterov=True),
    'SGD_s': SGD(lr=0.001, momentum=0.99, nesterov=True),
    'RMSprop': RMSprop(lr=0.0001)
}

def get_his_ele(his, attr, digits=4):
    if attr not in his:
        return "\'nan\'"
    return str(np.round(his[attr][0], digits))

def get_model(
        *,
        model_arch='ResNet50',
        task='',
        embedding_dim=2048,
        dropout_rate=0.2,
        optimizer=Adam(),
        loss={'out_lbl','binary_crossentropy'},
        loss_weights={'out_lbl':1},
        metrics={'out_lbl': f2},
        use_weights_from_model=None,
        se_ratio=0,
        gp_cardinality=0,
        feature='global_max_pooling',
        deeper_fc=False
    ):
    assert task in ['_clf', '_emb', '_dual', '_tri']
    if '50' in model_arch:
        from nn_model import ResNet50 as ResNet
    elif '101' in model_arch:
        from nn_model import ResNet101 as ResNet
    else:
        from nn_model import ResNet152 as ResNet
    model=ResNet(input_shape=IMAGE_SIZE, classes=NUM_CLASSES, se_ratio=se_ratio, gp_cardinality=gp_cardinality, feature=feature, deeper_fc=deeper_fc)
    print('---- creating vanilla resnet model')
    if task == '_clf':
        model=insert_last_dropout(model,dropout_rate=dropout_rate)
    if task == '_emb':
        model=retask_to_emb(model, embedding_dim=embedding_dim, dropout_rate=dropout_rate)
    if task == '_dual':
        model=retask_to_dual(model, embedding_dim=embedding_dim, dropout_rate=dropout_rate)
    if task == '_tri':
        model=retask_to_tri(model, embedding_dim=embedding_dim, dropout_rate=dropout_rate)

    if use_weights_from_model:
        print('---- loading weights from model')
        assert os.path.exists(use_weights_from_model), 'need a model checkpoint to load weights'
        other_model=load_model(use_weights_from_model, compile=False)
        model=reuse_weights(to_model=model, from_model=other_model)

    print('---- compiling model ')
    model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics)
    return model

# NOTE main is too heavy, but refactoring is: https://www.youtube.com/watch?v=zGxwbhkDjZM
def main(*,
        model_arch, task, label_emb_dim, feature, deeper_fc,
        batch_size, batch_per_iter, steps, from_iter, use_weights_from_model, save_checkpoints, log_history, lr_ratio,
        lbl_loss, emb_loss, card_loss,
        lbl_loss_weight, emb_loss_weight, card_loss_weight,
        lbl_metric, emb_metric, card_metric,
        aug_hflip, aug_rotation, aug_shift, aug_shear, aug_zoom,
        optimizer, recompile,
        gpu_to_use,
        defrost,
        swa,
    ):

    # use my first GPU to training, so that I can keep experiment with the second one
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_to_use

    model_name=model_arch + task
    if task in ['_emb', '_dual', '_tri']:
        model_name += '_{}'.format(label_emb_dim)
    # SE and X setting are right now hard coded, and halved the value from original papers
    se_ratio = 8 if 'SE' in model_arch else 0
    gp_cardinality = 16 if 'Xt' in model_arch else 0
    if feature != 'global_max_pooling':
        model_name += '_' + feature
    if deeper_fc:
        model_name += '_deeper_fc'
    print('---- model_name: ', model_name)

    # --- LOAD DATA
    print('---- loading data')
    label_encoder, reverse_label_encoder, NUM_CLASSES=get_label_encoder()
    label_embedding=get_label_embedding(label_emb_dim)
    s1_train, s1_val, sub=get_train_val_test()
    val_y=s1_val['val_labels'].toarray()

    loss = {
        'out_lbl': loss_functions[lbl_loss],
        'out_emb': loss_functions[emb_loss],
        'out_card': loss_functions[card_loss],
    }

    loss_weights = {
        'out_lbl': lbl_loss_weight,
        'out_emb': emb_loss_weight,
        'out_card': card_loss_weight,
    }

    metrics = {
        'out_lbl': monit_metrics[lbl_metric],
        'out_emb': monit_metrics[emb_metric],
        'out_card': monit_metrics[card_metric],
    }

    # --- CREATE GENERATORS
    print('---- creating generators')
    val_gen=BatchGenerator(
        image_ids=s1_val['val_image_ids'],
        labels=s1_val['val_labels'],
        data_dir=TEST_IMAGE_DIR,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_embedding=label_embedding,
        task='train' + task
    )
    train_gen=BatchGenerator(
        image_ids=s1_train['train_image_ids'],
        labels=s1_train['train_labels'],
        data_dir=TRAIN_IMAGE_DIR,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_embedding=label_embedding,
        task='train' + task,
        random_batches=True,
        augument_config={
            'flip_horizontal': aug_hflip,
            'rotation_range': aug_rotation,
            'width_shift_range': aug_shift,
            'height_shift_range':aug_shift,
            'shear_range': aug_shear,
            'zoom_range': aug_zoom,
        }
    )

    # look for previous model checkpoints
    latest_iteration=0
    if os.path.exists(MODEL_DIR + model_name):
        print('---- model checkpoints folder found')
        model_checkpoints=os.listdir(MODEL_DIR + model_name)
        if model_checkpoints:
            latest_iteration=max([int(i.split('_')[1].split('.')[0]) for i in model_checkpoints if 'iteration' in i])
            print('---- latest_checkpoints is {}'.format(latest_iteration))
        else:
            print('---- model checkpoints folder is empty')
    else:
        print('---- model checkpoints folder not found, creating it')
        os.mkdir(MODEL_DIR + model_name)

    if from_iter >= 0 and latest_iteration >= from_iter:
        print('---- from_iter is set, will start training from iteration {}'.format(from_iter))
        latest_iteration=from_iter

    # ---- CREATE / LOAD MODEL
    # ---- NOTE optimizer in model compile options are hard coded here rather than aruguments to the program
    if latest_iteration != 0:
        print('---- Loading model from checkpoints')
        model=load_model(
            filepath=MODEL_DIR + '{}/iteration_{}.h5'.format(model_name, latest_iteration),
            custom_objects={'f2': f2, 'focal_loss': focal_loss(), 'f2_loss':f2_loss, 'f2_bce_loss': f2_bce_loss},
            compile=True
        )
        if recompile:
            print('---- recompiling model on updated loss, loss_weights or metrics configs')
            # TODO: allow updates on optimizer as well
            model.compile(
                optimizer=model.optimizer,
                loss={k:v for k, v in loss.items() if k in model.loss.keys()},
                loss_weights={k:v for k, v in loss_weights.items() if k in model.loss_weights.keys()},
                metrics={k:v for k, v in metrics.items() if k in model.metrics.keys()}
            )
    else:
        print('---- Creating model since no model checkpoint exists')
        model=get_model(
            model_arch=model_arch,
            task=task,
            embedding_dim=label_emb_dim,
            dropout_rate=0.2,
            optimizer=optimizers[optimizer],
            loss={k:v for k,v in loss.items() if k in task_keys[task]},
            loss_weights={k:v for k,v in loss_weights.items() if k in task_keys[task]},
            metrics={k:v for k,v in metrics.items() if k in task_keys[task]},
            use_weights_from_model=use_weights_from_model,
            se_ratio=se_ratio,
            gp_cardinality=gp_cardinality,
            feature=feature,
            deeper_fc=deeper_fc
        )


    if defrost:
        print('---- make sure all layers are trainable')
        model = defrost_model(model)
        model.compile(
            optimizer=model.optimizer,
            loss={k:v for k, v in loss.items() if k in model.loss.keys()},
            loss_weights={k:v for k, v in loss_weights.items() if k in model.loss_weights.keys()},
            metrics={k:v for k, v in metrics.items() if k in model.metrics.keys()}
        )
    print('---- estimated model memeory usage: {}'.format(get_model_memory_usage(batch_size, model)))

    print('---- model summary')
    print(model.summary())
    try:
        print('---- loss config')
        for loss_name in model.loss.keys():
            print(model.loss[loss_name], model.loss_weights[loss_name])
    except:
        print('having trouble get access to loss config')

    print('---- current learning_rate = {}'.format(K.eval(model.optimizer.lr)))

    if swa:
        print('---- swa used, initialize swa weights using current model weights')
        swa_weights = model.get_weights()

    latest_iteration += 1
    # --- TRAINING MODEL
    print('---- training model for {} iterations'.format(steps))
    for iteration in range(latest_iteration, latest_iteration + steps):
        # NOTE right now learning rate adjustment is down by hand hacks with program argument
        # adjust learning rate when observed plateau
        if lr_ratio != 1:
            lr_multiplier(model.optimizer, ratio=lr_ratio)
            print('new learning rate: {}'.format(K.eval(model.optimizer.lr)))
            lr_ratio = 1 # make sure if training more than 1 iterations, only modify lr in the first iteration

        print('---- Training {} th iteration ---'.format(iteration))
        history=model.fit_generator(
            generator=train_gen,
            steps_per_epoch=batch_per_iter,
            max_queue_size=32,
            epochs=1,
            verbose=1,
            validation_data=val_gen,
            use_multiprocessing=True,
            workers=4,
            shuffle=True,
        )

        if swa:
            print('---- swa used, updating swa weights')
            n_models = iteration - latest_iteration + 1
            for i, weights in enumerate(model.get_weights()):
                swa_weights[i] = (n_models * swa_weights[i] + weights) / (n_models + 1)

        if save_checkpoints:
            print('---- saving model checkpoints')
            model.save(MODEL_DIR + '{}/iteration_{}.h5'.format(model_name, iteration))

        print('---- generating prediction on validation data')
        val_pred=model.predict_generator(
            generator=val_gen,
            use_multiprocessing=True,
            workers=4,
            verbose=1
        )

        if type(val_pred) == list:
            val_lbl_pred=val_pred[0][:1000, :]
        else:
            val_lbl_pred=val_pred[:1000,:]

        if task != '_emb':
            results, best_mean_f2, best_threshold=threshold_optimize(val_y, val_lbl_pred, verbose=True)
        else:
            best_threshold, best_mean_f2 = 'nan', 'nan'

        if log_history:
            if not os.path.exists(LOG_DIR + model_name):
                os.mkdir(LOG_DIR + model_name)

            fields = list(history.history.keys())
            with open(LOG_DIR + '{}/training_log.csv'.format(model_name), 'a') as f:
                if iteration == 1:
                    header=','.join(
                        ['iteration', 'batch_per_iter', 'batch_size'] +
                        fields +
                        ['threshold', 'tuned_val_f2']
                    ) + '\n'
                    f.write(header)
                result=','.join(
                    ['{:>3d}'.format(iteration), str(batch_per_iter), str(batch_size)] +
                    [get_his_ele(history.history, f, 4) for f in fields] +
                    [str(best_threshold), str(best_mean_f2)]
                ) + '\n'
                print('---- iteration - {} results:'.format(iteration))
                print(result)
                f.write(result)
            if iteration != 0:
                plot_training_history(LOG_DIR + model_name)

    if swa:
        print('---- fixing swa BatchNormalization statistics')
        model.set_weights(swa_weights)
        for layer in model.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False
        model.compile(
            optimizer=model.optimizer,
            loss={k:v for k, v in loss.items() if k in model.loss.keys()},
            loss_weights={k:v for k, v in loss_weights.items() if k in model.loss_weights.keys()},
            metrics={k:v for k, v in metrics.items() if k in model.metrics.keys()}
        )
        history=model.fit_generator(
            generator=train_gen,
            steps_per_epoch=batch_per_iter,
            max_queue_size=32,
            epochs=1,
            verbose=1,
            validation_data=val_gen,
            use_multiprocessing=True,
            workers=4,
            shuffle=True,
        )
        print('---- saving swa model checkpoints')
        model.save(MODEL_DIR + '{}/swa_{}_{}.h5'.format(model_name, iteration - steps + 1, iteration))
        print('---- generating prediction on validation data')
        val_pred=model.predict_generator(generator=val_gen, use_multiprocessing=True, workers=4, verbose=1)
        if type(val_pred) == list:
            val_lbl_pred=val_pred[0][:1000, :]
        else:
            val_lbl_pred=val_pred[:1000,:]
        if task != '_emb':
            results, best_mean_f2, best_threshold=threshold_optimize(val_y, val_lbl_pred, verbose=True)
    # latest_iteration=iteration


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='model training aruguments')
    parser.add_argument('--model_arch', type=str, default='ResNet50')
    parser.add_argument('--task', type=str, default='_dual')
    parser.add_argument('--label_emb_dim', type=int, default=1024)
    parser.add_argument('--feature', type=str, default='global_avg_pooling')
    parser.add_argument('--deeper_fc', type=str2bool, default='false')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--batch_per_iter', type=int, default=10000)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--from_iter', type=int, default=-1)
    parser.add_argument('--use_weights_from_model', type=str, default=None)
    parser.add_argument('--save_checkpoints', type=str2bool, default='true')
    parser.add_argument('--log_history', type=str2bool, default='true')
    parser.add_argument('--lr_ratio', type=float, default=1.)
    parser.add_argument('--lbl_loss', type=str, default='f2_loss')
    parser.add_argument('--emb_loss', type=str, default='cosine_proximity')
    parser.add_argument('--card_loss', type=str, default='mean_squared_error')
    parser.add_argument('--lbl_loss_weight', type=float, default=1)
    parser.add_argument('--emb_loss_weight', type=float, default=1)
    parser.add_argument('--card_loss_weight', type=float, default=0.0001)
    parser.add_argument('--lbl_metric', type=str, default='f2')
    parser.add_argument('--emb_metric', type=str, default='cosine_proximity')
    parser.add_argument('--card_metric', type=str, default='mean_squared_error')
    parser.add_argument('--aug_hflip', type=str2bool, default='true')
    parser.add_argument('--aug_rotation', type=int, default=5)
    parser.add_argument('--aug_shift', type=float, default=0.1)
    parser.add_argument('--aug_shear', type=float, default=0.1)
    parser.add_argument('--aug_zoom', type=float, default=0.1)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--recompile', type=str2bool, default='false')
    parser.add_argument('--gpu_to_use', type=str, default='0')
    parser.add_argument('--defrost', type=str2bool, default='false')
    parser.add_argument('--swa', type=str2bool, default='false')
    args=parser.parse_args()
    print('parameters are:')
    pprint(vars(args))
    main(**vars(args))
