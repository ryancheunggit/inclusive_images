#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com
import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from CONSTANTS import (
    IMAGE_SIZE, NUM_CLASSES, DATA_DIR
)
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import cv2


def read_image(path_to_image):
    image = cv2.imread(path_to_image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE[:2])
    return image / 255


def str2bool(v):
    """
    used to parse string argument true and false to bool in argparse
    # https://stackoverflow.com/questions/15008758/
    # parsing-boolean-values-with-argparse
    """
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def get_label_encoder():
    ''' Load mappings between label (like '/m/01p3xl') <-> numerical encoding (like 999) '''
    assert os.path.exists('../data/label_encoder.pkl'), 'run prepare_data.py first!'
    enc = pickle.load(file=open(file='../data/label_encoder.pkl', mode='rb'))
    label_encoder, reverse_label_encoder = enc['label_encoder'], enc['reverse_label_encoder']
    return label_encoder, reverse_label_encoder, NUM_CLASSES


def get_label_name_mapping():
    label_name_mapping = pd.read_csv(os.path.join(DATA_DIR,'class-descriptions.csv'))
    label_name_mapping = {r['label_code']:r['description'] for _, r in label_name_mapping.iterrows()}
    return label_name_mapping


def get_label_embedding(k=1024):
    ''' Load the label embedding matrix '''
    embedding_file = DATA_DIR + 'label_emb_{}d.pkl'.format(k)
    assert os.path.exists(embedding_file), 'run mi_label_embedding.py first!'
    label_embedding = pickle.load(open(embedding_file, 'rb'))
    return label_embedding


def get_train_val_test():
    ''' Load prepared data '''
    assert os.path.exists('../data/train_info.pkl'), 'run prepare_data.py first!'
    s1_train = pickle.load(open('../data/train_info.pkl', 'rb'))
    s1_val = pickle.load(open('../data/val_info.pkl', 'rb'))
    sub = pd.read_csv('../data/stage_1_sample_submission.csv')
    return s1_train, s1_val, sub


def get_subset_train():
    ''' Load a 10% subset of training data '''
    train_info = pickle.load(open(DATA_DIR + 'train_subset_info.pkl', 'rb'))
    train_image_ids, train_labels, train_indices = train_info['image_ids'], train_info['labels'], train_info['indices']
    return train_image_ids, train_labels, train_indices


def format_test_prediction(sub, test_pred, reverse_label_encoder, threshold=0.5):
    ''' test_pred is the raw score output from nn of shape (sub.shape[0] x NUM_CLASSES) '''
    for i in tqdm(range(sub.shape[0])):
        prob = test_pred[i]
        label = ' '.join([reverse_label_encoder[i] for i in np.where(prob > threshold)[0]])
        sub.iloc[i, 1] = label
    return sub


def mean_sample_f2(y_true, y_pred_raw, threshold = 0.5):
    ''' calculate the fbeta score with beta == 2 based on raw score and a threshold '''
    y_pred = y_pred_raw > threshold
    agreement = np.sum(np.multiply(y_true, y_pred), axis=1)
    tp = np.sum(y_true, axis =1)
    pp = np.sum(y_pred, axis = 1)
    recall = agreement / (tp + 1e-7)
    precision = agreement / (pp + 1e-7)
    f2score = (1+2**2)*((precision*recall)/(2**2*precision+recall+1e-7))
    return np.mean(f2score)


def threshold_optimize(y_true, y_pred_raw, thresholds=None, num_bags=100, verbose=False):
    ''' very coarse search for best overall threshold via bootstrapping '''
    if not thresholds:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sample_size = y_true.shape[0]
    results = dict()
    best_threshold = thresholds[0]
    best_mean_f2 = 0
    for threshold in thresholds:
        f2scores = []
        for _ in range(num_bags):
            indices = np.random.choice(range(sample_size), size = sample_size, replace = True)
            f2scores.append(mean_sample_f2(y_true[indices], y_pred_raw[indices], threshold))
        if np.mean(f2scores) > best_mean_f2:
            best_mean_f2 = np.round(np.mean(f2scores), 3)
            best_threshold = threshold
        results[threshold] = f2scores
        if verbose:
            print(threshold, np.mean(f2scores), np.std(f2scores))
    return results, best_mean_f2, best_threshold


def plot_training_history(outpath, start_iter=0):
    PLOTCOLS = [
        'loss', 'val_loss', 'out_lbl_loss', 'val_out_lbl_loss', 'out_emb_loss', 'val_out_emb_loss',
        'out_lbl_f2', 'val_out_lbl_f2', 'threshold', 'tuned_val_f2'
    ]
    filepath = os.path.join(outpath, 'training_log.csv')
    df = pd.read_csv(filepath)
    plt.rcParams['figure.figsize'] = [16, 9]
    plt.rcParams['legend.fontsize'] = 12
    plt.tight_layout()
    cols = [c for c in df.columns if c in PLOTCOLS]
    try:
        p = plt.figure()
        df[cols].iloc[start_iter:,:].plot()
        plt.legend(loc='upper left', ncol=len(cols) // 2, frameon=True)
        plt.ylim([-1, 1])
        plt.xlabel('Iteration')
        plt.savefig(outpath + '/plot_all.png')
        plt.close('all')

        p = plt.figure()
        df[['out_lbl_loss', 'val_out_lbl_loss']].iloc[start_iter:].plot()
        plt.xlabel('Iteration')
        plt.savefig(outpath + '/plot_ce_loss.png')
        plt.close('all')

        p = plt.figure()
        df[['out_emb_loss', 'val_out_emb_loss']].iloc[start_iter:].plot()
        plt.xlabel('Iteration')
        plt.savefig(outpath + '/plot_cp_loss.png')
        plt.close('all')

        p = plt.figure()
        df[['out_lbl_f2', 'val_out_lbl_f2', 'tuned_val_f2']].iloc[start_iter:].plot()
        plt.xlabel('Iteration')
        plt.savefig(outpath + '/plot_f2.png')
        plt.close('all')

        p = plt.figure()
        sns.regplot(x=df.iteration, y=df.tuned_val_f2 - df.val_out_lbl_f2, lowess=True)
        plt.ylabel('Diff - f2 score')
        plt.xlabel('Iteration')
        plt.savefig(outpath + '/plot_tuned_f2_diff.png')
        plt.close('all')
    except:
        print("failed to make plot")
