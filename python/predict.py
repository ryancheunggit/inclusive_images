#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from CONSTANTS import *
from util import read_image, get_label_encoder, str2bool
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import argparse
parser=argparse.ArgumentParser(description='model training aruguments')
parser.add_argument('--model_checkpoint', type=str, default='ResNet101_clf_global_avg_pooling/swa_31_40.h5')
# parser.add_argument('--model_checkpoint', type=str, default='ResNet50_clf_global_avg_pooling/swa_36_45.h5')
parser.add_argument('--test_image_dir', type=str, default=TEST_IMAGE_DIRNAME)
parser.add_argument('--subfile', type=str, default='stage_1_sample_submission.csv')
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--trim', type=float, default=2)
parser.add_argument('--top_k', type=float, default=2)
parser.add_argument('--gpu_to_use', type=str, default='1')
parser.add_argument('--save_raw_scores', type=str2bool, default='true')
args=parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_to_use

_, reverse_label_encoder, _ = get_label_encoder()
label_clusters = np.load(DATA_DIR + 'clusters_2030.npy')

if not os.path.exists(os.path.join(SUB_DIR, args.model_checkpoint)):
    os.mkdir(os.path.join(SUB_DIR, args.model_checkpoint))

model_path = os.path.join(MODEL_DIR, args.model_checkpoint)
model = load_model(model_path, compile=False)

test_image_dir = os.path.join(DATA_DIR, args.test_image_dir)
sub = pd.read_csv(os.path.join(DATA_DIR, args.subfile))


def predict_and_post_process(image, model, threshold=args.threshold, trim=args.trim, top_k=args.top_k):
    prediction = model.predict(image[np.newaxis,:,:,:])
    lbl_pred = []
    emb_pred = []
    if type(prediction) == 'list':
        lbl_pred, emb_pred = prediction
        lbl_pred, emb_pred = lbl_pred[0], emb_pred[0]
    elif prediction.shape == (1, 7178):
        lbl_pred = prediction[0]
    elif prediction.shape == (1, 1024):
        emb_pred = prediction[0]
    indices = np.where(lbl_pred >= threshold)[0]
    if trim and top_k:
        clusters = [label_clusters[x] for x in indices]
        cluster_indices_counts = pd.Series(clusters).value_counts().to_dict()
        cluster_indices_mapping = {}
        for idx, cluster in zip(indices, clusters):
            if cluster not in cluster_indices_mapping:
                cluster_indices_mapping[cluster] = [(idx, lbl_pred[idx])]
            else:
                cluster_indices_mapping[cluster].append((idx, lbl_pred[idx]))
        for cluster in cluster_indices_mapping:
            cluster_indices_mapping[cluster].sort(key=lambda x: x[1], reverse=True)
            if cluster_indices_counts[cluster] > trim:
                cluster_indices_mapping[cluster] = cluster_indices_mapping[cluster][:top_k]
        trimed_incies = []
        for cluster in cluster_indices_mapping:
            trimed_incies.extend([i[0] for i in cluster_indices_mapping[cluster]])
        indices = trimed_incies
    labels = [reverse_label_encoder[x] for x in indices]
    return {'labels': labels, 'lbl_pred':lbl_pred , 'emb_pred':emb_pred}


lbl_preds = []
emb_preds = []


for idx in tqdm(range(sub.shape[0])):
    image_path = os.path.join(test_image_dir, sub['image_id'][idx] + '.jpg')
    image = read_image(image_path)
    prediction = predict_and_post_process(image, model)
    sub['labels'][idx] = ' '.join(prediction['labels'])
    if args.save_raw_scores:
        lbl_preds.append(prediction['lbl_pred'])
        emb_preds.append(prediction['emb_pred'])

if not os.path.exists(os.path.join(SUB_DIR, args.model_checkpoint)):
    os.mkdir(os.path.join(SUB_DIR, args.model_checkpoint))
sub.to_csv(os.path.join(SUB_DIR, args.model_checkpoint, 'sub_thresh_{}_trim_{}_top_{}.csv'.format(args.threshold, args.trim, args.top_k)), index=False)

if args.save_raw_scores:
    np.save(os.path.join(SUB_DIR, args.model_checkpoint, 'lbl_preds.npy'), np.array(lbl_preds))
    np.save(os.path.join(SUB_DIR, args.model_checkpoint, 'emb_preds.npy'), np.array(emb_preds))

# naive_prior = '/m/01g317 /m/05s2s /m/07j7r'.split()
# sub_with_majority_prior = sub.copy()
# sub_with_majority_prior['labels'] = sub['labels'].map(lambda x: ' '.join(set(naive_prior + x.split())))
# sub_with_majority_prior.to_csv(os.path.join(SUB_DIR, args.model_name, 'sub_thresh_{}_with_major_prior.csv'.format(args.threshold)), index=False)
