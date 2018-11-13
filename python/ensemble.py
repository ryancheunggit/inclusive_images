#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com
import os
import numpy as np
import pandas as pd
from util import get_label_encoder
from CONSTANTS import SUB_DIR, DATA_DIR
from tqdm import tqdm

resnet50_lbl_pred = np.load(os.path.join(SUB_DIR, 'ResNet50_clf_global_avg_pooling/swa_36_45.h5/lbl_preds.npy'))
resnet101_lbl_pred = np.load(os.path.join(SUB_DIR, 'ResNet101_clf_global_avg_pooling/swa_31_40.h5/lbl_preds.npy'))
sub = pd.read_csv(os.path.join(DATA_DIR, 'stage_1_sample_submission.csv'))
_, reverse_label_encoder, _ = get_label_encoder()
label_clusters = np.load(DATA_DIR + 'clusters_2030.npy')
threshold = 0.5
trim = 2
top_k = 2

for sub_idx in tqdm(range(sub.shape[0])):
    lbl_pred = (resnet50_lbl_pred[sub_idx] +  resnet101_lbl_pred[sub_idx]) / 2
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
    # print(len(labels))
    sub.iloc[sub_idx, 1] = ' '.join(labels)

sub.to_csv(os.path.join(SUB_DIR, 'avg_resnet50_resnet101_threshold_{}_trim_{}_top_{}.csv'.format(threshold, trim, top_k)), index=False)
