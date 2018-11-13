#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com
import os
import pickle
import pandas as pd
import numpy as np
from CONSTANTS import DATA_DIR, NUM_CLASSES
from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix
tqdm.pandas()

classes = pd.read_csv(os.path.join(DATA_DIR, 'classes-trainable.csv'))
label_name_mapping = pd.read_csv(os.path.join(DATA_DIR,'class-descriptions.csv'))
train = pd.read_csv(os.path.join(DATA_DIR,'train_human_labels.csv'))
val = pd.read_csv(os.path.join(DATA_DIR,'tuning_labels.csv'), names=['ImageID', 'LabelName'])

# --- LABEL ENCODING ---
# create and save two way mapping between label_code and int number
reverse_label_encoder = classes.reset_index()['label_code'].to_dict()
label_encoder = {v:k for k, v in reverse_label_encoder.items()}
with open(os.path.join(DATA_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(obj={'label_encoder': label_encoder, 'reverse_label_encoder': reverse_label_encoder}, file=f)

# --- PREPRE FILENAME - Y MAPPING ---
# only keep trainable label
train = train.loc[train.LabelName.isin(classes.label_code)]

# --- SAVE TRAINING/VALIDATION - IMAGE_IDS AND TRAINING/VALIDATION - LABELS
trainp = train.groupby('ImageID')['LabelName'].progress_apply(lambda x: ' '.join(x)).reset_index()
trainp['num_labels'] = trainp['LabelName'].progress_map(lambda x: pd.Series(x.split()).map(label_encoder).tolist())
trainIDs = trainp['ImageID'].tolist()
trainLabels = trainp['num_labels'].tolist()
trainLabelsMatrix = lil_matrix((len(trainLabels), NUM_CLASSES), dtype='int8')
for row_idx, labels in tqdm(enumerate(trainLabels), total=len(trainLabels)):
    for label_loc in labels:
        trainLabelsMatrix[row_idx, label_loc] = 1
trainLabels = csr_matrix(trainLabelsMatrix)
with open(os.path.join(DATA_DIR, 'train_info.pkl'), 'wb') as f:
    pickle.dump(obj={'train_image_ids': trainIDs, 'train_labels': trainLabels}, file=f)

# prepare a 10% subset of training data
np.random.seed(42)
indices = np.random.choice(len(trainIDs), size=170000, replace=False)
with open(os.path.join(DATA_DIR, 'train_subset_info.pkl'), 'wb') as f:
    pickle.dump(
        obj={
            'image_ids': [trainIDs[idx] for idx in indices],
            'labels': trainLabels[indices],
            'indices':indices
        },
        file=f
    )

val['num_labels'] = val['LabelName'].progress_map(lambda x: pd.Series(x.split()).map(label_encoder).tolist())
valIDs = val['ImageID'].tolist()
valLabels = val['num_labels'].tolist()
valLabelsMatrix = lil_matrix((len(valLabels), NUM_CLASSES), dtype='int8')
for row_idx, labels in tqdm(enumerate(valLabels), total=len(valLabels)):
    for label_loc in labels:
        valLabelsMatrix[row_idx, label_loc] = 1
valLabels = csr_matrix(valLabelsMatrix)
with open(os.path.join(DATA_DIR, 'val_info.pkl'), 'wb') as f:
    pickle.dump(obj={'val_image_ids': valIDs, 'val_labels': valLabels}, file=f)
