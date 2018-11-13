#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com
import os
import pickle
from CONSTANTS import DATA_DIR
import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.cluster import AgglomerativeClustering


def get_mi_emb(M):
    ''' reference https://arxiv.org/pdf/1607.05691.pdf '''
    M = M.astype('int64')
    n, m = M.shape
    C = np.dot(M.T, M).todense()
    L = C.diagonal()
    L = np.dot(L.T, L)
    np.fill_diagonal(C, 0)
    C = C / n
    L = L / n**2
    PMI = np.log(C / L)
    PMI[PMI == - np.inf] = 0
    PMI = np.nan_to_num(PMI, 0)
    U, S, V = np.linalg.svd(PMI)
    E = np.multiply(U, np.sqrt(S))
    return {'E': E, 'U': U, 'S': S, 'V': V, 'L': L, 'C': C, 'PMI': PMI}


def cosine_similarity(u, v):
    return np.asscalar(u.dot(v.T) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def cosine_proximity(u, v):
    return np.asscalar(-1 * u.dot(v.T) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def rmse(u, v):
    return np.sqrt(np.mean(np.power(u - v, 2)))


train_info = pickle.load(open(os.path.join(DATA_DIR, 'train_info.pkl'), 'rb'))
train_image_ids, train_labels = train_info['train_image_ids'], train_info['train_labels']
label_name_mapping = pd.read_csv(os.path.join(DATA_DIR,'class-descriptions.csv'))
label_name_mapping = {row['label_code']:row['description'] for _, row in label_name_mapping.iterrows()}
label_encodings = pickle.load(open(os.path.join(DATA_DIR, 'label_encoder.pkl'), 'rb'))
label_encoder, reverse_label_encoder = label_encodings['label_encoder'], label_encodings['reverse_label_encoder']

emb_results = get_mi_emb(train_labels)
E = emb_results['E']
S = emb_results['S']
for k in (128, 256, 512, 1024, 2048, 4096):
    print('embedding dimension == {}'.format(k))
    print('variance explained : ', S[:k].sum() / S.sum() * 100)
    Ek = E[:, :k]
    print('randomly print out some similarity comparision using the embedding')
    for _ in range(10):
        results = []
        t = np.random.randint(train_labels.shape[1])
        for i in range(train_labels.shape[1]):
            results.append((
                i,
                label_name_mapping[reverse_label_encoder[i]],
                cosine_similarity(Ek[t,:].flatten(), Ek[i, :].flatten())
            ))
        results.sort(key=lambda x:x[2], reverse=True)
        pprint(results[:5])
    with open(os.path.join(DATA_DIR, 'label_emb_{}d.pkl'.format(k)), 'wb') as f:
        pickle.dump(Ek, f)

Ek = pickle.load(open(DATA_DIR + 'label_emb_1024d.pkl', 'rb'))
for num_clusters in [128, 256, 512, 1024, 2030]:
    clusters = AgglomerativeClustering(n_clusters=num_clusters, linkage='average', affinity='cosine').fit(Ek)
    np.save(DATA_DIR + 'clusters_{}.npy'.format(num_clusters), clusters.labels_)

# def greedy_reconstruct(idx=1106, norm_threshold = 2, card_threshold = 100, sim_threshold = 0.2, verbose=True, noise=0.):
#     l = train_labels[idx, :].todense()
#     l_ids = np.where(l == 1)[1]
#     labels = [reverse_label_encoder[i] for i in l_ids]
#     h_labels = [label_name_mapping[i] for i in labels]
#     if verbose:
#         print('true labels are: ')
#         print(h_labels)
#     sum_l = l.dot(Ek)
#     if noise > 0:
#         l_norm = np.linalg.norm(sum_l)
#         l_noise =  200 * (np.random.random(sum_l.shape[1]) - 0.5) * l_norm * noise / sum_l.shape[1]
#         print('norm of embedding sum {}, norm of noised version {}, rmse between the two {}'.format(
#             np.linalg.norm(sum_l), np.linalg.norm(l_noise), rmse(sum_l, l_noise) )
#         )
#         sum_l += l_noise
#     pred_h_labels = []
#     pred_l_ids = []
#     while np.linalg.norm(sum_l) >= norm_threshold and len(pred_h_labels) <= card_threshold:
#         results = []
#         for i in range(Ek.shape[0]):
#             if i not in pred_l_ids:
#                 results.append((i, label_name_mapping[reverse_label_encoder[i]], cosine_similarity(Ek[i, :], sum_l)))
#         results.sort(key=lambda x:x[2], reverse=True)
#         best_id, best_label, sim = results[0]
#         if sim <= sim_threshold:
#             break
#         if verbose:
#             print(best_label, sim)
#         v = Ek[best_id, :]
#         sum_l -= v
#         pred_h_labels.append(best_label)
#         pred_l_ids.append(best_id)
#     agreement = len([i for i in h_labels if i in pred_h_labels])
#     true_positive = len(h_labels)
#     pred_positive = len(pred_h_labels)
#     recall = agreement / (true_positive + 1e-11)
#     precision = agreement / (pred_positive + 1e-11)
#     f2score = (1+2**2)*((precision*recall)/(2**2*precision+recall+1e-11))
#     if verbose:
#         print('ag {}, tp {}, pp {}, recall {}, precision {}, f2 {}'.format(
#             agreement, true_positive, pred_positive, recall, precision, f2score)
#         )
#     return {
#         'agreement': agreement,
#         'true_positive': true_positive,
#         'pred_positive': pred_positive,
#         'recall': recall,
#         'precision': precision,
#         'f2score': f2score
#     }
#
# samples = np.random.choice(range(train_labels.shape[0]), size=1000, replace=False)
# Recover = []
# for i in tqdm(samples):
#     r = greedy_reconstruct(idx = i, verbose = False, noise=0.1)
#     Recover.append((i, *r.values()))
#
# print(np.mean([i[-1] for i in Recover]))
