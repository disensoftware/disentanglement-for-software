import torch
import json
import os
import sys
import numpy as np 
import pickle
from collections import Counter
import random
import copy
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score
    

def calculate_purity_scores(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 


def calculate_shen_f_score(y_true, y_pred):
    def get_f_score(i, j, n_i_j, n_i, n_j):
        recall = n_i_j / n_i
        precision = n_i_j / n_j
        if recall == 0 and precision == 0:
            f_score = 0.
        else:
            f_score = 2 * recall * precision / (recall + precision)
        return f_score
    
    y_true_cnt = dict(Counter(y_true))
    y_pred_cnt = dict(Counter(y_pred))
    y_pred_dict = dict()
    for i, val in enumerate(y_pred):
        if y_pred_dict.get(val, None) == None:
            y_pred_dict[val] = dict()
        if y_pred_dict[val].get(y_true[i], None) == None:
            y_pred_dict[val][y_true[i]] = 0
        y_pred_dict[val][y_true[i]] += 1
    shen_f_score = 0.
    for i, val_i in y_true_cnt.items():
        f_list = []
        for j, val_j in y_pred_cnt.items():
            f_list.append(get_f_score(i, j, y_pred_dict[j].get(i, 0), val_i, val_j))
        shen_f_score += max(f_list) * y_true_cnt[i] / len(y_true)
    return shen_f_score


def compare(predicted_labels, truth_labels, metric):
    if metric == 'purity':
        purity_scores = []
        for y_true, y_pred in zip(truth_labels, predicted_labels):
            purity_scores.append(calculate_purity_scores(y_true, y_pred))
        return sum(purity_scores)/len(purity_scores)
    elif metric == 'NMI':
        NMI_scores = []
        for y_true, y_pred in zip(truth_labels, predicted_labels):
            NMI_scores.append(normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic'))
        return sum(NMI_scores)/len(NMI_scores)
    elif metric == 'ARI':
        ARI_scores = []
        for y_true, y_pred in zip(truth_labels, predicted_labels):
            ARI_scores.append(metrics.adjusted_rand_score(y_true, y_pred))
        return sum(ARI_scores)/len(ARI_scores)
    elif metric == "shen_f":
        f_scores = []
        for y_true, y_pred in zip(truth_labels, predicted_labels):
            f_scores.append(calculate_shen_f_score(y_true, y_pred))
        return sum(f_scores)/len(f_scores)
