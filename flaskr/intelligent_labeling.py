#!/usr/bin/env python
#
# Copyright (C) 2019
# Christian Limberg
# Centre of Excellence Cognitive Interaction Technology (CITEC)
# Bielefeld University
#
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
# and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#


import numpy as np
import pickle as pkl
import random
import pdb
import math

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.manifold import TSNE
from sklearn import tree

from flaskr.helper import normalize_features


def get_train_set(args):
    '''extracts trainset from dataset with constant random_state'''
    ret = train_test_split(*args, test_size = 0.2, random_state = 42)
    return ret[0:len(ret):2]
def get_test_set(args):
    '''extracts testset from dataset with constant random_state'''
    ret = train_test_split(*args, test_size = 0.2, random_state = 42)
    return ret[1:len(ret):2]




def a2vq_querying(x_embedding_normalized, mask_unlabeled, probas, view_size, overlap):
    ''' a2vq querying as proposed in paper '''
    best_view = (-1,-1)
    least_confidence = -999999999999


    view_range = np.arange(0, 1 - view_size + overlap, overlap)
    print('VIEW_RANGES: ', view_range)
    num_samples = np.zeros((len(view_range),len(view_range)))
    mean_confidence = np.zeros((len(view_range),len(view_range)))

    for i,x in enumerate(view_range):
        for j, y in enumerate(view_range):
            mask = np.logical_and(x_embedding_normalized[:, 0] > x, x_embedding_normalized[:, 0] < x+view_size)
            mask = np.logical_and(mask, x_embedding_normalized[:, 1] > y)
            mask = np.logical_and(mask, x_embedding_normalized[:, 1] < y+view_size)
            fused_mask = np.logical_and(mask,mask_unlabeled)
            probas_temp = probas[fused_mask]

            mean_confidence[i,j] = np.mean([(1-cpu) for cpu in probas_temp])
            num_samples[i,j] = len(probas_temp)

    num_samples_ratio = num_samples / np.max(num_samples)

    mean_confidence = np.nan_to_num(mean_confidence)

    cost_map = mean_confidence * num_samples_ratio

    max_inds = cost_map.flatten().argsort()[::-1]


    max_views = np.array([np.unravel_index(ind, cost_map.shape) for ind in max_inds])
    max_ranges = [(view_range[x],view_range[y]) for x,y in max_views]
    max_costs = cost_map.flatten()[max_inds]

    return max_ranges, max_costs

def qbc_querying(x_train,y_train,x_query,batch_size):
    ''' implementation of qbc with vote disagreement and several classifers'''
    def _vote_disagreement(votes):
        ''' using vote entropy to measure disagreement '''
        ret = []
        for candidate in votes:
            ret.append(0.0)
            lab_count = {}
            for lab in candidate:
                lab_count[lab] = lab_count.setdefault(lab, 0) + 1
            for lab in lab_count.keys():
                ret[-1] -= lab_count[lab] / votes.shape[1] * math.log(float(lab_count[lab]) / votes.shape[1])
        return ret

    try:
        # train offline models
        logistic_regression = SGDClassifier(loss='log')
        logistic_regression.fit(x_train, y_train)

        svm = SVC(kernel='linear', probability=False)
        svm.fit(x_train, y_train)

        tree = DecisionTreeClassifier()
        tree.fit(x_train, y_train)

        # prediction
        prediction_svm = svm.predict(x_query)
        prediction_tree = tree.predict(x_query)
        prediction_logistic_regression = logistic_regression.predict(x_query)

        # calculate vote disagreement
        preds = np.vstack((prediction_svm, prediction_tree, prediction_logistic_regression)).transpose()
        scores = _vote_disagreement(preds)
        max_i = np.argsort(scores)[::-1]
    except:
        print('ERROR: can not train qbc classifiers')
        return None
    return max_i[:batch_size]



def filter_embedding(x_embedding, anchor_point, selection_size):
    ''' get boolean mask of samples within a specified selection bb rect (view)'''
    x_embedding_normalized = normalize_features(x_embedding)
    mask = np.logical_and(x_embedding_normalized[:,0] > anchor_point[0], x_embedding_normalized[:,0] < anchor_point[0]+selection_size[0])
    mask = np.logical_and(mask, x_embedding_normalized[:,1] > anchor_point[1])
    mask = np.logical_and(mask, x_embedding_normalized[:, 1] < anchor_point[1]+selection_size[1])
    return mask




def uncertainty_sampling_view_query(x_embedding_normalized, mask_unlabeled, probas, view_size, overlap):
    ''' obsolete function for comparing a2vq compared to uncertainty sampling within the visualisation'''
    unlabeled_x_embedding_normalized = x_embedding_normalized[mask_unlabeled]
    unlabeled_probas = probas[mask_unlabeled]
    min_i = np.argmin(unlabeled_probas)
    min_x_embedding = unlabeled_x_embedding_normalized[min_i]
    return min_x_embedding - (view_size/2), 0


#
# obsolete functions
# 

def random_sampling_view_query(x_embedding_normalized, mask_unlabeled, probas, view_size, overlap):
    ''' obsolete function for comparing a2vq compared to random sampling within the visualisation'''
    unlabeled_x_embedding_normalized = x_embedding_normalized[mask_unlabeled]
    unlabeled_probas = probas[mask_unlabeled]
    rand_i = random.randint(0,len(unlabeled_probas)-1)
    rand_x_embedding = unlabeled_x_embedding_normalized[rand_i]
    return rand_x_embedding - (view_size/2), 0

def do_build_embedding(FN_LOAD_DATASET, EMBEDDING_FILE):
    '''obsolete function for building embedding '''
    features = FN_LOAD_DATASET()[0]
    x_embedding = TSNE(n_components=2,random_state=42).fit_transform(features)
    pkl.dump(x_embedding,open(EMBEDDING_FILE,'wb'))

def do_init_labels(FN_LOAD_DATASET, LABEL_FILE):
    '''obsolete function for init labels '''
    features = FN_LOAD_DATASET(False)[0]
    labels = np.zeros((features.shape[0]),dtype=object)
    labels = labels -1
    pkl.dump(labels, open(LABEL_FILE,'wb'))

def iui_query_old(x_embedding_normalized, mask_unlabeled, probas, view_size, overlap):
    ''' testing querying method '''
    best_view = (-1,-1)
    least_confidence = 999999999999

    for x in np.arange(0, 1 - view_size + overlap, overlap):
        for y in np.arange(0, 1 - view_size + overlap, overlap):
            mask = np.logical_and(x_embedding_normalized[:, 0] > x, x_embedding_normalized[:, 0] < x+view_size)
            mask = np.logical_and(mask, x_embedding_normalized[:, 1] > y)
            mask = np.logical_and(mask, x_embedding_normalized[:, 1] < y+view_size)
            fused_mask = np.logical_and(mask,mask_unlabeled)
            probas_temp = probas[fused_mask]
            if len(probas_temp) < 10:
                print('view has lesser than 10 samples, skipping',x,y)
                continue
            mean_probas = probas_temp.mean()-len(probas_temp)
            if mean_probas < least_confidence:
                least_confidence = mean_probas
                best_view = (x,y)
    return best_view, least_confidence

def count_number_of_user_actions(X,Y):
    '''obsolete function to determin necessary user interactions while using A2VQ
    for comparing with classical labeling approaches. The function traines a tree to determine
    the neccessary number of bb rectancles which are neccessary to label a specific view
    (the number of rects are related to the number of leave nodes)
    '''
    clf = tree.DecisionTreeClassifier(max_depth=None)
    clf = clf.fit(X, Y)

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right


    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    return len(np.where(is_leaves)[0])