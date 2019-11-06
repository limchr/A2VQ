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

from sklearn.manifold import TSNE

from flaskr.helper import normalize_features





def embedding_tsne(x, y=None):
    x_embedding = TSNE(n_components=2, random_state=42).fit_transform(x)
    return x_embedding

#todo
def embedding_umap(x, y=None):
    pass


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


def filter_embedding(x_embedding, anchor_point, selection_size):
    ''' get boolean mask of samples within a specified selection bb rect (view)'''
    x_embedding_normalized = normalize_features(x_embedding)
    mask = np.logical_and(x_embedding_normalized[:,0] > anchor_point[0], x_embedding_normalized[:,0] < anchor_point[0]+selection_size[0])
    mask = np.logical_and(mask, x_embedding_normalized[:,1] > anchor_point[1])
    mask = np.logical_and(mask, x_embedding_normalized[:, 1] < anchor_point[1]+selection_size[1])
    return mask



