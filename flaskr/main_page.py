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


import sys
import os


import configparser
import time
import pdb
from scipy.spatial.distance import cdist

from flask import Flask
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

from flaskr.helper import *
from flaskr.intelligent_labeling import *


import flaskr.functions as fun
import flaskr.db_file_interface as db_interface
#import flaskr.db_ros_interface as db_interface







'''flask app'''
app = Flask(__name__)




#
# Debugging view of visualization embedding
#
@app.route('/')
@app.route('/embedding_view')
def embedding_view():
    '''display whole embedding'''
    try:
        x_embedding = db_interface.load_embedding()
        indices = db_interface.get_indices()
    except:
        return 'can not create embedding. please call first: /init'
    return render_template('embedding_view.html', indices =indices.tolist(), x_embedding = x_embedding.tolist(), thumb_dir = THUMBS_DIR_HTTP)


@app.route('/query')
def embedding_view_query():
    ''' this page displays A2VQ labeling interface '''

    # load all dump files
    x_embedding = db_interface.load_embedding()
    view_size = 0.2
    overlap = view_size/4
    feats = db_interface.load_features()
    indices = db_interface.get_indices()

    mask_unlabeled = get_unlabeled_mask()

    # load classifier
    probas = db_interface.classifier_predict_proba(feats)

    # query the best view (like it is described in the paper)
    best_views,least_confidences = a2vq_querying(x_embedding, mask_unlabeled, probas, view_size, overlap)
    best_view = best_views[0]

    # get mask of samples within view
    mask_queried = filter_embedding(x_embedding,best_view,(view_size,view_size))
    mask_combined = np.logical_and(mask_queried,mask_unlabeled)
    # normalize this queried samples from 0 to 1 for displaying
    x_embedding_filter_normalized = normalize_features(x_embedding[mask_combined])
    indices_filtered = indices[mask_combined]


    return render_template('embedding_view.html', indices = indices_filtered.tolist(), x_embedding = x_embedding_filter_normalized[:,:].tolist(),thumb_dir = THUMBS_DIR_HTTP)


@app.route('/add_labels', methods = ['POST'])
def add_labels():
    ''' add labels is executed every time a user has labeled samples with A2VQ. This function is called via AJAX asyncronously. '''
    print('add labels')
    print(request.form)
    label = request.form.get('label')
    selected = np.array(request.form.getlist('selected[]'), dtype=str)

    indices_selected = np.where(np.array([x in selected for x in db_interface.get_indices()]))[0]

    db_interface.update_labels(indices_selected,label)

    # train classifier with new labeled samples
    feats = db_interface.load_features()[indices_selected]
    db_interface.classifier_partial_fit(feats,[label] * len(selected))

    return '{"success": "true"}'




@app.route('/init')
def init():
    '''initialize everything (call at first run)'''
    db_interface.init()

    create_thumbs()

    x = db_interface.load_features()
    emb = build_embedding(x)
    db_interface.update_embedding(emb)

    return 'Done'

@app.route('/embedding_view_partial')
def embedding_view_partial():
    ''' display a predefined view of embedding '''
    try:
        x_embedding = db_interface.load_embedding()
        indices = db_interface.get_indices()
    except:
        return 'can not create embedding. please call first: /init'

    mask = filter_embedding(x_embedding,(0.2,0.2),(0.6,0.6))
    x_embedding_filter_normalized = normalize_features(x_embedding[mask])
    indices_filter = indices[mask]
    return render_template('embedding_view.html', indices = indices_filter.tolist(), x_embedding = x_embedding_filter_normalized.tolist(),thumb_dir = THUMBS_DIR_HTTP)

from flaskr.functions import *

@app.route('/labeled')
def embedding_view_labeled():
    '''display only labeled samples'''
    try:
        x_embedding = db_interface.load_embedding()
        indices = db_interface.get_indices()
    except:
        return 'can not create embedding. please call first: /init'
    mask = fun.get_labeled_mask()
    x_embedding_filter_normalized = normalize_features(x_embedding[mask])
    indices_filter = indices[mask]

    return render_template('embedding_view.html', indices = indices_filter.tolist(), x_embedding = x_embedding_filter_normalized.tolist(),thumb_dir = THUMBS_DIR_HTTP)

@app.route('/unlabeled')
def embedding_view_unlabeled():
    '''display only unlabeled samples'''
    try:
        x_embedding = db_interface.load_embedding()
        indices = db_interface.get_indices()
    except:
        return 'can not create embedding. please call first: /init'
    mask = fun.get_unlabeled_mask()
    x_embedding_filter_normalized = normalize_features(x_embedding[mask])
    indices_filter = indices[mask]

    return render_template('embedding_view.html', indices = indices_filter.tolist(), x_embedding = x_embedding_filter_normalized.tolist(),thumb_dir = THUMBS_DIR_HTTP)



if __name__ == '__main__':
    # working_dir = os.path.realpath(__file__)[:-19]
    # sys.path.append(working_dir)
    # os.chdir(working_dir)
    # print('change dir to ' + working_dir)
    app.run(host= '0.0.0.0')
