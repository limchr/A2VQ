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
from flaskr.settings import *

from flaskr.glvq import glvq
from flaskr.dbscan import dbscan


'''flask app'''
app = Flask(__name__)






@app.route('/embedding_view_query')
def embedding_view_query():
    ''' this page displays A2VQ labeling interface '''

    # load all dump files
    x_embedding = pkl.load(open(EMBEDDING_FILE, 'rb'))
    view_size = pkl.load(open(VIEW_SIZE_FILE, 'rb'))
    overlap = view_size/4
    labels = pkl.load(open(LABEL_FILE, 'rb'))
    x_embedding_normalized = normalize_features(x_embedding)
    mask_unlabeled = labels == -1
    mask_labeled = labels != -1
    features,groundtruth_labels = get_train_set(FN_LOAD_DATASET(False))

    # load classifier
    cls = pkl.load(open(CLASSIFIER_FILE,'rb'))
    try:
        probas = cls.predict_proba(features)
    except:
        print('can not evaluate probas, using zeros!')
        probas = np.zeros(len(features))

    # query the best view (like it is described in the paper)
    best_views,least_confidences = a2vq_querying(x_embedding_normalized, mask_unlabeled, probas, view_size, overlap)
    # if queried again without any labeled images in that view, the next least confident view is queried:
    if float(config['a2vq_helper']['first_queried_x']) != best_views[0][0] or float(config['a2vq_helper']['first_queried_y']) != best_views[0][1]:
        config['a2vq_helper']['first_queried_x'] = str(best_views[0][0])
        config['a2vq_helper']['first_queried_y'] = str(best_views[0][1])
        config['a2vq_helper']['queried_index'] = str(0)
    best_view = best_views[int(config['a2vq_helper']['queried_index'])]
    # save the queried index in the config file for memorizing the queried view
    config['a2vq_helper']['queried_index'] = str(int(config['a2vq_helper']['queried_index'])+1)
    with open('config.cfg', 'w') as f:
        config.write(f)

    # get mask of samples within view
    mask_queried = filter_embedding(x_embedding_normalized,best_view,(view_size,view_size))
    mask_combined = np.logical_and(mask_queried,mask_unlabeled)
    # normalize this queried samples from 0 to 1 for displaying
    x_embedding_filter_normalized = normalize_features(x_embedding_normalized[mask_combined])

    # if experiment was started, check if it is finished
    started = config['experiment']['started'] == 'True'
    current_duration = 0
    if started:
        current_duration = time.time() - float(config['experiment']['started_timestamp'])
        # abort experiment if number of unlabeled samples is below MIN_UNLABELED or if time was running out
        if len(np.where(mask_unlabeled)[0]) < MIN_UNLABELED or (current_duration > float(config['experiment']['duration'])) or ((config['experiment']['type'] == 'demo') and current_duration > DEMO_DURATION):
            config['experiment']['started'] = 'False'
            config['experiment']['started_timestamp'] = '-1'
            config['participant']['finished_' + config['experiment']['type']] = 'True'
            with open('config.cfg', 'w') as f:
                config.write(f)
            return experiments_frontpage()

    return render_template('embedding_view.html', label_names = np.unique(groundtruth_labels).tolist(), indices = np.where(mask_combined)[0].tolist(), x_embedding = x_embedding_filter_normalized[:,:].tolist(),thumb_dir = THUMBS_DIR_HTTP, started = started, current_duration = int(EXPERIMENTS_DURATION-current_duration))


@app.route('/add_labels', methods = ['POST'])
def add_labels():
    ''' add labels is executed every time a user has labeled samples with A2VQ. This function is called via AJAX asyncronously. '''
    print('add labels')
    print(request.form)
    label = request.form.get('label')
    num_rects = int(request.form.get('num_rects'))
    indices = np.array(request.form.getlist('indices[]'), dtype=int)
    selected = np.array(request.form.getlist('selected[]'), dtype=int)
    label_samples(indices[selected],label)

    # load label file
    labels = pkl.load(open(LABEL_FILE, 'rb'))
    mask_unlabeled = labels == -1
    mask_labeled = labels != -1
    print('UNLABELED SAMPLES: ',len(np.where(mask_unlabeled)[0]))

    # train classifier with new labeled samples
    cls = pkl.load(open(CLASSIFIER_FILE,'rb'))
    features,groundtruth_labels = get_train_set(FN_LOAD_DATASET(False))
    features_test, groundtruth_labels_test = get_test_set(FN_LOAD_DATASET(False))
    train_class = [label for _ in range(len(selected))]
    cls.fit(features[indices[selected]], train_class)
    pkl.dump(cls,open(CLASSIFIER_FILE,'wb'))
    # calculate the classifier's score on test set
    score = cls.score(features, groundtruth_labels)
    score_test = cls.score(features_test, groundtruth_labels_test)

    # load config file to determine if an experiment was started
    config = configparser.ConfigParser()
    config.read('config.cfg')
    started = config['experiment']['started'] == 'True'


    if started:
        # if an experiment was started, all user interactions are saved in the user's experiment csv file
        current_duration = time.time() - float(config['experiment']['started_timestamp'])
        with open(os.path.join('participants/', config['participant']['id'],config['experiment']['type']+'.csv'), "a") as f:
            for select in indices[selected]:
                append = [config['participant']['id'], str(current_duration), str(select), label, groundtruth_labels[select], str(score), str(score_test), str(num_rects)]
                f.write(';'.join(append)+'\n')
        # abort experiment if number of unlabeled samples is below MIN_UNLABELED or if time was running out
        if len(np.where(mask_unlabeled)[0]) < MIN_UNLABELED or (current_duration > float(config['experiment']['duration'])) or ((config['experiment']['type'] == 'demo') and current_duration > DEMO_DURATION):
            config['experiment']['started'] = 'False'
            config['experiment']['started_timestamp'] = '-1'
            config['participant']['finished_'+config['experiment']['type']] = 'True'
            with open('config.cfg', 'w') as f:
                config.write(f)
            return '{"success": "false","open_page": "experiments_frontpage"}'

    return '{"success": "true"}'





#
# Functions that need to be called for preparation
#


from common.helper import get_files_of_type

@app.route('/update_db')
def update_db():
    #pdb.set_trace()
    #update labels file
    current_db = get_image_filenames()
    all_images = get_files_of_type(IMAGE_PATH, 'jpg')
    new_images = get_elements_not_in(all_images, current_db)
    num_samples = len(new_images)
    new_labels = np.zeros(num_samples, dtype=str)
    new_labels_combined = np.vstack((new_images, new_labels)).T
    old_labels = read_csv(LABEL_FILE)
    save_csv(np.vstack((old_labels,new_labels_combined)), LABEL_FILE)

    # calculate features of new images
    new_feats, _, new_img_tuple = feature_extraction_of_arbitrary_image_ds(IMAGE_PATH, new_images)

    # update images dump file
    with open(IMAGES_FILE, 'rb') as f:
        images_file = pkl.load(f)
    new_image_file = np.vstack((images_file, new_img_tuple))
    with open(IMAGES_FILE,'wb') as f:
        pkl.dump(new_image_file, f)
    # write new thumbnails
    for i in range(len(new_img_tuple)):
        write_image(new_img_tuple[i], os.path.join(THUMBS_DIR, '%06d.jpg' % (len(current_db)+i)))


    # save new features file
    with open(FEATURES_FILE, 'rb') as f:
        feature_file = pkl.load(f)
    new_feature_file = np.vstack((feature_file, new_feats))
    with open(FEATURES_FILE,'wb') as f:
        pkl.dump(new_feature_file, f)

    # recalculate embedding
    do_build_embedding()

    return 'updated db with '+str(len(new_feats))+' new images'



@app.route('/delete_db')
def delete_db():
    '''delete label, feature, embedding file and thumbs directory'''
    pass


from common.images import write_image

from common.helper import save_csv
from common.data_handler.create_arbitrary_image_ds import feature_extraction_of_arbitrary_image_ds
@app.route('/init_everything')
def init_everything():
    '''initialize everything (call at first run)'''
    feats, image_list, img_tuple = feature_extraction_of_arbitrary_image_ds(IMAGE_PATH)

    num_samples = len(image_list)
    labels = np.zeros(num_samples, dtype=str)
    label_file = np.vstack((image_list, labels)).T
    save_csv(label_file, LABEL_FILE)

    with open(FEATURES_FILE,'wb') as f:
        pkl.dump(feats, f)


    with open(IMAGES_FILE,'wb') as f:
        pkl.dump(img_tuple, f)
    setup_clean_directory(THUMBS_DIR)
    for i in range(len(img_tuple)):
        write_image(img_tuple[i], os.path.join(THUMBS_DIR, '%06d.jpg' % (i)))




    return 'Done'

@app.route('/reset_labels')
def init_labels():
    ''' init labels file for saving labeled and unlabeled instances. '''
    features = get_train_set(FN_LOAD_DATASET(False))[0]
    labels = np.zeros((features.shape[0]),dtype=object)
    labels = labels -1
    pkl.dump(labels, open(LABEL_FILE,'wb'))
    pkl.dump(DEFAULT_VIEW_SIZE,open(VIEW_SIZE_FILE,'wb'))
    return 'Done'

@app.route('/do_build_embedding')
def do_build_embedding():
    ''' build the embedding to use '''
    with open(FEATURES_FILE, 'rb') as f:
        feature_file = pkl.load(f)
    build_embedding(feature_file)
    return 'Done'


#
# Debugging view of visualization embedding
#
@app.route('/')
@app.route('/embedding_view')
def embedding_view():
    '''display whole embedding'''
    try:
        x_embedding = pkl.load(open(EMBEDDING_FILE,'rb'))
    except:
        return 'can not create embedding. please call first: /init_db /init_labels and /build_embedding'
    return render_template('embedding_view.html', indices = list(range(len(x_embedding))), x_embedding = x_embedding.tolist(), thumb_dir = THUMBS_DIR_HTTP)

@app.route('/embedding_view_partial')
def embedding_view_partial():
    ''' display a predefined view of embedding '''
    try:
        x_embedding = pkl.load(open(EMBEDDING_FILE,'rb'))
    except:
        return 'can not create embedding. please call first: /init_db /init_labels and /build_embedding'

    mask = filter_embedding(x_embedding,(0.2,0.2),(0.6,0.6))
    x_embedding_filter_normalized = normalize_features(x_embedding[mask])

    return render_template('embedding_view.html', indices = np.where(mask)[0].tolist(), x_embedding = x_embedding_filter_normalized.tolist(),thumb_dir = THUMBS_DIR_HTTP)

from flaskr.data_handler import *

@app.route('/embedding_view_labeled')
def embedding_view_labeled():
    '''display whole embedding'''
    try:
        x_embedding = pkl.load(open(EMBEDDING_FILE,'rb'))
    except:
        return 'can not create embedding. please call first: /init_db /init_labels and /build_embedding'
    x_embedding_selection = x_embedding[get_labeled_i()]
    return render_template('embedding_view.html', indices = list(range(len(x_embedding_selection))),
                           x_embedding = x_embedding_selection.tolist(), thumb_dir = THUMBS_DIR_HTTP)

@app.route('/embedding_view_unlabeled')
def embedding_view_unlabeled():
    '''display whole embedding'''
    try:
        x_embedding = pkl.load(open(EMBEDDING_FILE, 'rb'))
    except:
        return 'can not create embedding. please call first: /init_db /init_labels and /build_embedding'
    x_embedding_selection = x_embedding[get_unlabeled_i()]
    return render_template('embedding_view.html', indices=list(range(len(x_embedding_selection))),
                           x_embedding=x_embedding_selection.tolist(), thumb_dir=THUMBS_DIR_HTTP)

    #x_embedding_labeled_normalized = normalize_features(x_embedding_labeled)



if __name__ == '__main__':
    # working_dir = os.path.realpath(__file__)[:-19]
    # sys.path.append(working_dir)
    # os.chdir(working_dir)
    # print('change dir to ' + working_dir)
    app.run(host= '0.0.0.0')
