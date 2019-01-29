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
working_dir = os.path.realpath(__file__)[:-19]
sys.path.append(working_dir)
os.chdir(working_dir)
print('change dir to '+working_dir)


import configparser
import time
import pdb
from scipy.spatial.distance import cdist

from flask import Flask
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

from flaskr.intelligent_labeling import *
from flaskr.settings import *

from flaskr.helper import setup_clean_directory
from flaskr.helper import normalize_features
from flaskr.glvq import glvq
from flaskr.dbscan import dbscan


'''flask app'''
app = Flask(__name__)





def label_samples(indices, label):
    '''helper function for fast writing new labels in label file '''
    labels = pkl.load(open(LABEL_FILE,'rb'))
    labels[indices] = label
    pkl.dump(labels, open(LABEL_FILE,'wb'))



#############################
# Functions for user study
#############################

@app.route('/')
@app.route('/new_participant')
def new_participant():
    ''' show the form for register a new participant'''
    return render_template('new_participant.html')

@app.route('/new_participant', methods = ['POST'])
def new_participantform():
    ''' take the ID of the new created participant and create a configfile which has the actual '''
    config = configparser.ConfigParser()
    config['participant'] = {'id': request.form['id'],'finished_a': False, 'finished_b': False, 'finished_c': False, 'finished_d': False, 'finished_demo': False}
    config['experiment'] = {'started': False, 'type': -1, 'started_timestamp': -1, 'duration': EXPERIMENTS_DURATION}
    config['a2vq_helper'] = {'first_queried_x': -1, 'first_queried_y': -1, 'queried_index': 0}
    with open('config.cfg', 'w') as f:
        config.write(f)

    setup_clean_directory(os.path.join('participants/',request.form['id']))

    return experiments_frontpage()


@app.route('/experiments_frontpage')
def experiments_frontpage():
    ''' frontpage showing buttons for starting particular experiments '''
    config = configparser.ConfigParser()
    config.read('config.cfg')

    return render_template('experiments_frontpage.html', finished_a=config['participant']['finished_a']=='True', finished_b=config['participant']['finished_b']=='True', finished_c=config['participant']['finished_c']=='True', finished_d=config['participant']['finished_d']=='True',finished_demo=config['participant']['finished_demo']=='True')


@app.route('/start_experiment')
def start_experiment():
    ''' starts a particular experiment and setting the config files. initializes an untrained classifier for training.'''

    # update config file
    config = configparser.ConfigParser()
    config.read('config.cfg')
    type = request.args.get('type')
    # if the experiment was completed before, a warning is displayed (a force tag is neccesary to surpass this)
    if config['participant']['finished_'+type] == 'True' and not (request.args.get('force_restart') == 'True'):
        return 'you already have finished this experiment. if you really want a restart use: [servername]:5000/start_experiment?type='+type+'&force_restart=True'
    config['experiment']['started_timestamp'] = str(time.time())
    config['experiment']['started'] = 'True'
    config['experiment']['type'] = type
    with open('config.cfg', 'w') as f:
        config.write(f)

    # resetting label file
    init_labels()

    # resetting experiment file if it exists
    if os.path.isfile(os.path.join('participants/',config['participant']['id'],type+'.csv')):
        os.remove(os.path.join('participants/',config['participant']['id'],type+'.csv'))

    # classifier is initialized and dumped
    glvq_kwargs = {'max_prototypes_per_class':9999999999, 'learning_rate':50, 'strech_factor':1}
    cls = glvq(**glvq_kwargs)
    pkl.dump(cls, open(CLASSIFIER_FILE, 'wb'))

    # start the particular experiment by returning their page
    if type == 'a':
        return embedding_view_query()
    elif type == 'demo':
        return embedding_view_query()
    else:
        return classic_label_view()





@app.route('/embedding_view_query')
def embedding_view_query():
    ''' this page displays A2VQ labeling interface '''
    config = configparser.ConfigParser()
    config.read('config.cfg')

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




@app.route('/classic_label_view')
def classic_label_view():
    ''' this label view shows a single sample and enables labeling it with use of a dropdown menu'''
    config = configparser.ConfigParser()
    config.read('config.cfg')
    started = config['experiment']['started'] == 'True'
    current_duration = 0

    labels = pkl.load(open(LABEL_FILE, 'rb'))
    mask_unlabeled = labels == -1
    mask_labeled = np.logical_and(labels != -1, labels != 'skipped')

    features,groundtruth_labels = get_train_set(FN_LOAD_DATASET(False))
    features_test, groundtruth_labels_test = get_test_set(FN_LOAD_DATASET(False))
    cls = pkl.load(open(CLASSIFIER_FILE,'rb'))

    labeled_index = request.args.get('index')
    label = request.args.get('label')

    if (labeled_index is not None) and (label is not None): # train new labeled instance
        labeled_index = int(labeled_index)
        print('label single image: '+str(labeled_index))
        if label != 'skipped': # train new sample
            cls.fit([features[labeled_index]], [label])
        else: #dbqe approach to exclude ambiguous samples from querying
            unlabeled_is = np.where(mask_unlabeled)[0]
            unlabeled_x = features[mask_unlabeled]
            distances_to_unrecognizable = cdist([features[labeled_index]], unlabeled_x)[0]
            # distances_filter = distances_to_unrecognizable < max_dist
            ex_inds = np.argsort(distances_to_unrecognizable)[:20] # max exclude 20 samples

            dists = cdist(unlabeled_x[ex_inds], unlabeled_x[ex_inds], 'euclidean')
            delete_im = dbscan.DBQE(unlabeled_x[ex_inds], 0, 50, 3, dists=dists)
            delete_im[0] = True  # always delete unrecognizable sample
            delete_i = ex_inds[delete_im]
            print('EXCLUDED '+str(len(delete_i))+' possible ambiguous samples from querying')
            labels[unlabeled_is[delete_i]] = 'skipped'

        # save new trained classifier
        pkl.dump(cls, open(CLASSIFIER_FILE, 'wb'))


        score = cls.score(features, groundtruth_labels)
        score_test = cls.score(features_test, groundtruth_labels_test)
        label_samples([labeled_index],label)

        # check if experiment has finished
        if started:
            current_duration = time.time() - float(config['experiment']['started_timestamp'])
            with open(os.path.join('participants/', config['participant']['id'], config['experiment']['type'] + '.csv'),"a") as f:
                append = [config['participant']['id'], str(current_duration), str(labeled_index), label, groundtruth_labels[labeled_index],str(score),str(score_test)]
                f.write(';'.join(append) + '\n')
            if current_duration > float(config['experiment']['duration']):
                config['experiment']['started'] = 'False'
                config['experiment']['started_timestamp'] = '-1'
                config['participant']['finished_' + config['experiment']['type']] = 'True'
                with open('config.cfg', 'w') as f:
                    config.write(f)

                return experiments_frontpage()

    #
    # classical querying approaches
    #
    if(config['experiment']['type'] == 'b'):
        # uncertainty sampling
        probas = cls.predict_proba(features[mask_unlabeled])
        if(not probas.any()):
            min_i_unlabeled = random.randint(0,len(probas)-1)
        else:
            min_i_unlabeled = np.argmin(probas)
    elif(config['experiment']['type'] == 'c'):
        # random sampling
        min_i_unlabeled = random.randint(0, len(features[mask_unlabeled]) - 1)
    elif(config['experiment']['type'] == 'd'):
        # query by commitee
        min_i_unlabeled = None
        if hasattr(cls,'x'):
            min_i_unlabeled = qbc_querying(cls.x,cls.y,features[mask_unlabeled],1)
        if min_i_unlabeled is None:
            min_i_unlabeled = random.randint(0, len(features[mask_unlabeled]) - 1)
        else:
            min_i_unlabeled = min_i_unlabeled[0]
    min_i = np.array(list(range(len(features))))[mask_unlabeled][min_i_unlabeled]

    return render_template('classic_label_view.html', label_names = np.unique(groundtruth_labels).tolist(), queried = min_i,thumb_dir = THUMBS_DIR_HTTP, started = started, current_duration = int(EXPERIMENTS_DURATION-current_duration))




#
# Functions that need to be called for preparation
#

@app.route('/init_db')
def init_db():
    ''' save data set as thumbnails in static directory, for loading by the browser '''
    from common.images import write_image
    dataset = get_train_set(FN_LOAD_DATASET(True))
    setup_clean_directory(THUMBS_DIR)
    for i in range(len(dataset[2])):
        write_image(dataset[2][i], os.path.join(THUMBS_DIR, '%06d.jpg' % (i)))
    return 'Done'

@app.route('/init_labels')
def init_labels():
    ''' init labels file for saving labeled and unlabeled instances. '''
    features = get_train_set(FN_LOAD_DATASET(False))[0]
    labels = np.zeros((features.shape[0]),dtype=object)
    labels = labels -1
    pkl.dump(labels, open(LABEL_FILE,'wb'))
    pkl.dump(DEFAULT_VIEW_SIZE,open(VIEW_SIZE_FILE,'wb'))
    return 'Done'

@app.route('/build_embedding')
def build_embedding():
    ''' build the embedding to use '''
    features = get_train_set(FN_LOAD_DATASET(False))[0]
    x_embedding = TSNE(n_components=2,random_state=42).fit_transform(features)
    pkl.dump(x_embedding,open(EMBEDDING_FILE,'wb'))
    return 'Done'




#
# Debugging view of visualization embedding
#

@app.route('/embedding_view')
def embedding_view():
    '''display whole embedding'''
    try:
        x_embedding = pkl.load(open(EMBEDDING_FILE,'rb'))
    except:
        return 'can not create embedding. please call first: /init_db /init_labels and /build_embedding'

    x_embedding_normalized = normalize_features(x_embedding)
    return render_template('embedding_view.html', indices = list(range(len(x_embedding_normalized))), x_embedding = x_embedding_normalized[:,:].tolist(), thumb_dir = THUMBS_DIR_HTTP)

@app.route('/embedding_view_partial')
def embedding_view_partial():
    ''' display a predefined view of embedding '''
    x_embedding = pkl.load(open('embedding.pkl','rb'))

    mask = filter_embedding(x_embedding,(0.5,0.5),(0.2,0.2))
    x_embedding_filter_normalized = normalize_features(x_embedding[mask])

    return render_template('embedding_view.html', indices = np.where(mask)[0].tolist(), x_embedding = x_embedding_filter_normalized[:,:].tolist(),thumb_dir = THUMBS_DIR_HTTP)

@app.route('/embedding_view_labeled')
def embedding_view_labeled():
    ''' display only the labeled instances '''
    x_embedding = pkl.load(open('embedding.pkl','rb'))
    labels = pkl.load(open('/hri/storage/user/climberg/datasets/outdoor/active_label_ui/labels.pkl','rb'))

    x_embedding_normalized = normalize_features(x_embedding)
    mask = labels != -1
    x_embedding_labeled = x_embedding_normalized[mask]

    x_embedding_labeled_normalized = normalize_features(x_embedding_labeled)

    return render_template('embedding_view.html', indices = np.where(mask)[0].tolist(), x_embedding = x_embedding_labeled_normalized[:,:].tolist(),thumb_dir = THUMBS_DIR_HTTP)




if __name__ == '__main__':
    # print(build_embedding())
    app.run(host= '0.0.0.0')
