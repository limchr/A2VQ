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

import numpy as np
import pandas as pd
import os

from flaskr.settings import *
from flaskr.helper import normalize_features

#from common.data_handler.create_arbitrary_image_ds import create_arbitrary_image_ds
from common.data_handler.load_arbitrary_image_ds import load_arbitrary_image_ds
from common.helper import read_csv, save_csv


def update_everything():
    '''update everything because new data is available for labeling'''
    extract_features()
    build_embedding()
    create_thumbs()

def extract_features():
    '''extract features of newly recorded images (which are not already in feature set) and extend feature set'''
    pass

def build_embedding(x):
    '''build embedding and save as dump for fast loading'''
    x_embedding = EMBEDDING_FUN(x)
    x_embedding_normalized = normalize_features(x_embedding)

    with open(EMBEDDING_FILE, 'wb') as f:
        pkl.dump(x_embedding_normalized, f)


def create_thumbs():
    '''create thumbnails not already there of all images for showing over HTTP'''
    pass


def ds_robot(load_images = False):
    '''load robot image data set'''
    rtn = load_arbitrary_image_ds('/homes/climberg/src/min_workspace/data/', load_images)
    return rtn


def init_classifier():
    from machine_learning_models.glvq import glvq
    cls = glvq()
    with open(CLASSIFIER_FILE, 'wb') as f:
        pkl.dump(cls,f)

def get_classifier():
    with open(CLASSIFIER_FILE, 'rb') as f:
        cls = pkl.load(f)
    return cls

def get_features():
    with open(EMBEDDING_FILE, 'rb') as f:
        return pkl.load(FEATURES_FILE)


def get_image_filenames():
    '''return the image filenames of the current data base'''
    return read_csv(LABEL_FILE)[:,0]

def get_labels():
    '''return all labels'''
    return read_csv(LABEL_FILE)[:, 1]


def get_unlabeled_i():
    '''return all unlabeled_indices'''
    return np.where(pd.isnull(read_csv(LABEL_FILE)[:, 1]))[0]

def get_labeled_i():
    '''return all labeled indices'''
    return np.where(np.invert(pd.isnull(read_csv(LABEL_FILE)[:, 1])))[0]

def label_samples(indices, label):
    '''helper function for fast writing new labels in label file '''
    labels = read_csv(LABEL_FILE)
    labels[indices,1] = label
    save_csv(labels, LABEL_FILE)


if __name__ == '__main__':
    labels = get_labels()
    get_image_filenames()