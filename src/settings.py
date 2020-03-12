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


import os
from a2vq.src.helper import create_directory_if_not_defined

from sklearn.manifold import TSNE
def embedding_tsne(x, y=None):
    x_embedding = TSNE(n_components=2, random_state=42).fit_transform(x)
    return x_embedding

#todo
def embedding_umap(x, y=None):
    pass


''' default view size in normalized visualization space '''
DEFAULT_VIEW_SIZE = 0.25

DEFAULT_OVERLAP = DEFAULT_VIEW_SIZE/4

''' experiment breaks when there are lesser than MIN_UNLABELED unlabeled samples '''
MIN_UNLABELED = 10

''' function for evaluating the embedding '''
EMBEDDING_FUN = embedding_tsne

''' define a name for the data set here, this is used to load e.g. thumbnails from the static directory'''
FILE_NAME_PREFIX = 'robot'

#
# CHANGE THIS BEFORE RUNNING
#

# change this to an empty directory on your hard drive
ROOT_DB_BASE_PATH = '/media/fast/climberg/test/a2vq'
# change this to a directory containing images for vizualizing
IMAGE_PATH = '/media/fast/climberg/test/images'
#
# /CHANGE THIS BEFORE RUNNING
#

DB_BASE_PATH = os.path.join(ROOT_DB_BASE_PATH,FILE_NAME_PREFIX)
EXPORT_PATH = os.path.join(DB_BASE_PATH,'export')


''' the output files used for a2vq operation are saved in DUMP_PATH'''
DUMP_PATH = os.path.join(DB_BASE_PATH,'dump')
''' when exporting a dataset, first step is to save all labeled samples class-wise to do a manual cleanup step'''
CLASS_WISE_DB_PATH = os.path.join(DB_BASE_PATH,'class_wise')
''' here, the db file is saved'''
DB_FILE = os.path.join(DB_BASE_PATH,'objs.db')


# define several paths (may move this to db_file_interface)
FEATURES_FILE = os.path.join(DUMP_PATH,'features.pkl')
IMAGES_FILE = os.path.join(DUMP_PATH,'images.pkl')
SETTINGS_FILE = os.path.join(DUMP_PATH,'settings.cfg')
LABEL_FILE = os.path.join(DUMP_PATH,'labels.csv')
CLASSIFIER_FILE = os.path.join(DUMP_PATH,'classifier.pkl')
EMBEDDING_FILE = os.path.join(DUMP_PATH,'embedding.pkl')
VIEW_SIZE_FILE = os.path.join(DUMP_PATH,'view_size.pkl')

THUMBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'static', FILE_NAME_PREFIX+'thumbs')
THUMBS_DIR_HTTP = os.path.join('static', FILE_NAME_PREFIX+'thumbs')
