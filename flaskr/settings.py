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
from flaskr.helper import create_directory_if_not_defined
from flaskr.intelligent_labeling import embedding_tsne, embedding_umap


''' default view size in normalized visualization space '''
DEFAULT_VIEW_SIZE = 0.25

''' experiment breaks when there are lesser than MIN_UNLABELED unlabeled samples '''
MIN_UNLABELED = 10

''' function for evaluating the embedding '''
EMBEDDING_FUN = embedding_tsne

''' define a name for the data set here, this is used to load e.g. thumbnails from the static directory'''
FILE_NAME_PREFIX = 'robot'

''' path with images (e.g. mnist)'''
IMAGE_PATH = '/homes/climberg/src/min_workspace/data/objects/'

''' path with images (e.g. mnist)'''
IMAGE_PATH = '/homes/climberg/src/min_workspace/data/objects/'
# IMAGE_PATH = '/hri/storage/user/climberg/datasets/outdoor/5classes'

''' the output files are saved in DUMP_PATH'''
DUMP_PATH = '/homes/climberg/src/min_workspace/data/'
# DUMP_PATH = '/hri/storage/user/climberg/dumps/a2vq'

# define several paths
FEATURES_FILE = os.path.join(DUMP_PATH,FILE_NAME_PREFIX+'features.pkl')
IMAGES_FILE = os.path.join(DUMP_PATH,FILE_NAME_PREFIX+'images.pkl')
SETTINGS_FILE = os.path.join(DUMP_PATH,FILE_NAME_PREFIX+'settings.cfg')
LABEL_FILE = os.path.join(DUMP_PATH,FILE_NAME_PREFIX+'labels.csv')
CLASSIFIER_FILE = os.path.join(DUMP_PATH,FILE_NAME_PREFIX+'classifier.pkl')
EMBEDDING_FILE = os.path.join(DUMP_PATH,FILE_NAME_PREFIX+'embedding.pkl')
THUMBS_DIR = os.path.join('flaskr','static', FILE_NAME_PREFIX+'thumbs')
THUMBS_DIR_HTTP = os.path.join('static', FILE_NAME_PREFIX+'thumbs')
VIEW_SIZE_FILE = os.path.join(DUMP_PATH,FILE_NAME_PREFIX+'view_size.pkl')

create_directory_if_not_defined(DUMP_PATH)
