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
import pandas as pd

import pickle as pkl


from a2vq.src.settings import *

from a2vq.src.helper import save_csv,read_csv


class data_interface:
    def __init__(self):
        """this method gets called one when the interface is started"""
        print(">> init data interface")


    def setup(self):
        '''initialization procedure for setup the interface, can be executed by calling [URL]/setup'''
        pass

    def get_sample_ids(self):
        """this methods returns the ids for querying all data form the data base. Implementation of ids is
        is data base specific
        """
        pass

    def get_sample_features(self, ids):
        """get features of queried samples"""
        pass
    def get_sample_thumbs(self, ids):
        """get thumbnails of queried samples"""
        pass
    def get_sample_labels(self, ids):
        """get labels of queried samples"""
        pass

    def update_sample_labels(self, ids, y):
        """update labels of queried samples"""
        pass

    def get_unique_classes(self):
        """get a dict with unique class names"""
        pass

    def add_new_class(self, class_name):
        """add a new class with the respective class_name"""
        pass

    def remove_class(self, class_name):
        """remove unique class by its name"""
        pass

    def update_classes(self, cls):
        """updates classes by passed cls list"""
        pass

    # embedding methods
    def get_sample_embeddings(self, ids):
        """returns the embedding of specified indices"""
        pass

    def generate_embedding(self):
        """generates the embedding to self.embedding"""
        pass



    #
    # helper
    #

    def get_unlabeled_mask(self):
        """return all unlabeled uuids"""
        return pd.isnull(self.get_sample_labels(self.get_sample_ids()))

    def get_labeled_mask(self):
        """return all labeled uuids"""
        return np.invert(self.get_unlabeled_mask())
