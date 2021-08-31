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
import numpy as np
import pandas as pd

import pickle as pkl
import dill

from a2vq.src.helper import setup_clean_directory,save_csv,read_csv,encode_img
from a2vq.src.feature_extraction import feature_extraction_of_arbitrary_image_ds,resize_image_tuple_to_max_edge_length
from a2vq.src.glvq import glvq

from a2vq.src.data_interface import data_interface

from a2vq.src.settings import EMBEDDING_FUN

class harddrive_interface(data_interface):
    def __init__(self, image_path, out_path):
        super().__init__()

        ''' define a name for the data set here, this is used to load e.g. thumbnails from the static directory'''
        self.FILE_NAME_PREFIX = 'common'
        # change this to an empty directory on your hard drive
        self.ROOT_DB_BASE_PATH = out_path
        # change this to a directory containing images for vizualizing
        self.IMAGE_PATH = image_path

        self.DB_BASE_PATH = os.path.join(self.ROOT_DB_BASE_PATH,self.FILE_NAME_PREFIX)
        self.EXPORT_PATH = os.path.join(self.DB_BASE_PATH,'export')


        ''' the output files used for a2vq operation are saved in DUMP_PATH'''
        self.DUMP_PATH = os.path.join(self.DB_BASE_PATH,'dump')
        ''' when exporting a dataset, first step is to save all labeled samples class-wise to do a manual cleanup step'''
        self.CLASS_WISE_DB_PATH = os.path.join(self.DB_BASE_PATH,'class_wise')
        ''' here, the db file is saved'''
        self.DB_FILE = os.path.join(self.DB_BASE_PATH,'objs.db')


        # define several paths (may move this to db_file_interface)
        self.FEATURES_FILE = os.path.join(self.DUMP_PATH,'features.pkl')
        self.IMAGES_FILE = os.path.join(self.DUMP_PATH,'images.pkl')
        self.LABEL_FILE = os.path.join(self.DUMP_PATH,'labels.csv')
        self.CLASSIFIER_FILE = os.path.join(self.DUMP_PATH,'classifier.pkl')
        self.UNIQUE_CLASSES_FILE = os.path.join(self.DUMP_PATH,'unique_classes.csv')

    def setup(self):
        """loads images, calculates features, and creates thumbnails and dumps everything to harddisk"""
        setup_clean_directory(self.DB_BASE_PATH)
        setup_clean_directory(self.DUMP_PATH)
        if(hasattr(self,'embedding')):
            del self.embedding


        print('>> extract features')
        feats, image_list, img_tuple = feature_extraction_of_arbitrary_image_ds(self.IMAGE_PATH)

        print('>> save label file, images, features and classifier')
        num_samples = len(image_list)
        labels = np.zeros(num_samples, dtype=str)
        label_file = np.vstack((image_list, labels)).T
        save_csv(label_file, self.LABEL_FILE)
        thumbs = resize_image_tuple_to_max_edge_length(img_tuple,100)

        thumbs_encoded = np.array([encode_img(t) for t in thumbs])

        with open(self.FEATURES_FILE, 'wb') as f:
            pkl.dump(feats, f)

        with open(self.IMAGES_FILE, 'wb') as f:
            pkl.dump(thumbs_encoded, f)

        with open(self.UNIQUE_CLASSES_FILE, 'w') as f:
            f.write('')

        cls = glvq()
        with open(self.CLASSIFIER_FILE, 'wb') as f:
            dill.dump(cls, f)


    def get_sample_ids(self):
        """this methods returns the line index of the csv as ids since it is assumed that this will not change (may be
        different in other interfaces)
        """
        return list(range(len(read_csv(self.LABEL_FILE)[:, 0])))[0:5000:40]
        #return list(range(len(read_csv(self.LABEL_FILE)[:, 0])))

    def get_sample_features(self, ids):
        """get features of queried samples"""
        with open(self.FEATURES_FILE, 'rb') as f:
            feats = pkl.load(f)
        if ids is None:
            return feats
        else:
            return feats[ids]

    def get_sample_thumbs(self, ids):
        """get thumbnails of queried samples"""
        with open(self.IMAGES_FILE, 'rb') as f:
            images_file = pkl.load(f)
        return images_file[ids]

    def get_sample_labels(self, ids):
        """get labels of queried samples"""
        csv = read_csv(self.LABEL_FILE)[:, 1]
        csv[pd.isnull(csv)] = None
        if not (ids is None):
            csv = csv[ids]
        return csv

        # # gt for cups n bottles
        # df = pd.read_csv('/home/chris/datasets/cupsnbottles/properties.csv')
        # df.index = df['index']
        # a = [x[:-4] for x in read_csv(self.LABEL_FILE)[:, 0]]
        # gt = np.array(df.loc[a].label)
        # none_ids = np.random.choice(len(gt), len(gt)//2, replace=False)
        # # gt[none_ids] = None
        # return gt[ids]

        # csv[pd.isnull(csv)] = None # replace nans with None
        # return csv

    def update_sample_labels(self, ids, y):
        """update the labels for specified ids"""
        int_ids = [ int(i) for i in ids]
        labels = read_csv(self.LABEL_FILE)
        labels[int_ids, 1] = y
        save_csv(labels, self.LABEL_FILE)

        self._fit_classifier(int_ids)


    def get_unique_classes(self):
        """get a dict with unique class names"""
        with open(self.UNIQUE_CLASSES_FILE, 'r') as f:
            cls = f.read().split(';')
        if '' in cls:
            cls.remove('')
        return cls

    def add_new_class(self, class_name):
        """add a new class with the respective class_name"""
        cls = self.get_unique_classes()
        if not class_name in cls:
            cls.append(class_name)
            self.update_classes(cls)


    def remove_class(self, class_name):
        """remove unique class by its name"""
        cls = self.get_unique_classes()
        cls.remove(class_name)
        self.update_classes(cls)

    def update_classes(self, cls):
        """updates classes by passed cls list"""
        with open(self.UNIQUE_CLASSES_FILE, 'w') as f:
            f.write(';'.join(cls))

    # embedding methods
    def get_sample_embeddings(self, ids):
        """returns the embedding of specified indices"""
        if not hasattr(self,'embedding'):
            self.generate_embedding() # calculate embedding with all ids
        return self.embedding[ids]

    def generate_embedding(self):
        """generates the embedding to self.embedding"""
        x = self.get_sample_features(None)
        y = self.get_sample_labels(None)
        self.embedding = EMBEDDING_FUN(x, y)




    # optional methods for using a2vq querying approach
    def get_sample_probas(self, ids):
        """returns classifier probability estimates of specified samples by ids"""
        self._load_classifier()
        return self._classifier.predict_proba(self.get_sample_features(ids))

    def get_sample_preds(self, ids):
        """returns classifier estimates of specified samples by ids"""
        self._load_classifier()
        return self._classifier.predict(self.get_sample_features(ids))


    def _load_classifier(self):
        with open(self.CLASSIFIER_FILE, 'rb') as f:
            self._classifier = dill.load(f)

    def _dump_classifier(self):
        with open(self.CLASSIFIER_FILE, 'wb') as f:
            dill.dump(self._classifier, f)

    def _fit_classifier(self,ids):
        """updates the classifier with new labeled data for updated probability estimates"""
        self._load_classifier()
        self._classifier.fit(self.get_sample_features(ids), self.get_sample_labels(ids))
        self._dump_classifier()






if __name__ == '__main__':
    print('main')
    img_path = '/hri/localdisk/climberg/data/cupsnbottles/small'
    out_path = '/hri/localdisk/climberg/data/a2vq_out_path'

    inter = harddrive_interface(image_path=img_path,out_path=out_path)
    # inter.setup()

    ids = inter.get_sample_ids()
    inter.get_sample_features(ids)
    inter.get_sample_labels(ids)

    inter.update_classes([])
    inter.get_unique_classes()
    inter.add_new_class('asdf')
    inter.add_new_class('ascv')
    inter.get_unique_classes()
    inter.remove_class('asdf')