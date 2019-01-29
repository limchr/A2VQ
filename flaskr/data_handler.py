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

from mnist import MNIST
import numpy as np
import os

from flaskr.settings import *


#
# Dataset functions. If you want to use an own data set, you have to implement a function here which looks like:
# def my_fancy_ds(load_images=False)
#
# if load_images==True it returns a tuple with (features,labels,images) [everything as a numpy array]
# otherwise it just returns a tuple with (features,labels)


def load_mnist(num_data=None,filter_labels=None,path=''):
    # Load the dataset
    (x_train, y_train) = MNIST(os.path.join(path,'mnist_original')).load_training()
    (x_test, y_test) = MNIST(os.path.join(path,'mnist_original')).load_testing()


    x_train = np.array(x_train,dtype=np.float32)
    x_test = np.array(x_test,dtype=np.float32)
    y_train = np.array(y_train,dtype=np.uint32)
    y_test = np.array(y_test,dtype=np.uint32)

    # Rescale -1 to 1
    x_train = (x_train - 127.5) / 127.5
    #x_train = np.expand_dims(x_train, axis=3)
    # Rescale -1 to 1
    x_test = (x_test - 127.5) / 127.5
    #x_test = np.expand_dims(x_test, axis=3)

    x_train = x_train.reshape((len(x_train), 28, 28, 1))
    x_test = x_test.reshape((len(x_test), 28, 28, 1))


    #filter specific labels out
    if filter_labels != None:
        inds = np.where([True if yy in filter_labels else False for yy in y_train])
        x_train = x_train[inds]
        y_train = y_train[inds]
        inds = np.where([True if yy in filter_labels else False for yy in y_test])
        x_test = x_test[inds]
        y_test = y_test[inds]

    #limit the number of samples
    if num_data != None:
        return x_train[:num_data],y_train[:num_data]
    else:
        return x_train,y_train

def ds_mnist(load_images = False):
    '''load mnist data set'''
    from flaskr.settings import DATA_PATH
    ret = load_mnist(num_data=500, path=DATA_PATH)
    features = ret[0].reshape((ret[0].shape[0], 28 * 28))
    labels = ret[1]
    labels = np.array([str(l) for l in labels])
    if load_images:
        imgs = np.squeeze(ret[0])
        return features, labels, imgs
    else:
        return features, labels
