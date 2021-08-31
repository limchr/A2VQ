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


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

from tensorflow.keras import backend as K

from sklearn.decomposition import PCA

import os
import numpy as np
import pickle as pkl
from PIL import Image


from a2vq.src.helper import get_files_of_type



def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x



def create_arbitrary_image_ds(image_dir, output_dir, image_list=None):
    deep_feats, image_list, img_tuple_resized = feature_extraction_of_arbitrary_image_ds(image_dir,image_list)

    with open(os.path.join(output_dir,'features.pkl'),'wb') as f:
        pkl.dump(deep_feats,f)
    with open(os.path.join(output_dir,'filenames.pkl'),'wb') as f:
        pkl.dump(image_list,f)
    with open(os.path.join(output_dir,'images.pkl'),'wb') as f:
        pkl.dump(img_tuple_resized,f)
    print('done')


def feature_extraction_of_arbitrary_image_ds(image_dir, image_list=None):
    if image_list is None:
        image_list = get_files_of_type(image_dir,'jpg')
    image_paths = []
    for i in image_list:
        image_paths.append(os.path.join(image_dir, i))

    img_tuple = read_images(image_paths)
    img_tuple_rgb = convert_image_tuple_to_rgb(img_tuple)
    #img_tuple_resized = resize_image_tuple(img_tuple_rgb,size=(299,299,3))
    img_tuple_resized = resize_image_tuple_to_max_edge_length(img_tuple_rgb, 80)

    deep_feats = get_deep_feats(img_tuple_rgb,'vgg16')

    return deep_feats, image_list, img_tuple_resized



def get_deep_feats(imgs, use_model='vgg16', pca_dim=None):
    model, imgsize = get_model(use_model)
    if pca_dim != None:
        pca = PCA(pca_dim)

    rtn_feats = []
    for img in imgs:
        img_data = preprocess_image(img,imgsize)
        features = model.predict(img_data).flatten()
        rtn_feats.append(features)
    rtn_feats = np.array(rtn_feats)
    if pca_dim != None:
        rtn_feats = pca.fit_transform(rtn_feats)

    return rtn_feats

def preprocess_image(img, imgsize):
    img_load = np.array(Image.fromarray(img).resize(imgsize[:2]))
    img_data = image.img_to_array(img_load)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data, dim_ordering='tf')
    return img_data

def get_model(use_model):
    if use_model == 'inception':
        imgsize = (299,299)
        model = InceptionV3(weights='imagenet',include_top=False)
    elif use_model == 'vgg16':
        imgsize = (224,224)
        model = VGG16(weights='imagenet',include_top=True)
        model = Model(inputs=model.input, outputs=model.get_layer('fc2').output)
    else:
        raise NotImplemented('model '+use_model+' is not there')
    return model, imgsize


def resize_image_tuple_to_max_edge_length(imgs, max_edge_length=200):
    rtn_imgs = []
    for img in imgs:
        s = img.shape
        if s[0] > s[1]: #height is greater
            h = max_edge_length
            w = int((float(s[1])/s[0]) * max_edge_length)
        else:
            w = max_edge_length
            h = int((float(s[0])/s[1]) * max_edge_length)
        rtn_imgs.append(np.array(Image.fromarray(img).resize((w,h))))
    return rtn_imgs



def is_bw_image(img):
    return img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)

def convert_bw_to_rgb(img):
    if img.ndim == 3:
        img = np.squeeze(img)
    result = np.stack((img, img, img), axis=2)
    return result



def convert_image_tuple_to_rgb(imgs):
    for i in range(len(imgs)):
        if is_bw_image(imgs[i]):
            imgs[i] = convert_bw_to_rgb(imgs[i])
    return imgs




def read_images(files,dir=None):
    imgs = []
    for f in files:
        img = np.array(Image.open(os.path.join(dir, f) if dir is not None else f))
        imgs.append(img)
    return imgs

if __name__ == '__main__':
    imgs = read_images(['1581526322609183518_3.jpg','1581526311342329584_0.jpg'],dir='/hri/localdisk/climberg/data/cupsnbottles/images')
    resized = resize_image_tuple_to_max_edge_length(imgs)

    print('done')