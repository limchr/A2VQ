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
from shutil import rmtree
import numpy as np
import pandas as pd

from PIL import Image

import io
from base64 import encodebytes


def create_directory_if_not_defined(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def delete_files_in_directory(dir,recursive=False):
    for the_file in os.listdir(dir):
        file_path = os.path.join(dir, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path) and recursive: rmtree(file_path)
        except Exception as e:
            print(e)

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def setup_clean_directory(dir):
    create_directory_if_not_defined(dir)
    delete_files_in_directory(dir,recursive=True)

def get_files_of_type(path, type='jpg'):
    return np.array([x for x in sorted(os.listdir(path)) if x.lower().endswith(type.lower())])

def get_files_filtered(path, regex):
    import re
    matches = []
    pattern = re.compile(regex)
    for file in get_files_of_type(path,''):
        if pattern.match(file):
            matches.append(file)
    return np.array(matches)


def get_subdirectories(path):
    return os.walk(path).__next__()[1]

def array_is_in_array(arr1,arr2):
    rtn = np.zeros(arr1.shape, dtype='bool')
    for e in arr2:
        rtn = np.logical_or(rtn,arr1 == e)
    return rtn


def encode_img(img):
    pil_img = Image.fromarray(img) # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

def get_intersection(arr1, arr2):
    rtn = []
    for a1 in arr1:
        if a1 in arr2:
            rtn.append(a1)
    return rtn

def get_union(arr1, arr2):
    return list(set(arr1) | set(arr2))

def get_elements_not_in(arr1, arr2):
    '''returns elements from arr1 that are not in arr2'''
    rtn = []
    for a1 in arr1:
        if not a1 in arr2:
            rtn.append(a1)
    return rtn

def write_image(img,file):
    if np.max(img) <= 1: # is float array
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
    else:
        pil_img = Image.fromarray(img.astype(np.uint8))
    pil_img.save(file)

# db helper

def init_db(fields,index=None):
    'init db with the specified fields which is a tuple of tuples consisting of the type and name of the field'
    #assert len(columns) == len(types)
    df = pd.DataFrame(index=None)
    for t,c in fields:
        df[c] = pd.Series(dtype=t)
    if index is not None:
        df = df.set_index(index)
    return df


def export_db_as_csv(df, export_path):
    df.to_csv(export_path)


def add_db_rows(df, add_dict):
    types1 = df.dtypes
    was_empty = True if len(df) == 0 else False
    if df.index.name is None:
        df = df.append(add_dict, ignore_index=True)
    else:
        df = df.append(pd.Series(add_dict,name=add_dict[df.index.name])).drop_duplicates(subset=df.index.name, keep='last').sort_index()

    types2 = df.dtypes
    if not was_empty and not (types1 == types2).all():
        raise TypeError('Typechange detected in data base')
    else:
        return df

def read_csv(path):
    return np.array(pd.read_csv(path, header=None))
def save_csv(data, path):
    pd.DataFrame(data).to_csv(path, index=False, header=False)



if __name__ == '__main__':
    list1 = [4,6,1,0]
    list2 = [6,1,1,10]

    print(get_intersection(list1,list2))
    print(get_union(list1,list2))
    print(get_elements_not_in(list1,list2))

