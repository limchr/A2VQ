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

def normalize_features(features, scale = (0,1)):
    normalized = np.zeros(features.shape)
    for i in range(features.shape[1]):
        min, max = np.min(features[:, i]), np.max(features[:, i])
        normalized[:, i] = (features[:, i] - min) / (max - min) * (scale[1]-scale[0]) + scale[0]
    return normalized