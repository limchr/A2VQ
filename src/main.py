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

from flask import Flask
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)


from a2vq.src.settings import HD_IMAGE_PATH, HD_OUT_PATH



"""import the file interface for loading everything from harddrive"""
from a2vq.src.harddrive_interface import harddrive_interface

"""define the interface to be used"""
interface = harddrive_interface(image_path=HD_IMAGE_PATH, out_path=HD_OUT_PATH)




"""flask app"""
app = Flask(__name__)


@app.route('/setup')
def setup():
    """initialize the interface (if needed, depends on the actual implementation), (call at first run)"""
    interface.setup()
    return 'Done'

@app.route('/add_class_<class_label>')
def new_class(class_label):
    print('Introducting new class '+class_label)
    interface.add_new_class(class_label)
    return embedding_view()



@app.route('/interface_tunnel_<method_name>')
def interface_tunnel(method_name):
    """can be used for debugging the interface, by calling an interface method from webpage e.g. http://localhost:5000/interface_method_get_sample_ids"""
    try:
        rtn = str(getattr(interface,method_name)())
    except:
        rtn = 'invalid request'
    return rtn

@app.route('/add_labels', methods = ['POST'])
def add_labels():
    """add labels is executed every time a user has labeled samples with A2VQ. This function is called via AJAX asyncronously."""
    label = request.form.get('label')
    ids = request.form.getlist('ids[]') # note that label is converted to str
    interface.update_sample_labels(ids,label)
    return '{"success": "true"}'


#
# functions for different sample indexing
#

@app.route('/')
@app.route('/all')
def embedding_view():
    """main embedding view for displaying whole embedding"""
    try:
        ids = interface.get_sample_ids()
        x_embedding = interface.get_sample_embeddings(ids)
        thumbs = interface.get_sample_thumbs(ids)
        labels = interface.get_sample_labels(ids)
        batch = {}
        for i,e,t,l in zip(ids,x_embedding,thumbs,labels):
            batch[i] = {'id':i,'embedding':list(e),'thumb':t,'label':l}
        classes = interface.get_unique_classes()
    except:
        return 'can not create embedding. please call first: /setup'
    return render_template('embedding_view.html', samples=batch, classes=classes)


if __name__ == '__main__':
    # working_dir = os.path.realpath(__file__)[:-19]
    # sys.path.append(working_dir)
    # os.chdir(working_dir)
    # print('change dir to ' + working_dir)
    app.run(host= '0.0.0.0', debug=True)
