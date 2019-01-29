# Active Learning for Image Recognition using a Visualization-Based User Interface

## Description

In this project are all the files required to run the active learning approach A2VQ together with the web-based labeling interface. Additionally the experimental setup together with the data recording functionality is integrated within the source for fast replication of all trials from our user study.

## Setup

To run the project you need python3. Additionally you need the package flask, configparser and sklearn:

```
pip install flask configparser sklearn
```

To test the project we implemented data set functions for using the mnist data set. First change DATA_PATH variable in flaskr/settings.py to the directory where your data sets are stored. To use the MNIST data set you have to install [python-mnist](https://pypi.org/project/python-mnist/) package with pip and download the mnist files from [here](http://yann.lecun.com/exdb/mnist/). The downloaded archives have to be extracted in a subdirectory of DATA_PATH called 'mnist_original'.

```
pip install python-mnist 
```


## Run
To start the interface web page, run main_page with python3:

```
python3 main_page.py
```

This is starting flask within a debugging server on localhost on port 5000.

To prepare experiments for the first run, exec the following pages:

```
http://localhost:5000/init_db
http://localhost:5000/init_labels
http://localhost:5000/build_embedding
```

for setting some initial files and building the t-SNE embedding for visualization.

Now you can go to the main page http://localhost:5000/ and register a new participant with an ID. On the next screen you can run all trials used in the experiment. Please note that we can not provide the data set used in the paper because of copyright issues.

## Usage


The following gif is showing the user interface while labeling MNIST. However, preliminary experiments showed that the t-SNE visualization is much better when using image features from a state of the art CNN. In the MNIST examples are some outlier because t-SNE is trained directly on pixel values. 

![A2VQ workflow](./img/a2vq.gif)

The captured user interactions are saved under the subdirectory 'participants' in several csv files.

## Used Software

The web-based Interface makes use of [JQuery](https://jquery.org/) and [Bootstrap](https://getbootstrap.com/) which is both licenced under MIT license. 

## Additional Literature
[Christian Limberg, Heiko Wersing, Helge Ritter (2018)<br>
Efficient Accuracy Estimation for Instance-Based Incremental Active Learning<br>
European Symposium on Artificial Neural Networks (ESANN)](http://www.honda-ri.de/pubs/pdf/2125.pdf)

## License
Copyright (C) 2019<br>
Christian Limberg<br>
Centre of Excellence Cognitive Interaction Technology (CITEC)<br>
Bielefeld University<br><br>

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
