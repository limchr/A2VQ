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
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from math import exp
import matplotlib.pyplot as plt
import sys
class glvq():

    def __init__(self,max_prototypes_per_class=5,learning_rate=2,strech_factor=10,placement_strategy=None):
        """Constructor takes some additional arguments like max_prototypes_per_class, learning_rate. Additionally a
        placement strategy can passed as a callback. The function takes one sample x and corresponding label y and
        returns True if a new prototype should be placed otherwise false. Then the prototype is moved (see fit sample)."""
        self.max_prototypes_per_class = sys.maxsize if max_prototypes_per_class is None else max_prototypes_per_class
        self.learning_rate = learning_rate
        self.strech_factor = strech_factor
        self.placement_strategy = placement_strategy if placement_strategy is not None else self.placement_always



    def fit(self,x,y):
        """fit samples x and y incrementally. x and y are saved together with the yet trained samples in self.x and self.y"""
        x = np.array(x)
        y = np.array(y)
        feat_dim = x.shape[1]
        if not hasattr(self,'x'):
            self.x = x
            self.y = y
            self.prototypes = np.zeros(shape=(0,feat_dim))
            self.labels = np.array([])
        else:
            self.x = np.vstack((self.x,x))
            self.y = np.hstack((self.y,y))

        for xi,yi in zip(x,y):
            self.fit_sample(xi,yi)



    def placement_adaptive(self,xi,yi):
        """if the new sample got classified correctly before training it, no prototype is inserted."""
        return self.predict([xi]) != [yi]

    def placement_certainty_adaptive(self,xi,yi,thresh=0.8):
        """if the new sample got classified correctly before training it, no prototype is inserted."""
        return self.predict([xi]) != [yi] or self.predict_proba([xi]) < thresh

    def placement_always(self,xi,yi):
        """always insert new prototype"""
        return True

    def fit_sample(self,xi,yi):
        """fit a specific sample incrementally. Checks if a new prototype is neccessary (with self.placement_strategy).
        If so, a new prototype is inserted, otherwise the nearest prototype is moved corresponding to GLVQ update rule.
        """
        num_prototypes_per_class = len(np.where(self.labels == yi)[0])
        if (num_prototypes_per_class == 0 or self.placement_strategy(xi,yi)) \
                and not num_prototypes_per_class >= self.max_prototypes_per_class:  # add new
            # print('PLACE NEW prototype for class '+str(yi)+' (number of prototypes for this class: '+str(num_prototypes_per_class)+')')
            self.prototypes = np.vstack((self.prototypes, xi))
            self.labels = np.hstack((self.labels, yi))
            #print("adding prototype for class" + str(yi))
        elif len(set(self.labels)) > 1:  # move prototype
            # print('MOVE EXISTING prototype for class '+str(yi)+' (number of prototypes for this class: '+str(num_prototypes_per_class)+')')
            proto_dist = self.dist(np.array([xi]), self.prototypes)
            proto_dist = proto_dist[0]

            # find out nearest proto of same class and different class
            smallest_dist_wrong = float("inf")
            smallest_dist_right = float("inf")
            w1i = -1
            w2i = -1
            for i, p in enumerate(proto_dist):
                if self.labels[i] == yi and smallest_dist_right > p:
                    smallest_dist_right = p
                    w1i = i
                if self.labels[i] != yi and smallest_dist_wrong > p:
                    smallest_dist_wrong = p
                    w2i = i
            w1 = self.prototypes[w1i].copy()
            w2 = self.prototypes[w2i].copy()
            d1 = proto_dist[w1i]
            d2 = proto_dist[w2i]

            mu = (d1 - d2) / (d1 + d2)
            # sigm = (1/(1+exp(-mu)))
            derive = exp(mu * self.strech_factor) / (
            (exp(mu * self.strech_factor) + 1) * (exp(mu * self.strech_factor) + 1))
            #print('mu: ' + str(mu) + ' derive: ' + str(derive))
            # GLVQ
            self.prototypes[w1i] = w1 + self.learning_rate * derive * (d2 / ((d1 + d2) * (d1 + d2))) * (xi - w1)
            self.prototypes[w2i] = w2 - self.learning_rate * derive * (d1 / ((d1 + d2) * (d1 + d2))) * ( xi - w2)
            #print('derive ' + str(derive))
            #print('move p1 from ' + str(w1) + ' to ' + str(self.prototypes[w1i]))
            #print('move p2 from ' + str(w2) + ' to ' + str(self.prototypes[w2i]))

        else:
            print('cant move because only one labeled class')
            print(set(self.labels))

    def dist(self,x,y):
        """calculates the distance matrix used for determine the winner and looser prototype"""
        return cdist(x,y,'euclidean')

    def predict_proba(self,x,full_matrix=False,return_winning_prototype_i=False):
        """returns the relative distance of prototypes to samples from x"""


        if not hasattr(self,'labels') or len(set(self.labels)) < 2:
            rtn = np.array([0] * len(x)) if not full_matrix else np.zeros((len(x),1))
            if return_winning_prototype_i:
                return rtn, np.array([None] * len(x))
            else:
                return rtn

        num_classes = len(set(self.labels))

        ds = self.dist(x,self.prototypes)
        relsims = []
        winning_prototype_is = []
        for d in ds:
            if full_matrix:
                protos = self.get_win_loose_prototypes(d,num_classes)
                proto_relsims = np.zeros((num_classes))
                # step through the loosers and calculate relsim for all to get full certainty matrix
                for i,p in enumerate(protos[1:]):
                    proto_relsims[protos[i]] = (d[p] - d[protos[0]]) / (d[p] + d[protos[0]])
                relsims.append(proto_relsims)
                winning_prototype_is.append(protos[0])
            else:
                winner,looser = self.get_win_loose_prototypes(d)
                relsims.append((d[looser]-d[winner])/(d[looser]+d[winner]))
                winning_prototype_is.append(winner)

        if return_winning_prototype_i:
            return (np.array(relsims),np.array(winning_prototype_is))
        else:
            return np.array(relsims)

    def get_win_loose_prototypes(self,dists,n=2):
        """get the winning prototype and the n-1 loosing prototypes"""
        ds = np.argsort(dists)
        # the classes already included into prototype list
        labels_included = []
        prototypes_i = []

        for id, d in enumerate(ds):
            if not self.labels[d] in labels_included:
                labels_included.append(self.labels[d])
                prototypes_i.append(d)
                if len(prototypes_i) >= n:
                    break
        return prototypes_i

    def predict(self,x):
        """predicts samples from x"""
        if not hasattr(self,'labels') or len(set(self.labels)) < 2:
            return [-1] * len(x)
        return np.array(self.labels[np.argmin(self.dist(x,self.prototypes), axis=1)])

    def predict_sample(self,x):
        """predicts a single sample"""
        return self.predict(x[np.newaxis])

    def score(self,x,y):
        """0/1 loss"""
        y_pred = self.predict(x)
        return float(len(np.where(y==y_pred)[0]))/len(x)


    def visualize_2d(self,ax=None):
        """draws a nice little visualization about the actual train state with matplotlib"""
        if not hasattr(self,'pltCount'):
            self.pltCount = 0
        if ax is None:
            ax = plt.gca()
        plt.cla()
        plt.ion()
        some_colors = ['red','green','blue','yellow','orange','pink','black','brown']
        pred = self.predict(self.x)
        for x,y in zip(self.x,pred):
            plt.scatter(x[0],x[1],c='grey')#some_colors[int(y)]
        for p,l in zip(self.prototypes,self.labels):
            plt.scatter(p[0],p[1],c=some_colors[int(l)],marker='D',s=80,edgecolors='black')
        plt.pause(0.001)
        plt.savefig('./plt/plt'+str(self.pltCount)+'.png',format='png')
        self.pltCount +=1

        plt.ioff()


if __name__ == '__main__':
    """a small test program wich demonstrate the use of the classifier"""
    # x,y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_repeated=0,
    #                                      n_classes=2, n_clusters_per_class=1, weights=None, flip_y=0.01, class_sep=0.5,
    #                                      hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=43)
    x = np.random.multivariate_normal((0,0), [[1,0],[0,1]], 500)
    x = np.vstack((x,np.random.multivariate_normal((0.5,0), [[1,0],[0,1]], 500)))
    y = np.array(['class1' for _ in range(500)])
    y = np.hstack((y,['class2' for _ in range(500)]))


    x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.33, random_state = 45)
    b = glvq()

    ax = plt.figure().gca()

    for x,y in zip(x_train,y_train):
        b.fit([x],[y])

    #b.visualize_2d(ax)
    #plt.show()

    print(b.predict(x_test))
    print(b.predict_proba(x_test))
    print(y_test)
    print('score: '+str(b.score(x_test,y_test)))