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
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class dbscan:
    ''' implementation of DBSCAN and of DBQE'''
    def __init__(self,eps,min_pts,cannot_link_mat=None):
        self.eps = eps
        self.min_pts = min_pts
        self.cannot_link_mat = cannot_link_mat

    def fit_rec(self,x):
        self.x = x
        # calc dist matrix
        self.dists = cdist(x,x,'euclidean')

        self.visited = np.zeros(len(x),dtype='bool')
        self.clusters = np.zeros(len(x))
        self.cluster_no = 1
        for i in range(len(self.x)):
            if not self.visited[i]:
                self.visited[i] = True
                n = self.region_query(i)
                if len(n) < self.min_pts:
                    self.clusters[i] = -1  # mark as noise
                else:
                    self.expand_rec(i, n)
                    self.cluster_no += 1


    def expand_rec(self,i,n):
        self.clusters[i] = self.cluster_no
        for p in n:
            if not self.visited[p]:
                self.visited[p] = True
                n_new = self.region_query(p)
                if len(n_new) < self.min_pts:
                    self.clusters[i] = -1  # mark as noise
                else:
                    self.expand_rec(p, n_new)

    def fit(self,x):
        self.x = x
        # calc dist matrix
        self.dists = cdist(x,x,'euclidean')

        self.visited = np.zeros(len(x),dtype='bool')
        self.clusters = np.zeros(len(x))
        self.cluster_no = 1
        for i in range(len(self.x)):
            if not self.visited[i]:
                self.visited[i] = True
                n = self.region_query(i)
                if len(n) < self.min_pts:
                    self.clusters[i] = -1  # mark as noise
                else:
                    self.expand(i, n)
                    self.cluster_no += 1

    def expand(self,i,n):
        self.clusters[i] = self.cluster_no
        while len(n) > 0:
            p = n[0]
            n = np.delete(n,0)
            if not self.visited[p]:
                self.visited[p] = True
                n_new = self.region_query(p)
                if len(n_new) >= self.min_pts:
                    n = np.hstack((n,n_new))
            if self.clusters[p] == 0:
                self.clusters[p] = self.cluster_no


    def region_query(self,i):
        points = self.dists[i] < self.eps
        if not self.cannot_link_mat is None:
            points = np.logical_and(points,  np.logical_not(self.cannot_link_mat[i,:]))
        points[i] = False
        return np.where(points)[0]
    @staticmethod
    def region_query_static(i,dists,eps):
        points = dists[i] < eps
        points[i] = False
        return np.where(points)[0]


    @staticmethod
    def fit_single(x,i,eps,min_pts,dists=None ):
        # calc dist matrix if not passed
        dists = dists if dists is not None else cdist(x,x,'euclidean')

        visited = np.zeros(len(x),dtype='bool')
        clusters = np.zeros(len(x))

        visited[i] = True
        n = dbscan.region_query_static(i,dists,eps)
        if len(n) < min_pts:
            clusters[i] = -1  # mark as noise
        else:
            clusters[i] = 1
            while len(n) > 0:
                p = n[0]
                n = np.delete(n, 0)
                if not visited[p]:
                    visited[p] = True
                    n_new = dbscan.region_query_static(p,dists,eps)
                    if len(n_new) >= min_pts:
                        n = np.hstack((n, n_new))
                if clusters[p] == 0:
                    clusters[p] = 1
        return clusters==1


    @staticmethod
    def DBQE(x,xe,eps,min_pts,dists=None ):
        # calc dist matrix if not passed
        dists = dists if dists is not None else cdist(x,x,'euclidean')

        v = np.zeros(len(x),dtype='bool')
        c = np.zeros(len(x))
        t = np.array([xe])
        c[xe] = 1
        while len(t) > 0:
            a = t[0]
            t = np.delete(t, 0)
            if not v[a]:
                v[a] = True
                n = dbscan.region_query_static(a,dists,eps)
                if len(n) >= min_pts:
                    c[a] = 1
                    t = np.hstack((t, n))
        return c==1

if __name__ == '__main__':
    ''' small test program demonstrating use of DBQE '''
    from sklearn.datasets.samples_generator import make_blobs
    X, y = make_blobs(n_samples=90, centers=2, n_features=2, random_state = 0, cluster_std=.3)
    

    db = dbscan(1.5,3)
    clusters = dbscan.DBQE(X,0,1.5,3)



    plt.scatter(X[:,0],X[:,1],c=clusters,label=clusters)
    plt.ioff()
    plt.legend()
    plt.show()
