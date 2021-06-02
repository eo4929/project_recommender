

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from six import iteritems
import heapq

import pyximport; pyximport.install()

from algo_base import AlgoBase



class SymmetricAlgo(AlgoBase):

    def __init__(self, sim_options={}, verbose=True, **kwargs):

        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)
        self.verbose = verbose

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users
        self.xr = self.trainset.ur if ub else self.trainset.ir
        self.yr = self.trainset.ir if ub else self.trainset.ur

        return self

    def switch(self, u_stuff, i_stuff):

        if self.sim_options['user_based']:
            return u_stuff, i_stuff
        else:
            return i_stuff, u_stuff


class KNNBasic(SymmetricAlgo):

    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options, verbose=verbose,
                               **kwargs)
        self.k = k
        self.min_k = min_k

    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)
        self.sim = self.compute_similarities(verbose=self.verbose)

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            #print('User and/or item is unkown.')
            exit()

        x, y = self.switch(u, i)

        neighbors = [(self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (sim, r) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * r
                actual_k += 1

        if actual_k < self.min_k:
            #print('Not enough neighbors.')
            exit()

        est = sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details


class KNNWithMeans(SymmetricAlgo):

    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options,
                               verbose=verbose, **kwargs)

        self.k = k
        self.min_k = min_k

    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)
        self.sim = self.compute_similarities(verbose=self.verbose)

        self.means = np.zeros(self.n_x)
        for x, ratings in iteritems(self.xr):
            self.means[x] = np.mean([r for (_, r) in ratings])

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            #print('User and/or item is unkown.')
            exit()

        x, y = self.switch(u, i)

        neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[1])

        est = self.means[x]

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (nb, sim, r) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * (r - self.means[nb])
                actual_k += 1

        if actual_k < self.min_k:
            sum_ratings = 0

        try:
            est += sum_ratings / sum_sim
        except ZeroDivisionError:
            pass  # return mean

        details = {'actual_k': actual_k}
        return est, details


class KNNBaseline(SymmetricAlgo):

    def __init__(self, k=40, min_k=1, sim_options={}, bsl_options={},
                 verbose=True, **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options,
                               bsl_options=bsl_options, verbose=verbose,
                               **kwargs)

        self.k = k
        self.min_k = min_k

    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)
        self.bu, self.bi = self.compute_baselines(verbose=self.verbose)
        self.bx, self.by = self.switch(self.bu, self.bi)
        self.sim = self.compute_similarities(verbose=self.verbose)

        return self

    def estimate(self, u, i):

        est = self.trainset.global_mean
        if self.trainset.knows_user(u):
            est += self.bu[u]
        if self.trainset.knows_item(i):
            est += self.bi[i]

        x, y = self.switch(u, i)

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            return est

        neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[1])

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (nb, sim, r) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                nb_bsl = self.trainset.global_mean + self.bx[nb] + self.by[y]
                sum_ratings += sim * (r - nb_bsl)
                actual_k += 1

        if actual_k < self.min_k:
            sum_ratings = 0

        try:
            est += sum_ratings / sum_sim
        except ZeroDivisionError:
            pass  # just baseline again

        details = {'actual_k': actual_k}
        return est, details


class KNNWithZScore(SymmetricAlgo):

    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options, verbose=verbose,
                               **kwargs)

        self.k = k
        self.min_k = min_k

    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)

        self.means = np.zeros(self.n_x)
        self.sigmas = np.zeros(self.n_x)
        # when certain sigma is 0, use overall sigma
        self.overall_sigma = np.std([r for (_, _, r)
                                     in self.trainset.all_ratings()])

        for x, ratings in iteritems(self.xr):
            self.means[x] = np.mean([r for (_, r) in ratings])
            sigma = np.std([r for (_, r) in ratings])
            self.sigmas[x] = self.overall_sigma if sigma == 0.0 else sigma

        self.sim = self.compute_similarities(verbose=self.verbose)

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            #print('User and/or item is unkown.')
            exit()

        x, y = self.switch(u, i)

        neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[1])

        est = self.means[x]

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (nb, sim, r) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * (r - self.means[nb]) / self.sigmas[nb]
                actual_k += 1

        if actual_k < self.min_k:
            sum_ratings = 0

        try:
            est += sum_ratings / sum_sim * self.sigmas[x]
        except ZeroDivisionError:
            pass  # return mean

        details = {'actual_k': actual_k}
        return est, details