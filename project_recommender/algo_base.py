
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import warnings
import numpy as np
from six import get_unbound_function as guf

import pyximport; pyximport.install()

import similarity as sim
import optimizing_baseline

from collections import namedtuple

from abc import ABCMeta, abstractmethod

class AlgoBase(object):
    #__metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        #print('algo init')
        self.bsl_options = kwargs.get('bsl_options', {})
        self.sim_options = kwargs.get('sim_options', {})
        if 'user_based' not in self.sim_options:
            self.sim_options['user_based'] = True
        self.skip_train = False

        if (guf(self.__class__.fit) is guf(AlgoBase.fit) and
           guf(self.__class__.train) is not guf(AlgoBase.train)):
            warnings.warn('It looks like this algorithm (' +
                          str(self.__class__) +
                          ') implements train() '
                          'instead of fit(): train() is deprecated, '
                          'please use fit() instead.', UserWarning)

    def train(self, trainset):

        warnings.warn('train() is deprecated. Use fit() instead', UserWarning)

        self.skip_train = True
        self.fit(trainset)

        return self

    def fit(self, trainset):
        #print('fit func in algo_base')
        if (guf(self.__class__.train) is not guf(AlgoBase.train) and
                not self.skip_train):
            self.train(trainset)
            return
        self.skip_train = False

        self.trainset = trainset

        # (re) Initialise baselines
        self.bu = self.bi = None

        return self


    def compute_baselines(self, verbose=False):

        if self.bu is not None:
            return self.bu, self.bi

        method = dict(sgd=optimizing_baseline.baseline_sgd)

        method_name = self.bsl_options.get('method', 'sgd')

        try:
            #if verbose:
            #    print('Estimating biases using', method_name + '...')
            self.bu, self.bi = method[method_name](self)
            return self.bu, self.bi
        except KeyError:
            raise ValueError('Invalid method ' + method_name +
                             ' for baseline computation.' +
                             ' Available method is sgd.')

    def compute_similarities(self, verbose=False):

        construction_func = {'cosine': sim.cosine,
                             'msd': sim.msd,
                             'pearson': sim.pearson,
                             'pearson_baseline': sim.pearson_baseline}

        if self.sim_options['user_based']:
            n_x, yr = self.trainset.n_users, self.trainset.ir
        else:
            n_x, yr = self.trainset.n_items, self.trainset.ur

        min_support = self.sim_options.get('min_support', 1)

        args = [n_x, yr, min_support]

        name = self.sim_options.get('name', 'msd').lower()
        name = 'pearson_baseline' # 이걸로 고정시키자
        if name == 'pearson_baseline':
            shrinkage = self.sim_options.get('shrinkage', 100)
            bu, bi = self.compute_baselines()
            if self.sim_options['user_based']:
                bx, by = bu, bi
            else:
                bx, by = bi, bu

            args += [self.trainset.global_mean, bx, by, shrinkage]

        try:
            #if verbose:
            #    print('Computing the {0} similarity matrix...'.format(name))
            sim_res = construction_func[name](*args)
            #if verbose:
            #    print('Done computing similarity matrix.')
            return sim_res
        except KeyError:
            raise NameError('Wrong sim name ' + name + '. Allowed values ' +
                            'are ' + ', '.join(construction_func.keys()) + '.')

    #@abstractmethod
    #def estimate(self,u,i):
    #    print('pass')
    #    pass

    def predict(self, uid, iid, r_ui=None, clip=True, verbose=False):

        # Convert raw ids to inner ids
        try:
            iuid = self.trainset.to_inner_uid(uid)
        except ValueError:
            iuid = 'UKN__' + str(uid)
        try:
            iiid = self.trainset.to_inner_iid(iid)
        except ValueError:
            iiid = 'UKN__' + str(iid)

        details = {}
        try:
            #print('before estimate')
            est = self.estimate(iuid, iiid)
            #print('after estimate')
            
            if isinstance(est, tuple):
                est, details = est
            
            details['was_impossible'] = False
        except:
            est = self.trainset.global_mean 
            details['was_impossible'] = True
        

        #details['was_impossible'] = False

        # clip estimate into [lower_bound, higher_bound]
        if clip:
            lower_bound, higher_bound = self.trainset.rating_scale

            est = min(higher_bound, est)
            est = max(lower_bound, est)

        pred = Prediction(uid, iid, r_ui, est, details)

        if verbose:
            print(pred)

        return pred

    
    def test(self, testset, verbose=False):

        # The ratings are translated back to their original scale.
        predictions = [self.predict(uid,
                                    iid,
                                    r_ui_trans,
                                    verbose=verbose)
                       for (uid, iid, r_ui_trans,timestamp) in testset]
        return predictions
    
    def rmse(self, predictions, verbose=True):

        if not predictions:
            raise ValueError('Prediction list is empty.')

        mse = np.mean([float((true_r - est)**2)
                   for (_, _, true_r, est, _) in predictions])
        rmse_ = np.sqrt(mse)

        if verbose:
            print('RMSE: {0:1.4f}'.format(rmse_))

        return rmse_


class Prediction(namedtuple('Prediction',
                            ['uid', 'iid', 'r_ui', 'est', 'details'])):

    __slots__ = ()
    '''
    def __str__(self):

        s = 'user: {uid:<10} '.format(uid=self.uid)
        s += 'item: {iid:<10} '.format(iid=self.iid)
        if self.r_ui is not None:
            s += 'r_ui = {r_ui:1.2f}   '.format(r_ui=self.r_ui)
        else:
            s += 'r_ui = None   '
        s += 'est = {est:1.2f}   '.format(est=self.est)
        s += str(self.details)
        # s의 내용 아래처럼 바꾸기
        s = str(self.uid) + '\t' + str(self.iid) + '\t' + str(self.est) + '\n'

        return s
    '''