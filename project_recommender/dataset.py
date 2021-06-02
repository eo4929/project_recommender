from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import sys
import os
import itertools
import random
import warnings

from six.moves import input
from six.moves import range

from reader import Reader
from trainset import Trainset


class Dataset:

    def __init__(self, reader=None, rating_scale=None):

        self.reader = reader
        self.rating_scale = rating_scale

        if self.rating_scale is None:
            if self.reader.rating_scale is None:
                raise ValueError('Oooops')
            #warnings.warn('Using rating_scale from reader, deprecated. Set '
            #              'rating_scale at dataset creation instead '
            #              '(load_from_file, load_from_folds, or load_from_df).')
            self.rating_scale = self.reader.rating_scale


    @classmethod
    def load_from_file(cls, file_path, reader, rating_scale=(1,5)): # NONE에서 (1,5) 로 수정함

        return DatasetAutoFolds(ratings_file=file_path, reader=reader,
                                rating_scale=rating_scale)


    def read_ratings(self, file_name):

        with open(os.path.expanduser(file_name)) as f:
            raw_ratings = [self.reader.parse_line(line) for line in
                           itertools.islice(f, self.reader.skip_lines, None)]
        return raw_ratings


    def construct_trainset(self, raw_trainset):

        raw2inner_id_users = {}
        raw2inner_id_items = {}

        current_u_index = 0
        current_i_index = 0

        ur = defaultdict(list)
        ir = defaultdict(list)

        # user raw id, item raw id, translated rating, time stamp
        for urid, irid, r, timestamp in raw_trainset:
            timestamp
            try:
                uid = raw2inner_id_users[urid]
            except KeyError:
                uid = current_u_index
                raw2inner_id_users[urid] = current_u_index
                current_u_index += 1
            try:
                iid = raw2inner_id_items[irid]
            except KeyError:
                iid = current_i_index
                raw2inner_id_items[irid] = current_i_index
                current_i_index += 1

            ur[uid].append((iid, r))
            ir[iid].append((uid, r))

        n_users = len(ur)  # number of users
        n_items = len(ir)  # number of items
        n_ratings = len(raw_trainset)

        trainset = Trainset(ur,
                            ir,
                            n_users,
                            n_items,
                            n_ratings,
                            self.rating_scale,
                            raw2inner_id_users,
                            raw2inner_id_items)

        return trainset


class DatasetAutoFolds(Dataset):

    def __init__(self, ratings_file=None, reader=None, df=None,
                 rating_scale=None):

        Dataset.__init__(self, reader, rating_scale)
        self.has_been_split = False  # flag indicating if split() was called.

        if ratings_file is not None:
            self.ratings_file = ratings_file
            self.raw_ratings = self.read_ratings(self.ratings_file)
        elif df is not None:
            self.df = df
            self.raw_ratings = [(uid, iid, float(r), None)
                                for (uid, iid, r) in
                                self.df.itertuples(index=False)]
        else:
            raise ValueError('Must specify ratings file or dataframe.')

    def build_full_trainset(self):

        return self.construct_trainset(self.raw_ratings)