
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

cimport numpy as np
import numpy as np
from six.moves import range


def baseline_sgd(self):

    cdef np.ndarray[np.double_t] bu = np.zeros(self.trainset.n_users)
    cdef np.ndarray[np.double_t] bi = np.zeros(self.trainset.n_items)

    cdef int u, i
    cdef double r, err
    cdef double global_mean = self.trainset.global_mean

    cdef int n_epochs = self.bsl_options.get('n_epochs', 20)
    cdef double reg = self.bsl_options.get('reg', 0.02)
    cdef double lr = self.bsl_options.get('learning_rate', 0.005)

    for dummy in range(n_epochs):
        for u, i, r in self.trainset.all_ratings():
            err = (r - (global_mean + bu[u] + bi[i]))
            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])

    return bu, bi