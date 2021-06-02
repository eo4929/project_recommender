from __future__ import (division, print_function,
                        unicode_literals)

import pyximport; pyximport.install()

from dataset import Dataset
from reader import Reader
from six import get_unbound_function
from os.path import expanduser

import sys
import numpy as np
from algo_base import AlgoBase
import matrixFactorization as MF
from knns import SymmetricAlgo,KNNBasic,KNNBaseline,KNNWithMeans,KNNWithZScore
#import NMF

def getInputFile(argFile,saving_variable):
    with open(argFile, 'r') as f:
        row_list = f.read().split('\n')

        for row in row_list:
            row = row.split('\t')
            #print('each obj: ',end='')
            #print(obj)
            saving_variable.append(row)

        saving_variable.pop()
        print()


reader = Reader(line_format='user item rating timestamp', sep='\t')

bsl_options = {'method': 'sgd',
               'n_epochs': 20,
               }
sim_options = {'name': 'pearson_baseline','shrinkage': 100}

#algo = MF.NMF(sim_options=sim_options,n_factors=20,n_epochs=500,reg_pu=0.01,reg_qi=0.01,biased=True,lr_bu=0.0033,lr_bi=0.0033,reg_bu=0.01,reg_bi=0.01)
#algo = KNNBasic(sim_options=sim_options)
#algo = KNNWithMeans(sim_options=sim_options,bsl_options=bsl_options)
algo = KNNBaseline(sim_options=sim_options,bsl_options=bsl_options)
#algo = KNNWithZScore(sim_options=sim_options,bsl_options=bsl_options)

if __name__ == '__main__':
    
    argv = sys.argv
    
    training_file = argv[1]
    test_file = argv[2]

    training_set = list()
    test_set = list()
    getInputFile(training_file,training_set)
    getInputFile(test_file,test_set)

    filepath1 = 'C:\\Users\\eo492\\' + training_file
    filepath2 = 'C:\\Users\\eo492\\' + test_file
    
    data = Dataset.load_from_file(filepath1, reader=reader)
    training_set = data.build_full_trainset()
    newModel = algo.fit(training_set)
    
    prediction = algo.test(test_set)

    frontName = test_file.split('.')
    frontName = frontName[0]
    outputfile = frontName + '.base_prediction.txt'
    with open(outputfile, 'a') as f:
        for (uid,iid,_,est,_) in prediction:
            f.write( str(uid) + '\t' + str(iid) + '\t' + str(est) + '\n' )
    
    #algo.rmse(prediction)