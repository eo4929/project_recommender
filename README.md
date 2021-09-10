# project_recommender
Recommender System 

# Setting
- Python 3.7.2

# Requirement
- numpy
- joblib
- scipy
- six
- cython

# Algorithm used
- MF and KNN algorithm

# The detailed Description
- similarity.pyx (in recommender.py): this file has methods that are used for user's similarity measure. As a result of test, I check pearson_baseline measure is the best. pearson_baseline make use of shrunk Pearson correlation coefficient between all of users. So, I do not cosine, msd, pearson function. 
- optimizing_baseline.pyx (in recommender.py): this file defines baseline_sgd function. the function computes  base user and base item value by using given iteration count(n_epochs) and regularization and learning rate.
- class AlgoBase (in algo_base.py): this class is base class for MF class and KNN class.
- class Prediction (in algo_base.py): this class is used as a prediction instance that contains uid, iid, real rating, estimated rating information.

# Metric for evaluation
- RMSE
