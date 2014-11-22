"""
Kaggle in Class
Predict Movie Ratings

Predict movie ratings for the MovieLens Dataset

Acknowledgements
We thank Movielens for providing this dataset.

https://inclass.kaggle.com/c/predict-movie-ratings

Author: Neil Summers
Using the Repeated Matrix Reconstruction method from
http://cs229.stanford.edu/proj2006/KleemanDenuitHenderson-MatrixFactorizationForCollaborativePrediction.pdf

best solution was average of 2 runs with 15 and 20 SVD components, and 10 iterations each
Scoring     0.87478 Public     0.87376 Private
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD

pd.options.display.max_columns = 10 
pd.options.display.width = 134
pd.options.display.max_rows = 20

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

train = pd.read_csv('train_v2.csv')
test = pd.read_csv('test_v2.csv')
matrix = pd.concat([train,test]).pivot('user','movie','rating')
movie_means = matrix.mean()
user_means = matrix.mean(axis=1)
mzm = matrix-movie_means
mz = mzm.fillna(0)
mask = -mzm.isnull()

iteration = 0
mse_last = 999
while iteration<10:
    iteration += 1
    svd = TruncatedSVD(n_components=15,random_state=42)
    svd.fit(mz)
    mzsvd = pd.DataFrame(svd.inverse_transform(svd.transform(mz)),columns=mz.columns,index=mz.index)

    mse = mean_squared_error(mzsvd[mask].fillna(0),mzm[mask].fillna(0))
    print('%i %.5f %.5f'%(iteration,mse,mse_last-mse))
    mzsvd[mask] = mzm[mask]

    mz = mzsvd
    if mse_last-mse<0.00001: break
    mse_last = mse

m = mz+movie_means
m = m.clip(lower=1,upper=5)

test['rating'] = test.apply(lambda x:m[m.index==x.user][x.movie].values[0],axis=1)

# There are some movies who did not have enough info to make prediction, so just used average value for user
missing = np.where(test.rating.isnull())[0]
test.ix[missing,'rating'] = user_means[test.loc[missing].user].values

test.to_csv('submission.csv',index=False,columns=['ID','rating'])
