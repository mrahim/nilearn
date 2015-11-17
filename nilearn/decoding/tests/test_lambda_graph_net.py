# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:51:44 2015

@author: rahim.mehdi@gmail.com
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedShuffleSplit
from nilearn.decoding import SpaceNetClassifier
from nilearn.decoding.objective_functions import (_inv_lambda_matrix,
                                                  _lambda_matrix)
from fetch_data import fetch_adni_longitudinal_fdg_pet, fetch_adni_masks
from fetch_data._utils.utils import _set_classification_data
from sklearn.linear_model import LogisticRegressionCV 
from nilearn.input_data import NiftiMasker


mask = fetch_adni_masks()['mask_pet_longitudinal']
masker = NiftiMasker(mask_img=mask)
masker.fit()

dataset = fetch_adni_longitudinal_fdg_pet()

df = pd.DataFrame(data=dataset)
grouped = df.groupby('subjects').groups

df_count = df.groupby('subjects')['pet'].count()
df_count = df_count[df_count > 2]
df_count = df_count[df_count < 6]

# n_images per subject
img_per_subject = df_count.values
# unique subjects with multiple images
subjects = df_count.keys().values
subj = np.array([dataset.subjects[grouped[s]] for s in subjects])
# diagnosis of the subjects
dx_group = np.hstack([dataset.dx_group[grouped[s][0]] for s in subjects])
dx_all = np.array([dataset.dx_group[grouped[s]] for s in subjects])
# all images of the subjects
pet = np.array([dataset.pet[grouped[s]] for s in subjects])


def set_subjects_splits(subjects, dx_group, pet):
    """Returns X, y
    """
    X, y, idx = _set_classification_data(pet, dx_group, ['AD', 'MCI'],
                                         return_idx=True)
    sss = StratifiedShuffleSplit(y, n_iter=10, test_size=.25,
                                 random_state=42)
    return X, idx, sss


Xa, idx, sss = set_subjects_splits(subjects, dx_group, pet)
dxa = dx_all[idx]
subja = subj[idx]

spn = SpaceNetClassifier(penalty='graph-net',
                         loss='lambda',
                         n_alphas=10,
                         verbose=1,
                         n_jobs=1)

lr = LogisticRegressionCV()


# Test is on subject with multiple known and unknown images.
# So first the covariance matrix is computed.
# Then this matrix is used in a conditional multivariate Gaussian fashion.
# y(u) = X(u).w  + (cov(u,k).v(k)^-1).(y(k) - X(k).w)


def _compute_lambda_covariance(X, gamma=1.0):
    """Returns the lambda covariance.
    X should be (n_img, n_vox).
    output is (n_img, n_img)
    """
    s = np.array([1]*X.shape[0])
    L = _lambda_matrix(s, gamma=gamma).todense()
    # L = _inv_lambda_matrix(s, gamma=gamma).todense()
    return L + np.cov(X)


def _predict_subject(Xu, Xk, yu, yk, w, masker):
    """ test multiple images of one subject
    X: (list of n_imgs) : [Xu, Xk]
    y: target for each image : [yu, yk]
    w: n_voxels
    """
    n_k = np.shape(Xk)[0]
    X_img = np.hstack((Xk, Xu))
    X = masker.transform(X_img)
    X_k = X[:n_k, :]
    X_u = X[n_k:, :]

    covariance = _compute_lambda_covariance(X, gamma=10.0)
    ck = np.linalg.inv(covariance[:n_k, :n_k])
    cku = covariance[n_k:, :n_k]

    d = np.expand_dims(yk, axis=1) - np.dot(X_k, w.T)
    ypred = np.dot(X_u, w.T) + np.dot(np.dot(cku, ck), d)

    return ypred


def spn_predict(spn, xtest, ytest, nb_known=2):
    Xtest_known = xtest[:nb_known]
    ytest_known = ytest[:nb_known]
    Xtest_unknown = xtest[nb_known:]
    ytest_unknown = ytest[nb_known:]
    yp = _predict_subject(Xtest_unknown, Xtest_known,
                          ytest_unknown, ytest_known,
                          spn.coef_, spn.masker_)
    ypred = np.ones(yp.shape)
    ypred[yp < 0] = -1
    return np.hstack((np.expand_dims(ytest_unknown, axis=1), ypred))

accuracy0 = []
accuracy1 = []
accuracy2 = []


def train_and_score_lr(lr, Xa, dxa, subja, train, test):
    Xtrain = np.hstack(Xa[train])
    yt = np.hstack(dxa[train])
    ytrain = - np.ones(yt.shape)
    ytrain[yt == 'AD'] = 1

    xtr = masker.transform(Xtrain)

    lr.fit(xtr, ytrain)

    Xtest = np.hstack(Xa[test])
    yt = np.hstack(dxa[test])
    ytest = - np.ones(yt.shape)
    ytest[yt == 'AD'] = 1

    xte = masker.transform(Xtest)

    return accuracy_score(ytest, lr.predict(xte))


def train_and_score(spn, Xa, dxa, subja, train, test):
    Xtrain = np.hstack(Xa[train])
    yt = np.hstack(dxa[train])
    ytrain = - np.ones(yt.shape)
    ytrain[yt == 'AD'] = 1
    strain = np.hstack(subja[train])
    spn.fit(Xtrain, ytrain, strain, gamma=10.0)

    spn_pure = SpaceNetClassifier(penalty='graph-net',
                                  loss='logistic',
                                  n_alphas=10,
                                  verbose=1,
                                  n_jobs=1)
    spn_pure.fit(Xtrain, ytrain)

    acc1 = []
    acc2 = []
    for t in test:
        Xtest = np.hstack(Xa[t])
        yt = np.hstack(dxa[t])
        ytest = - np.ones(yt.shape)
        ytest[yt == 'AD'] = 1
        yp = spn_predict(spn, Xtest, ytest, nb_known=1)
        acc1.append(yp)
        yp = spn_predict(spn, Xtest, ytest, nb_known=2)
        acc2.append(yp)

    Xtest = np.hstack(Xa[test])
    yt = np.hstack(dxa[test])
    ytest = - np.ones(yt.shape)
    ytest[yt == 'AD'] = 1
    a0 = accuracy_score(ytest, spn.predict(Xtest))

    a3 = accuracy_score(ytest, spn_pure.predict(Xtest))

    pred = np.vstack(acc1)
    a1 = accuracy_score(pred[:, 0], pred[:, 1])
    pred = np.vstack(acc2)
    a2 = accuracy_score(pred[:, 0], pred[:, 1])
    return [a3, a0, a1, a2]


from joblib import Parallel, delayed

# scores = Parallel(n_jobs=10, verbose=5)(
#         delayed(train_and_score_lr)(lr, Xa, dxa, subja, train, test)
#         for train, test in sss)


scores = Parallel(n_jobs=10, verbose=5)(
         delayed(train_and_score)(spn, Xa, dxa, subja, train, test)
         for train, test in sss)


"""
for train, test in sss:
    Xtrain = np.hstack(Xa[train])
    yt = np.hstack(dxa[train])
    ytrain = - np.ones(yt.shape)
    ytrain[yt == 'AD'] = 1
    strain = np.hstack(subja[train])
    spn.fit(Xtrain, ytrain, strain, gamma=5.)

    acc1 = []
    acc2 = []
    for t in test:
        Xtest = np.hstack(Xa[t])
        yt = np.hstack(dxa[t])
        ytest = - np.ones(yt.shape)
        ytest[yt == 'AD'] = 1
        yp = spn_predict(spn, Xtest, ytest, nb_known=1)
        acc1.append(yp)
        yp = spn_predict(spn, Xtest, ytest, nb_known=2)
        acc2.append(yp)
    pred = np.vstack(acc1)
    accuracy1.append(accuracy_score(pred[:, 0], pred[:, 1]))
    pred = np.vstack(acc2)
    accuracy2.append(accuracy_score(pred[:, 0], pred[:, 1]))

np.savez('graphnet_scores',
         accuracy1=accuracy1,
         accuracy2=accuracy2)
"""
