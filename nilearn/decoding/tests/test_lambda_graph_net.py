# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:51:44 2015

@author: rahim.mehdi@gmail.com
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import binarize
from nilearn.decoding import SpaceNetClassifier
from nilearn.decoding.objective_functions import (_inv_lambda_matrix,
                                                  _lambda_matrix)
from fetch_data import fetch_adni_longitudinal_fdg_pet
from fetch_data._utils.utils import (_set_classification_data,
                                     _set_group_indices,
                                     StratifiedSubjectShuffleSplit)
from nilearn.plotting import plot_stat_map


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
# diagnosis of the subjects
dx_group = np.hstack([dataset.dx_group[grouped[s][0]] for s in subjects])
# all images of the subjects
pet = np.array([dataset.pet[grouped[s]] for s in subjects])


def set_subjects_splits(subjects, dx_group, pet):
    """Returns X, y
    """
    X, y = _set_classification_data(pet, dx_group,
                                    ['AD', 'MCI'])
    return X, y


Xa, ya = set_subjects_splits(subjects, dx_group, pet)


subjects = subjects[:20].copy()
img_per_subject = img_per_subject[:20].copy()

idx_u = [np.array(np.where(dataset.subjects == s)).ravel()[0]
         for s in subjects]

idx = [np.array(np.where(dataset.subjects == s)).ravel()
       for s in subjects]
idx = np.hstack(idx)

X, y = _set_classification_data(dataset.pet[idx], dataset.dx_group[idx],
                                ['AD', 'Normal'])

s = [np.array(i*[b]) for b, i in zip(subjects, img_per_subject)]
s = np.hstack(s)

spn = SpaceNetClassifier(penalty='graph-net',
                         loss='lambda',
                         n_alphas=10,
                         verbose=2,
                         n_jobs=10)

# spn.fit(X, y, s, gamma=5.)

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

    covariance = _compute_lambda_covariance(X, gamma=5.)
    ck = np.linalg.inv(covariance[:n_k, :n_k])
    cku = covariance[n_k:, :n_k]

    d = np.expand_dims(yk, axis=1) - np.dot(X_k, w.T)
    ypred = np.dot(X_u, w.T) + np.dot(np.dot(cku, ck), d)

    return ypred


# Test on real data
img_per_subject = df_count.values
subjects = df_count.keys().values


"""
subjects = subjects[20:25].copy()
img_per_subject = img_per_subject[20:25].copy()

idx_u = [np.array(np.where(dataset.subjects == s)).ravel()[0]
         for s in subjects]

idx = [np.array(np.where(dataset.subjects == s)).ravel()
       for s in subjects]
idx = np.hstack(idx)

X, y = _set_classification_data(dataset.pet[idx], dataset.dx_group[idx],
                                ['AD', 'Normal'])

s = [np.array(i*[b]) for b, i in zip(subjects, img_per_subject)]
s = np.hstack(s)

# last subject :
nb_known = 2
i = np.sum(img_per_subject[:-1])
Xtest_known = X[i:i+nb_known]
ytest_known = y[i:i+nb_known]
Xtest_unknown = X[i+nb_known:]
ytest_unknown = y[i+nb_known:]


yp = _predict_subject(Xtest_unknown, Xtest_known,
                      ytest_unknown, ytest_known,
                      spn.coef_, spn.masker_)
"""
