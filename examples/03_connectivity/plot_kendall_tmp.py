"""
Producing single subject maps of seed-based correlation
========================================================

This example shows how to produce seed-based correlation maps for a single
subject based on resting-state fMRI scans. These maps depict the temporal
correlation of a **seed region** with the **rest of the brain**.

This example is an advanced one that requires manipulating the data with numpy.
Note the difference between images, that lie in brain space, and the
numpy array, corresponding to the data inside the mask.
"""

# author: Franz Liem


##########################################################################
# Getting the data
# ----------------

# We will work with the first subject of the adhd data set.
# adhd_dataset.func is a list of filenames. We select the 1st (0-based)
# subject by indexing with [0]).
from nilearn import datasets

adhd_dataset = datasets.fetch_adhd(n_subjects=1)
func_filename = adhd_dataset.func[0]
confound_filename = adhd_dataset.confounds[0]

from nilearn import image
msdl_data = datasets.fetch_atlas_msdl()
msdl_filename = msdl_data.maps
index = 5
insula = image.index_img(msdl_filename, index)
binary_insula = image.math_img("img > {0}".format(insula.get_data().max() * .6),
                               img=insula)
insula_coords = msdl_data.region_coords[index]
#
#atlas_imgs = image.iter_img(msdl_filename)
#log_p_values_img = new_img_like(fmri_img, log_p_values)

##########################################################################
# Time series extraction
# ----------------------
#
# We are going to extract signals from the functional time series in two
# steps. First we will extract the mean signal within the **seed region of
# interest**. Second, we will extract the **brain-wide voxel-wise time series**.
#
# We will be working with one seed sphere in the Posterior Cingulate Cortex,
# considered part of the Default Mode Network.
import numpy as np
dosenbach = datasets.fetch_coords_dosenbach_2010()
seeds = np.array([[roi['x'], roi['y'], roi['z']] for roi in dosenbach.rois])
insula_index = np.argmin(np.linalg.norm(insula_coords - seeds, axis=1))
seeds = seeds[insula_index]

from nilearn import plotting
mean_func_img = image.mean_img(func_filename)
plotting.plot_roi(binary_insula, bg_img=mean_func_img, cut_coords=seeds)
plotting.show()

##########################################################################
# Compute the minimal distance between ROIs
#from scipy import spatial
#distances = spatial.distance.squareform(spatial.distance.pdist(seeds))
#print(distances[np.triu_indices_from(distances, k=1)].min())
##########################################################################
# Next, we can proceed similarly for the **brain-wide voxel-wise time
# series**, using :class:`nilearn.input_data.NiftiMasker` with the same input
# arguments as in the seed_masker in addition to smoothing with a 6 mm kernel
from nilearn import input_data
insula_masker = input_data.NiftiMasker(
    mask_img=binary_insula,
    detrend=True, normalize="std",
    memory='nilearn_cache', memory_level=1, verbose=0)

insula_time_series = insula_masker.fit_transform(func_filename,
                                                 confounds=[confound_filename])
insula_ranks = np.argsort(insula_time_series, axis=0) + 1


##########################################################################
# We use :class:`nilearn.input_data.NiftiSpheresMasker` to extract the
# **time series from the functional imaging within the sphere**. The
# sphere is centered at pcc_coords and will have the radius we pass the
# NiftiSpheresMasker function (here 8 mm).
#
# The extraction will also detrend, standardize, and bandpass filter the data.
# This will create a NiftiSpheresMasker object.

seed_time_series = {}
radii = [5., 8., 12.]
kundall = {}
from nilearn.input_data.nifti_spheres_masker import _iter_signals_from_spheres
import nibabel
from nilearn import signal
cleaned_spheres = {}
seed_masker = {}
for radius in radii:
    seed_masker[radius] = input_data.NiftiSpheresMasker(
        [seeds], radius=radius,
#        mask_img=binary_insula,
        detrend=True, normalize="std",
        memory='nilearn_cache', memory_level=1, verbose=0)
    seed_masker[radius].fit()
    sphere = _iter_signals_from_spheres(
        seed_masker[radius].seeds_, nibabel.load(func_filename),
        seed_masker[radius].radius,
        seed_masker[radius].allow_overlap,
        mask_img=seed_masker[radius].mask_img).next()
    cleaned_spheres[radius] = signal.clean(
        sphere, confounds=[confound_filename])
    cleaned_spheres[radius] = sphere
    seed_time_series[radius] = seed_masker[radius].fit_transform(
        func_filename, confounds=[confound_filename])

    # time courses matrix
#    time_series = np.hstack((seed_time_series[radius], insula_time_series))
    time_series = cleaned_spheres[radius].copy()
    print time_series.shape
    time_series -= time_series.mean(axis=1)[:, np.newaxis]

#    from sklearn import covariance
#    covariance_estimator = covariance.EmpiricalCovariance()
#    covariance_estimator.fit(time_series.T)
#    covariance_matrix = covariance_estimator.covariance_
#    ts_ranks = covariance_matrix.sum(axis=0)
    ts_ranks = np.sum(time_series ** 2, axis=0)
    n_times, n_voxels = time_series.shape
    constant = (n_voxels ** 2) * n_times * (n_times ** 2 - 1) / (12. * (
        n_voxels - 1))
    kundall[radius] = 1. - ts_ranks / constant

    # sort time series
#    ts_ranks = np.argsort(seed_time_series[radius], axis=0) + 1
    # compute KCC of insula voxels with the sphere average
#    kundall[radius] = np.sum((np.sum(insula_ranks, axis=1) + ts_ranks) ** 2) -\
#                       n * ((n + 1) * k / 2) ** 2
#    kundall[radius] /= (k ** 2) * (n ** 3 - n) / 12.
#    assert(np.sum(ts_ranks) == n_voxels * n_times * (n_times + 1) / 2)

# Kendall's coefficient concordance (KCC) can measure the similarity of
# a number of time series.

import matplotlib.pyplot as plt
#plt.plot(insula_time_series, '--')
for radius in radii:
    plt.plot(seed_time_series[radius], label=radius, linewidth=3)
plt.xlabel('Scan number')
plt.ylabel('Normalized signal')
plt.tight_layout()
plt.legend()
plt.show()

seed_based_correlation_img = {}
seed_based_correlations = {}
for radius in radii:
#    seed_ts = seed_time_series[radius]
#    seed_based_correlations[radius] = np.dot(insula_time_series.T, seed_ts) / \
#                              seed_ts.shape[0]       
#    seed_based_correlation_img[radius] = insula_masker.inverse_transform(
#        seed_based_correlations[radius].T)
    seed_based_correlation_img[radius] = seed_masker[radius].inverse_transform(
        kundall[radius].T)


##########################################################################
# Plotting the seed-based correlation map
# ---------------------------------------
# We can also plot this image and perform thresholding to only show values
# more extreme than +/- 0.3. Furthermore, we can display the location of the
# seed with a sphere and set the cross to the center of the seed region of
# interest.
from nilearn import plotting
for radius in radii:
    display = plotting.plot_stat_map(seed_based_correlation_img[radius],
#                                     threshold=0.3,
                                     cut_coords=seeds,
                                     title='{}'.format(radius))
plt.figure()
plt.boxplot([kundall[radius] for radius in radii],
            whis=10., showmeans=True, labels=radii)
plt.show()