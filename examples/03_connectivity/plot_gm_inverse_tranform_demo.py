"""
Inverse transform
====================================

"""
stop
from nilearn import datasets, input_data
adhd_dataset = datasets.fetch_adhd(n_subjects=1)
func_filename = adhd_dataset.func[0]
icbm152_grey_mask = datasets.fetch_icbm152_brain_gm_mask()
gm_masker = input_data.NiftiMasker(mask_img=icbm152_grey_mask,
                                   normalize="std")
gm_voxels_time_series = gm_masker.fit_transform(func_filename)
print('Check no zero time series: min std is {0}'.format(
    gm_voxels_time_series.std(axis=0).min()))

# Select some random voxels
import numpy as np
import nibabel
niimg = nibabel.load(func_filename)
affine = niimg.affine

from nilearn.image.resampling import coord_transform
sx, sy, sz = (-46, -68, 32)
seeds = [(sx, sy, sz)]
nearest = np.round(coord_transform(sx, sy, sz, np.linalg.inv(affine)))
nearest = nearest.astype(int)
nearest = (nearest[0], nearest[1], nearest[2])

from nilearn import image, masking, _utils
mask_img = _utils.check_niimg_3d(icbm152_grey_mask)
mask_img = image.resample_img(mask_img, target_affine=affine,
                              target_shape=niimg.shape[:3],
                              interpolation='nearest')
mask, _ = masking._load_mask_img(mask_img)
mask_coords = list(zip(*np.where(mask != 0)))

nearests = []
try:
    nearests.append(mask_coords.index(nearest))
except ValueError:
    print('{0} not in mask'.format(nearest))
    nearests.append(None)

mask_coords = np.asarray(list(zip(*mask_coords)))
mask_coords = coord_transform(mask_coords[0], mask_coords[1],
                              mask_coords[2], affine)
mask_coords = np.asarray(mask_coords).T

from sklearn import neighbors
clf = neighbors.NearestNeighbors(radius=6.)
A = clf.fit(mask_coords).radius_neighbors_graph(seeds)
A = A.tolil()
for i, nearest in enumerate(nearests):
    if nearest is None:
        continue
    A[i, nearest] = True


mask_coords = mask_coords.astype(int).tolist()
for i, seed in enumerate(seeds):
    try:
        A[i, mask_coords.index(seed)] = True
    except ValueError:
        # seed is not in the mask
        pass

X = np.zeros(gm_voxels_time_series.shape[1:], dtype=float)
for i, row in enumerate(A.rows):
    X[row] = 1.

# remove zero indices
voxels_indices = [index for index in mask_coords
                  if gm_voxels_time_series.std(axis=0)[index] > 1e-7]
voxels_indices = row
selected_timeseries = gm_voxels_time_series[:, voxels_indices]
gm_selected_voxels_mask_array = np.zeros(gm_voxels_time_series.shape[1:])
gm_indices = []
for ts in selected_timeseries.T:
    voxel_index = np.argmin(np.linalg.norm(
        gm_voxels_time_series - ts[:, np.newaxis], axis=0))
    gm_indices.append(voxel_index)

gm_selected_voxels_mask_array[gm_indices] = 1.
assert(np.all(gm_selected_voxels_mask_array == X))
gm_selected_voxels_mask_img = gm_masker.inverse_transform(
    gm_selected_voxels_mask_array)

# Repeat without GM mask
brain_masker = input_data.NiftiMasker(normalize="std")
brain_voxels_time_series = brain_masker.fit_transform(func_filename)
brain_selected_voxels_mask_array = np.zeros(brain_voxels_time_series.shape[1:])
brain_indices = []
for ts in (selected_timeseries / selected_timeseries.std(axis=0)).T:
    voxel_index = np.argmin(np.linalg.norm(
        brain_voxels_time_series / brain_voxels_time_series.std(axis=0) -
        ts[:, np.newaxis], axis=0))
    if voxel_index in brain_indices:
        raise ValueError('stop')
    brain_indices.append(voxel_index)

brain_selected_voxels_mask_array[brain_indices] = 1.
brain_selected_voxels_mask_img = brain_masker.inverse_transform(
    brain_selected_voxels_mask_array)


np.testing.assert_array_almost_equal(
    brain_voxels_time_series[:, brain_indices],
    gm_voxels_time_series[:, gm_indices])

from nilearn import _utils, plotting
for img, title in zip([brain_selected_voxels_mask_img,
                       gm_selected_voxels_mask_img], ['unmasked', 'GM masked']):
    display = plotting.plot_roi(_utils.check_niimg(img, atleast_4d=True),
#                                display_mode='y', cut_coords=1,
#                                cut_coords=range(75, 80),
                                title=title, bg_img=icbm152_grey_mask)
plotting.show()
