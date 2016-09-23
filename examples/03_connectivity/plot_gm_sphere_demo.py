"""
Inverse transform
====================================

"""
stop
from nilearn import datasets, input_data
adhd_dataset = datasets.fetch_adhd(n_subjects=1)
func_filename = adhd_dataset.func[0]
rtj_coords = (-46, -68, 32)
radius = 6.
brain_masker = input_data.NiftiMasker(normalize=None)
brain_voxels_time_series = brain_masker.fit_transform(func_filename)

# Manual extraction of voxel timeseries within the masked sphere
import nibabel
import numpy as np
from nilearn.input_data.nifti_spheres_masker import _iter_signals_from_spheres
sphere_masker = input_data.NiftiSpheresMasker(
    [rtj_coords], normalize=None, radius=radius)
sphere_masker.fit()
func_img = nibabel.load(func_filename)
sphere_time_series = _iter_signals_from_spheres(
    sphere_masker.seeds_, func_img, sphere_masker.radius,
    sphere_masker.allow_overlap, mask_img=sphere_masker.mask_img).next()
print('Check no zero time series: min std is {0}'.format(
    sphere_time_series.std(axis=0).min()))

# Form sphere image
sphere_indicator = np.zeros(brain_voxels_time_series.shape[1:])
indices = []
for ts in (sphere_time_series / sphere_time_series.std(axis=0)).T:
    voxel_index = np.argmin(np.linalg.norm(
        brain_voxels_time_series / brain_voxels_time_series.std(axis=0) -
        ts[:, np.newaxis], axis=0))
    indices.append(voxel_index)
print('outputted voxel signal first time point is {0}'.format(
    brain_voxels_time_series[0, indices]))

sphere_indicator[indices] = 1.
assert(np.sum(sphere_indicator) == sphere_time_series.shape[1])
sphere_indicator_img = brain_masker.inverse_transform(sphere_indicator)

# Repeat wile masking with GM
icbm152_grey_mask = datasets.fetch_icbm152_brain_gm_mask()
mask_img = icbm152_grey_mask
gm_masker = input_data.NiftiMasker(mask_img=mask_img, normalize=None)
gm_voxels_time_series = gm_masker.fit_transform(func_filename)
gm_seed_masker = input_data.NiftiSpheresMasker(
    [rtj_coords], normalize=None, radius=radius, mask_img=mask_img)
gm_seed_masker.fit()
resampled_func_img = gm_masker.inverse_transform(gm_masker.transform(func_img))
gm_sphere_time_series = _iter_signals_from_spheres(
    gm_seed_masker.seeds_, resampled_func_img, gm_seed_masker.radius,
    gm_seed_masker.allow_overlap, mask_img=gm_seed_masker.mask_img).next()
gm_sphere_indicator = np.zeros(gm_voxels_time_series.shape[1:])
indices = []
for ts in (gm_sphere_time_series / gm_sphere_time_series.std(axis=0)).T:
    voxel_index = np.argmin(np.linalg.norm(
        gm_voxels_time_series / gm_voxels_time_series.std(axis=0) -
        ts[:, np.newaxis], axis=0))
    indices.append(voxel_index)

print('outputted GM masked voxel signal first time point is {0}'.format(
    gm_voxels_time_series[0, indices]))

gm_sphere_indicator[indices] = 1.
assert(np.sum(gm_sphere_indicator) == gm_sphere_time_series.shape[1])
gm_sphere_indicator_img = gm_masker.inverse_transform(gm_sphere_indicator)

from nilearn import image, _utils, plotting
for img, title in zip([sphere_indicator_img, gm_sphere_indicator_img],
                      ['unmasked', 'GM masked']):
    img = image.new_img_like(img, np.array(img.get_data(), dtype=float))
    display = plotting.plot_roi(_utils.check_niimg(img, atleast_4d=True),
                                display_mode='z',
                                cut_coords=(33,),  # range(24, 45, 3),
                                title=title, bg_img=mask_img)
plotting.show()
