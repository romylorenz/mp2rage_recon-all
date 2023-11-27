# mp2rage_recon-all

usage:

`mp2rage_recon-all.py [-h] [--fs_dir FS_DIR] [--spm_dir SPM_DIR] [--gdc_coeff_file GDC_COEFF_FILE] inv2 uni`

example:
```
mp2rage_recon-all.py --fs_dir freesurfer \
                     --spm_dir /data/pt_02389/Software/spm12/ \
                     --gdc_coeff_file coeff_SC72CD.grad \
                     sub-08_ses-1_inv-2_MP2RAGE.nii \
                     sub-08_ses-1_UNIT1.nii
```

# requirements

own code:
* mp2rage_recon-all.py
* anatomy.py
* cat12_seg.m
* run_gdc.sh

external software:
* freesurfer 7.1
* spm
* cat12
* matlab
* gradient_unwarp.py (https://github.com/Washington-University/gradunwarp, optional)
* fsl (imcp, convertwarp, applywarp)
* python 3

python packages:
* nipype
* nibabel=3 (gradient_unwarp uses functionality deprecated as of version 4.0)
* numpy
(gradient_unwarp.py installation has some additional dependencies)




