
import os
import urllib.request
import nibabel as nib
import numpy as np
from morphint.morphint import morphint


if __name__ == "__main__":
    clobber = True
    input_http_path = "https://ftp.bigbrainproject.org/bigbrain-ftp/BigBrainRelease.2015/3D_Volumes/MNI-ICBM152_Space/nii/full16_200um_2009b_sym.nii.gz"


    tmp_dir = "/tmp/brainbuilder_test_data"

    input_local_path = f"{tmp_dir}/full16_200um_2009b_sym.nii.gz"

    if not os.path.exists(input_local_path) or clobber:
        print(f"Downloading {input_http_path} to {input_local_path}...")
        urllib.request.urlretrieve(input_http_path, input_local_path)
        print("Download complete.")

    img = nib.load(input_local_path)
    ar = np.array(img.dataobj, dtype=np.uint8)

    sampling_rate = 3

    # keep evevy one in sampling rate y sections, set rest to zero
    reduced_ar = np.zeros_like(ar)
    reduced_ar[:, ::sampling_rate, :] = ar[:, ::sampling_rate, :]

    reduced_img = nib.Nifti1Image(reduced_ar, img.affine)
    reduced_filename = f"{tmp_dir}/reduced_sampling_rate_{sampling_rate}_input.nii.gz"
    reduced_img.to_filename(reduced_filename)


    interp_filename, _ = morphint(reduced_filename, tmp_dir, 0.4, [3,2,1,0.8,0.4], clobber=clobber)

    # Calculate correlation between original and interpolated data in the missing sections
    interp_img = nib.load(interp_filename)
    interp_ar = np.array(interp_img.dataobj, dtype=np.uint8)
    missing_mask = reduced_ar == 0
    original_values = ar[missing_mask]
    interp_values = interp_ar[missing_mask] 
    correlation = np.corrcoef(original_values.flatten(), interp_values.flatten())[0, 1]
    print(f"Correlation between original and interpolated values in missing sections: {correlation}")


    


