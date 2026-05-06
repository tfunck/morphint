import os
import subprocess

def compute_ants_alignment(
    prefixdir: str,
    sec0_path: str,
    sec1_path: str,
    ymin: int,
    ymax: int,
    fwd_tfm_path: str = None,
    inv_tfm_path: str = None,
    resolution_list: list = [4, 2, 1, 0.5],
    resolution: float = 0.5,
    base_itr: int = 30,
    clobber: bool = False,
):
    """Compute the ANTs alignment between two sections and save the forward and inverse transforms.

    Args:
        prefixdir (str): Directory to save the output files.
        sec0_path (str): Path to the first section.
        sec1_path (str): Path to the second section.
        ymin (int): Minimum y-coordinate of the section.
        ymax (int): Maximum y-coordinate of the section.
        resolution_list (list): List of resolutions for multi-resolution registration.
        resolution (float): Final resolution for the registration.
        clobber (bool): If True, overwrite existing files.

    Returns:
        fwd_tfm_path (str): Path to the forward transform file.
        inv_tfm_path (str): Path to the inverse transform file.
    """
    outprefix = f"{prefixdir}/deformation_field_{ymin}_{ymax}"
    os.makedirs(prefixdir, exist_ok=True)

    write_composite_transform = 0

    if fwd_tfm_path is None:
        if write_composite_transform:
            fwd_tfm_path = f"{outprefix}_Composite.h5"
        else:
            fwd_tfm_path = f"{outprefix}_0Warp.nii.gz"

    if inv_tfm_path is None:
        if write_composite_transform:
            inv_tfm_path = f"{outprefix}_InverseComposite.h5"
        else:
            inv_tfm_path = f"{outprefix}_0InverseWarp.nii.gz"

    mv_rsl_fn = f"{outprefix}_SyN_GC_cls_rsl.nii.gz"

    if not os.path.exists(fwd_tfm_path) or not os.path.exists(inv_tfm_path) or clobber:
        # Load the sections

        from brainbuilder.utils.utils import AntsParams

        nlParams = AntsParams(resolution_list, resolution, base_itr)

        try:
            cmd = "antsRegistration --verbose 0 --dimensionality 2 --float 0 --collapse-output-transforms 1"
            cmd += f" --output [ {outprefix}_,{mv_rsl_fn},/tmp/tmp.nii.gz ] --interpolation Linear --use-histogram-matching 0 --winsorize-image-intensities [ 0.005,0.995 ]"
            cmd += f" --transform SyN[ 0.1,3,0 ] --metric CC[ {sec1_path},{sec0_path},1,4 ]"
            cmd += f" --convergence  {nlParams.itr_str} --shrink-factors {nlParams.f_str} --smoothing-sigmas {nlParams.s_str} "

            #print(cmd)

            subprocess.run(cmd, shell=True, executable="/bin/bash")

        except RuntimeError as e:
            print("Error in registration:", e)


    assert os.path.exists(fwd_tfm_path), f"Error: output does not exist {fwd_tfm_path}"
    assert os.path.exists(inv_tfm_path), f"Error: output does not exist {inv_tfm_path}"

    return fwd_tfm_path, inv_tfm_path