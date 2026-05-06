"""Rigid alignment of sparse 2D sections within a volume.

Designed to fit alongside `morphint.nl_deformation_flow*` and
`section_refine.refine_volume`, with the same volume-in / volume-out
interface and the same ANTs subprocess style. The intended pipeline is:

    vol_aligned, _ = rigid_align_volume(vol, ...)         # this module
    vol_refined, _ = section_refine.refine_volume(vol_aligned, ...)
    interp_vol     = morphint.nl_deformation_flow_3d(vol_refined, ...)

The implementation is deliberately minimal: each non-anchor section is
rigid-registered against its already-aligned neighbour using ANTs' Rigid
transform model, and transforms compose along chains rooted at a single
anchor section (the middle valid section by default). The simple-default
case is handled cleanly; harder cases (heterogeneous contrast,
multi-acquisition stacks, alignment to a 3D template) are explicitly out
of scope and should be handled by BrainBuilder upstream.

One practical robustness extension is included: when a chain registration
gives a poor metric, the section retries against the next-but-one
already-aligned predecessor, skipping over a presumably-bad immediate
neighbour. This is much simpler than a full graph-based optimiser but
catches the common failure mode where one bad pair-registration
poisons the rest of the chain. See `_metric_is_bad` and the retry path
in `_align_chain_outward`.
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Optional

import numpy as np


import morphint.ants_nibabel as nib  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Pairwise rigid registration via ANTs subprocess
# ---------------------------------------------------------------------------

def compute_ants_rigid_alignment(
    prefixdir: str,
    moving_path: str,
    fixed_path: str,
    moving_idx: int,
    fixed_idx: int,
    fwd_tfm_path: Optional[str] = None,
    moving_rsl_path: Optional[str] = None,
    resolution_list: list = (4, 2, 1, 0.5),
    resolution: float = 0.5,
    base_iter: int = 100,
    clobber: bool = False,
) -> tuple:
    """Compute a 2-D rigid transform aligning `moving_path` onto `fixed_path`.

    Mirrors `morphint.compute_ants_alignment` in shape and conventions but
    requests a Rigid transform with a Mattes mutual-information metric --
    a pairing that is more robust than CC for image pairs with subtly
    different intensity statistics, which is common across adjacent
    histology sections.

    Returns
    -------
    (fwd_tfm_path, moving_rsl_path)
        Paths to the saved transform (.mat) and the resampled moving image.
    """
    outprefix = f"{prefixdir}/rigid_{fixed_idx}_{moving_idx}_"
    os.makedirs(prefixdir, exist_ok=True)

    if fwd_tfm_path is None:
        fwd_tfm_path = f"{outprefix}0GenericAffine.mat"
    if moving_rsl_path is None:
        moving_rsl_path = f"{outprefix}rsl.nii.gz"

    if (os.path.exists(fwd_tfm_path) and os.path.exists(moving_rsl_path)
            and not clobber):
        return fwd_tfm_path, moving_rsl_path

    # Build a multi-resolution schedule the same way morphint does.
    try:
        from brainbuilder.utils.utils import AntsParams
        params = AntsParams(list(resolution_list), resolution, base_iter)
        itr_str, f_str, s_str = params.itr_str, params.f_str, params.s_str
    except ImportError:
        # Fallback: build a basic schedule by hand. We map resolution_list
        # to shrink factors via factor = round(scale / resolution), and
        # smoothing sigmas via sigma = scale / 2 (vox).
        scales = list(resolution_list)
        f_list = [max(1, int(round(s / resolution))) for s in scales]
        s_list = [s / 2.0 for s in scales]
        itrs = [int(base_iter * (s / resolution)) for s in scales]
        f_str = "x".join(str(f) for f in f_list)
        s_str = "x".join(f"{s:g}" for s in s_list) + "vox"
        itr_str = "[" + "x".join(str(i) for i in itrs) + ",1e-7,20]"

    cmd = (
        "antsRegistration --verbose 0 --dimensionality 2 --float 0 "
        "--collapse-output-transforms 1 --use-histogram-matching 0 "
        f"--output [ {outprefix},{moving_rsl_path} ] "
        "--interpolation Linear "
        "--initial-moving-transform "
        f"[ {fixed_path},{moving_path},1 ] "
        "--transform Rigid[ 0.1 ] "
        f"--metric Mattes[ {fixed_path},{moving_path},1,32,Random,0.5 ] "
        f"--convergence {itr_str} "
        f"--shrink-factors {f_str} "
        f"--smoothing-sigmas {s_str} "
        "--winsorize-image-intensities [ 0.005,0.995 ]"
    )
    result = subprocess.run(
        cmd, shell=True, executable="/bin/bash",
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"antsRegistration (Rigid) failed (exit {result.returncode}):\n"
            f"  cmd: {cmd}\n"
            f"  stdout (tail): ...{result.stdout[-500:] if result.stdout else ''}\n"
            f"  stderr (tail): ...{result.stderr[-500:] if result.stderr else ''}"
        )

    if not os.path.exists(fwd_tfm_path):
        raise RuntimeError(
            f"ANTs rigid registration returned 0 but produced no transform "
            f"at {fwd_tfm_path}; command was:\n{cmd}"
        )
    return fwd_tfm_path, moving_rsl_path


# ---------------------------------------------------------------------------
# Transform composition and image resampling helpers
# ---------------------------------------------------------------------------

def _ants_compose_to_image(
    moving_path: str,
    reference_path: str,
    transform_chain: list,
    out_path: str,
    interpolator: str = "Linear",
    clobber: bool = False,
) -> str:
    """Apply a chain of transforms to `moving_path` and write to `out_path`.

    `transform_chain` is the LOGICAL order of transforms -- transform_chain[0]
    is the first transform in the chain (closest to the anchor) and
    transform_chain[-1] is the last. ANTs' command-line convention is that
    transforms are applied to coordinates in the order they are listed
    (rightmost first when reading the data flow). We therefore pass the
    chain to ANTs in the same order it appears in the list, which is what
    your existing `align_neighbours_to_fixed` does.
    """
    if os.path.exists(out_path) and not clobber:
        return out_path

    if transform_chain:
        tfm_args = " ".join(f"-t {t}" for t in transform_chain)
    else:
        tfm_args = ""

    cmd = (
        f"antsApplyTransforms -v 0 -d 2 -i {moving_path} "
        f"-r {reference_path} {tfm_args} -o {out_path} -n {interpolator}"
    )
    result = subprocess.run(
        cmd, shell=True, executable="/bin/bash",
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"antsApplyTransforms failed (exit {result.returncode}):\n"
            f"  cmd: {cmd}\n"
            f"  stdout: {result.stdout}\n"
            f"  stderr: {result.stderr}"
        )
    if not os.path.exists(out_path):
        raise RuntimeError(
            f"antsApplyTransforms returned 0 but produced no output at "
            f"{out_path}; cmd was:\n{cmd}"
        )
    return out_path


def _metric_is_bad(fixed_arr: np.ndarray, resampled_arr: np.ndarray,
                   threshold: float = 0.6) -> bool:
    """Return True if the registration looks suspect.

    Computes the foreground-restricted normalised cross-correlation between
    the fixed image and the registration output. "Foreground" is the union
    of nonzero voxels in either image. NCC is a robust similarity proxy
    that's cheap to compute on any 2-D pair and produces values in roughly
    [-1, 1], with values near 1 indicating a good match.

    The threshold is a heuristic. With clean tissue-segmentation inputs
    (the morphint use case) NCC is typically 0.85+ for a good registration
    and drops below 0.6 when the registration has failed catastrophically.
    """
    fg = (fixed_arr > 0) | (resampled_arr > 0)
    if fg.sum() < 50:
        return True  # too little overlap to evaluate
    f = fixed_arr[fg].astype(np.float32)
    m = resampled_arr[fg].astype(np.float32)
    f -= f.mean(); m -= m.mean()
    denom = np.sqrt((f * f).sum() * (m * m).sum())
    if denom < 1e-9:
        return True
    ncc = float((f * m).sum() / denom)
    return ncc < threshold


# ---------------------------------------------------------------------------
# Per-section bookkeeping and chain alignment
# ---------------------------------------------------------------------------

def _write_section(
    arr2d: np.ndarray, origin: tuple, spacing: tuple, path: str,
    clobber: bool = False,
) -> str:
    """Write a 2-D section to disk as a NIfTI for ANTs to read."""
    if os.path.exists(path) and not clobber:
        return path
    affine = np.eye(4)
    affine[0, 0] = spacing[0]; affine[1, 1] = spacing[1]
    affine[0, 3] = origin[0];  affine[1, 3] = origin[1]
    nib.Nifti1Image(arr2d.astype(np.float32), affine,
                    direction_order="lpi").to_filename(path)
    return path


def _read_section(path: str) -> np.ndarray:
    img = nib.load(path)
    arr = img.get_fdata()
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def _align_chain_outward(
    sorted_valid: list,
    anchor_idx_pos: int,
    direction: int,
    section_paths: dict,
    transforms: dict,
    aligned_paths: dict,
    work_dir: str,
    resolution_list: list,
    resolution: float,
    base_iter: int,
    enable_skip_retry: bool,
    metric_threshold: float,
    clobber: bool = False,
) -> None:
    """Sweep outward from the anchor in one direction (+1 or -1), aligning
    each section to its (already aligned) predecessor along the sweep.

    `direction=+1` moves toward higher y-indices (sections after the anchor);
    `direction=-1` toward lower indices.

    Updates `transforms` and `aligned_paths` in place. `transforms[y]`
    is the ordered list of pairwise transforms that, applied to the
    original section, produces the aligned section.

    If `enable_skip_retry` is True, when the immediate-predecessor
    registration produces a poor metric we retry against the predecessor
    one further back along the chain; whichever gives the better metric
    wins. This catches the common failure mode where one bad pair
    poisons the rest of the chain without requiring the more elaborate
    machinery of graph-based registration methods.
    """
    n = len(sorted_valid)
    # range over chain positions outward from anchor
    if direction > 0:
        positions = range(anchor_idx_pos + 1, n)
    else:
        positions = range(anchor_idx_pos - 1, -1, -1)

    for pos in positions:
        y_moving = sorted_valid[pos]
        y_predecessor = sorted_valid[pos - direction]

        # Standard chain step: register the original moving section against
        # the original predecessor section. The pairwise transform `fwd`
        # maps predecessor-frame coordinates to moving-frame coordinates;
        # composed with the predecessor's own chain (predecessor->anchor)
        # it gives moving->anchor.
        #
        # Why register against the ORIGINAL predecessor rather than the
        # already-aligned one: the aligned image has been resampled, which
        # introduces interpolation blur that compounds along the chain.
        # Registering between two original sections gives a sharper
        # pairwise registration result, and explicit chain composition
        # propagates the alignment context. This matches the convention
        # used by `align_neighbours_to_fixed` in BrainBuilder's initalign.
        chain_dir = os.path.join(work_dir, f"chain_y{y_moving}")
        fwd, _rsl_pair = compute_ants_rigid_alignment(
            prefixdir=chain_dir,
            moving_path=section_paths[y_moving],
            fixed_path=section_paths[y_predecessor],
            moving_idx=y_moving,
            fixed_idx=y_predecessor,
            resolution_list=resolution_list,
            resolution=resolution,
            base_iter=base_iter,
            clobber=clobber,
        )
        # Build the full chain by extending the predecessor's chain.
        chain_pred = transforms[y_predecessor] + [fwd]
        rsl_chain = os.path.join(work_dir, f"aligned_y{y_moving}.nii.gz")
        # Reference grid is the predecessor's grid (= anchor grid since the
        # whole chain has been aligned to anchor frame). Use the predecessor's
        # ORIGINAL section as the reference -- we sample onto its grid.
        _ants_compose_to_image(
            moving_path=section_paths[y_moving],
            reference_path=section_paths[y_predecessor],
            transform_chain=chain_pred,
            out_path=rsl_chain,
            clobber=clobber,
        )

        chosen_chain = chain_pred
        chosen_rsl = rsl_chain

        # Optional skip-retry: if the result looks bad and there's a
        # second-predecessor available, try registering against that and
        # take whichever is better. The second predecessor is two chain
        # positions back, so we register against its ORIGINAL section
        # and compose with its chain (same convention as above).
        if enable_skip_retry:
            fixed_arr = _read_section(section_paths[y_predecessor])
            rsl_arr = _read_section(rsl_chain)
            primary_bad = _metric_is_bad(
                fixed_arr, rsl_arr, threshold=metric_threshold
            )
            second_pos = pos - 2 * direction
            if primary_bad and 0 <= second_pos < n:
                y_second = sorted_valid[second_pos]
                skip_dir = os.path.join(work_dir, f"chain_y{y_moving}_skip")
                fwd_skip, _ = compute_ants_rigid_alignment(
                    prefixdir=skip_dir,
                    moving_path=section_paths[y_moving],
                    fixed_path=section_paths[y_second],
                    moving_idx=y_moving,
                    fixed_idx=y_second,
                    resolution_list=resolution_list,
                    resolution=resolution,
                    base_iter=base_iter,
                    clobber=clobber,
                )
                chain_skip = transforms[y_second] + [fwd_skip]
                rsl_skip = os.path.join(
                    work_dir, f"aligned_y{y_moving}_skip.nii.gz"
                )
                _ants_compose_to_image(
                    moving_path=section_paths[y_moving],
                    reference_path=section_paths[y_second],
                    transform_chain=chain_skip,
                    out_path=rsl_skip,
                    clobber=clobber,
                )
                skip_fixed = _read_section(section_paths[y_second])
                skip_rsl = _read_section(rsl_skip)
                skip_bad = _metric_is_bad(
                    skip_fixed, skip_rsl, threshold=metric_threshold
                )
                # Only switch if the skip retry is *not* bad. If both are
                # bad, the immediate-predecessor chain is closer to what
                # the rest of the stack expects.
                if not skip_bad:
                    chosen_chain = chain_skip
                    chosen_rsl = rsl_skip

        transforms[y_moving] = chosen_chain
        aligned_paths[y_moving] = chosen_rsl


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def rigid_align_volume(
    vol: np.ndarray,
    output_dir: str,
    origin: tuple,
    spacing: tuple,
    anchor_index: Optional[int] = None,
    resolution_list: list = (4, 2, 1, 0.5),
    resolution: float = 0.5,
    base_iter: int = 100,
    enable_skip_retry: bool = True,
    metric_threshold: float = 0.6,
    clobber: bool = False,
) -> tuple:
    """Rigidly align valid sections within a sparse 2-D-section volume.

    Parameters
    ----------
    vol : np.ndarray
        3-D numpy array, sections along axis=1. Sections that are entirely
        zero are treated as missing and skipped.
    output_dir : str
        Working directory for transforms and intermediate sections.
    origin, spacing : tuple of (float, float)
        2-D origin / spacing for each section, matching morphint's
        convention (`affine[[0, 2], 3]` and `affine[[0, 2], [0, 2]]`).
    anchor_index : int, optional
        y-index of the section to use as anchor (its position is fixed
        by the identity transform). Defaults to the median valid index.
    resolution_list, resolution : list, float
        Multi-resolution schedule for ANTs (coarse-to-fine).
    base_iter : int
        Base ANTs iteration count at the finest level. Scales up at
        coarser levels via `AntsParams`.
    enable_skip_retry : bool
        If True, when a chain step's metric is poor, retry against the
        next-but-one predecessor and keep whichever gives a better
        registration. Defends against the common failure mode where one
        bad pair-registration poisons the rest of the chain.
    metric_threshold : float
        NCC value below which a registration is flagged as bad and
        triggers the skip-retry path.
    clobber : bool
        Overwrite existing intermediate files.

    Returns
    -------
    aligned_vol : np.ndarray
        A volume the same shape as `vol`, with each valid section replaced
        by its rigid-aligned version (sampled on the same grid). Empty
        slices remain empty.
    transforms : dict
        Map from y-index to the ordered list of pairwise transform paths
        that produces the aligned section from the original. Useful for
        applying the same alignment to a different volume (e.g. a higher-
        resolution version of the same sections).
    """
    out_vol = vol.copy()
    valid_idx = np.where(np.max(vol, axis=(0, 2)) > 0)[0]
    if len(valid_idx) < 2:
        # 0 or 1 valid sections: nothing to align.
        return out_vol, {}
    sorted_valid = [int(y) for y in sorted(int(y) for y in valid_idx)]

    # Anchor: middle valid section by default. With even-length stacks
    # this is the lower of the two middles, which matches the convention
    # in your existing initalign.adjust_alignment.
    if anchor_index is None:
        anchor_index = sorted_valid[len(sorted_valid) // 2]
    if anchor_index not in sorted_valid:
        raise ValueError(
            f"anchor_index {anchor_index} is not a valid (nonzero) "
            f"section index in vol; valid indices are {sorted_valid[:5]} ..."
        )
    anchor_idx_pos = sorted_valid.index(anchor_index)

    # Workspace setup
    os.makedirs(output_dir, exist_ok=True)
    sections_dir = os.path.join(output_dir, "sections")
    os.makedirs(sections_dir, exist_ok=True)
    chain_dir = os.path.join(output_dir, "chain")
    os.makedirs(chain_dir, exist_ok=True)

    # Write each valid section to disk for ANTs to read.
    section_paths = {
        y: _write_section(
            vol[:, y, :], origin, spacing,
            os.path.join(sections_dir, f"section_{y}.nii.gz"),
            clobber=clobber,
        )
        for y in sorted_valid
    }

    # Anchor: identity transform, aligned image = original.
    transforms: dict = {anchor_index: []}
    aligned_paths = {anchor_index: section_paths[anchor_index]}

    # Sweep outward in both directions.
    for direction in (+1, -1):
        _align_chain_outward(
            sorted_valid=sorted_valid,
            anchor_idx_pos=anchor_idx_pos,
            direction=direction,
            section_paths=section_paths,
            transforms=transforms,
            aligned_paths=aligned_paths,
            work_dir=chain_dir,
            resolution_list=list(resolution_list),
            resolution=resolution,
            base_iter=base_iter,
            enable_skip_retry=enable_skip_retry,
            metric_threshold=metric_threshold,
            clobber=clobber,
        )

    # Pack aligned sections back into the volume.
    for y in sorted_valid:
        aligned_arr = _read_section(aligned_paths[y])
        # Resample shape if antsApplyTransforms produced a slightly different
        # grid (it shouldn't, since reference == moving for the chain
        # composition, but handle defensively).
        if aligned_arr.shape != vol[:, y, :].shape:
            from skimage.transform import resize
            aligned_arr = resize(
                aligned_arr, vol[:, y, :].shape,
                order=1, preserve_range=True, anti_aliasing=False,
            ).astype(np.float32)
        out_vol[:, y, :] = aligned_arr

    return out_vol, transforms


def rigid_align_nii(
    in_fin: str,
    out_fin: str,
    output_dir: str,
    anchor_index: Optional[int] = None,
    resolution_list: list = (4, 2, 1, 0.5),
    resolution: float = 0.5,
    base_iter: int = 100,
    enable_skip_retry: bool = True,
    metric_threshold: float = 0.6,
    clobber: bool = False,
) -> dict:
    """File-in / file-out wrapper. Mirrors `morphint.nl_deformation_flow_nii`."""
    tfm_json = out_fin.replace(".nii.gz", "") + "_rigid_tfm.json"

    if (not os.path.exists(out_fin)
            or not os.path.exists(tfm_json)
            or clobber):
        in_img = nib.load(in_fin)
        in_vol = in_img.get_fdata()
        in_vol[in_vol < 0] = 0

        origin = tuple(in_img.affine[[0, 2], 3])
        spacing = tuple(in_img.affine[[0, 2], [0, 2]])

        aligned_vol, transforms = rigid_align_volume(
            in_vol,
            output_dir + "/rigid_align/",
            origin=origin,
            spacing=spacing,
            anchor_index=anchor_index,
            resolution_list=resolution_list,
            resolution=resolution,
            base_iter=base_iter,
            enable_skip_retry=enable_skip_retry,
            metric_threshold=metric_threshold,
            clobber=clobber,
        )

        out_img = nib.Nifti1Image(
            aligned_vol, in_img.affine, direction_order="lpi"
        )
        out_img.to_filename(out_fin)

        # Serialise transforms for re-use (e.g. resampling a higher-res
        # volume of the same sections through the same alignment).
        tfm_serialisable = {str(y): list(chain) for y, chain in transforms.items()}
        with open(tfm_json, "w") as f:
            json.dump(tfm_serialisable, f)
    else:
        with open(tfm_json) as f:
            tfm_serialisable = json.load(f)
        transforms = {int(y): list(chain) for y, chain in tfm_serialisable.items()}

    return transforms


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rigidly align sparse 2D sections within a NIfTI volume."
    )
    parser.add_argument("in_fin", help="Input NIfTI volume.")
    parser.add_argument("out_fin", help="Output NIfTI path.")
    parser.add_argument("output_dir", help="Working directory for transforms.")
    parser.add_argument("--anchor-index", type=int, default=None,
                        help="y-index of the anchor section. Defaults to the "
                             "median valid section.")
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--resolution-list", type=float, nargs="+",
                        default=None)
    parser.add_argument("--base-iter", type=int, default=100)
    parser.add_argument("--metric-threshold", type=float, default=0.6,
                        help="NCC threshold below which a chain step is "
                             "flagged as bad and skip-retried.")
    parser.add_argument("--no-skip-retry", action="store_true",
                        help="Disable the skip-retry path; use plain chain "
                             "registration only.")
    parser.add_argument("--clobber", action="store_true")
    args = parser.parse_args()

    rigid_align_nii(
        in_fin=args.in_fin,
        out_fin=args.out_fin,
        output_dir=args.output_dir,
        anchor_index=args.anchor_index,
        resolution_list=tuple(args.resolution_list) if args.resolution_list
                       else (4, 2, 1, 0.5),
        resolution=args.resolution,
        base_iter=args.base_iter,
        enable_skip_retry=not args.no_skip_retry,
        metric_threshold=args.metric_threshold,
        clobber=args.clobber,
    )