"""
section_refine.py
=================

Section-to-section nonlinear refinement of an MRI-aligned histology stack,
in the same spirit and code style as `morphint.py`. This is a Python/ANTs
re-implementation of part 2 of C.L.'s MINC pipeline
(`pipeline_finland_moTB_nissl.pl`, lines ~1155-1378), restructured to
operate directly on a 3-D numpy volume with sections along axis=1 (y).

What it does (one-paragraph version)
------------------------------------
For each interior section at y-index ``y_m`` whose neighbours sit at
``y_p`` (previous valid) and ``y_n`` (next valid), we register prev->mid
and next->mid using ANTs SyN. The two resulting displacement fields are
combined with a distance-weighted average and applied as a *small
correction* to mid's cumulative warp. The original (un-warped) section
is then re-resampled through the updated cumulative warp. This is
iterated. The cumulative warp is what guarantees that anatomic structure
of mid is preserved across iterations -- mid's intensities are always
read from the original section, never from the smeared neighbours.

Relationship to the warp-algebra version in the original Perl
-------------------------------------------------------------
The original Perl builds the prev<->next composite, half-steps the
displacement field, and composes back through the prev->mid and next->mid
warps. To first order in displacement magnitude (which is the regime this
pipeline is designed for -- the author was emphatic that "the large scales
must have been resolved" before this stage runs), that whole construction
collapses to a weighted average of the prev->mid and next->mid pull-back
warps. We do the linearized version here: it produces the same answer for
small corrections, has no displacement-field composition, and slots into
morphint's existing style cleanly. If you ever need the exact non-linear
form (large displacements per iteration, which is not recommended) the
hooks are noted in `_compute_midway_increment` and would require
`ants.apply_transforms(..., compose=...)`.

Distance weighting
------------------
The author noted that with sparse sampling the symmetric average becomes
``mid* = w * prev + (1 - w) * next`` where ``w`` reflects the relative
distance between mid and its two neighbours. We implement this as
``s = (y_m - y_p) / (y_n - y_p)``, the fraction of the prev->next path at
which mid sits. The pull-back-warp average is then
``(1 - s) * phi_m_to_p + s * phi_m_to_n``, which reduces to the symmetric
``0.5 * (phi_m_to_p + phi_m_to_n)`` when sampling is uniform and the
neighbours are equidistant.

Style match with morphint.py
----------------------------
- volume in / volume out, no per-section directory required from the caller
- ANTs run via the same antsRegistration subprocess invocation as
  `compute_ants_alignment` in morphint
- forward and inverse warps cached on disk per (y_p, y_m) pair and re-used
- `joblib.Parallel` parallelism across interior sections within an
  iteration (the iteration loop itself is sequential because each
  iteration depends on the previous)
- `clobber` flag plumbed through

Public entry points (mirroring morphint's structure)
----------------------------------------------------
- ``refine_volume(vol, ...)``   -- the workhorse, takes a numpy volume
- ``refine_nii(in_fin, out_fin, ...)``  -- file-in / file-out wrapper
- ``section_refine(in_fin, ...)``       -- top-level analogue of
  ``morphint.morphint``: full pipeline including iso-resampling
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import ants
import numpy as np
from joblib import Parallel, delayed
from morphint.inter_section_similarity import inter_section_similarity
from morphint.compute_ants_alignment import compute_ants_alignment


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _slice_to_ants(
    arr2d: np.ndarray,
    origin: tuple,
    spacing: tuple,
) -> ants.ANTsImage:
    """Wrap a 2-D numpy slice as an ANTsImage with the given xz origin/spacing.

    This matches the convention used in `morphint.process_section`, which
    treats `vol[:, y, :]` as the section and uses x and z geometry only.
    """
    return ants.from_numpy(arr2d.astype(np.float32), origin=origin, spacing=spacing)


def _write_section(arr2d: np.ndarray, origin, spacing, path: str) -> str:
    """Write a 2-D numpy slice to disk as a NIfTI at `path`."""
    if not os.path.exists(path):
        img = _slice_to_ants(arr2d, origin, spacing)
        img.to_filename(path)
    return path


def _resample_through_warp(
    section_path: str,
    reference_path: str,
    warp_path: str,
    output_path: str,
    interpolator: str = "linear",
    clobber: bool = False,
) -> str:
    """Apply a single displacement field warp to `section_path`, sampling on
    the grid defined by `reference_path`.

    This is the morphint idiom: write transforms to disk, call
    ``ants.apply_transforms`` with a list of paths.
    """
    if os.path.exists(output_path) and not clobber:
        return output_path
    fixed = ants.image_read(reference_path)
    moving = ants.image_read(section_path)
    out = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        interpolator=interpolator,
        transformlist=[warp_path],
        verbose=False,
    )
    ants.image_write(out, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Per-section refinement worker
# ---------------------------------------------------------------------------

@dataclass
class _SectionResult:
    """Bookkeeping for one interior section refined in one iteration."""
    y_m: int
    new_warp_path: Optional[str]   # path to the updated cumulative warp
    new_section_path: str          # path to the section resampled through it
    pair_warps: dict = field(default_factory=dict)
    # `pair_warps` caches the (fwd, inv) paths for the (prev,mid) and
    # (mid,next) pairs computed in this iteration, so the caller can stash
    # them in tfm_dict for re-use on subsequent iterations.


def _compute_midway_increment(
    phi_m_to_p_path: str,
    phi_m_to_n_path: str,
    s: float,
    relax: float,
    output_path: str,
    clobber: bool = False,
) -> str:
    """Compute the small-correction displacement field

        delta = relax * ( (1 - s) * phi_m_to_p + s * phi_m_to_n )

    and save it to `output_path` as a 2-D NIfTI vector image.

    `phi_m_to_p_path` and `phi_m_to_n_path` are the ANTs forward warps
    (`*_0Warp.nii.gz`) returned by registering mid->prev and mid->next
    respectively (mid as moving, neighbour as fixed). Applied to the
    original mid section, each warp produces something in the neighbour's
    coordinate frame -- exactly the direction of correction we want.

    The combination above is the linearized "midway" correction: it nudges
    mid TOWARD the average of where its neighbours sit, weighted by the
    relative distance to each neighbour. If mid is already aligned with
    both neighbours, both warps are ~zero and delta is ~zero.

    `s` is the fraction of the prev->next path at which mid sits:
        s = (y_m - y_p) / (y_n - y_p)
    With uniform sampling and immediate neighbours, s = 0.5. With sparse
    or non-uniform sampling the asymmetric weight implements the
    `mid* = w*prev + (1-w)*next` generalization.

    `relax` corresponds to the same parameter in the original Perl
    (default 0.5), damping each iteration.
    """
    if os.path.exists(output_path) and not clobber:
        return output_path

    phi_m_to_p = ants.image_read(phi_m_to_p_path)
    phi_m_to_n = ants.image_read(phi_m_to_n_path)

    # Sanity: both warps must live on the same grid (they do, because both
    # registrations used the same fixed image = the current resampled mid).
    if phi_m_to_p.shape != phi_m_to_n.shape:
        raise ValueError(
            f"prev->mid and next->mid warps have incompatible shapes "
            f"{phi_m_to_p.shape} vs {phi_m_to_n.shape}; "
            f"both registrations must use the same fixed image."
        )

    delta = (phi_m_to_p * float(relax * (1.0 - s))
             + phi_m_to_n * float(relax * s))

    ants.image_write(delta, output_path)
    return output_path


def _compute_midway_refinement(
        out,
        y_m: int,
        y_p: int,
        y_n: int,
        mid_resampled_path: str,
        prev_resampled_path: str,
        next_resampled_path: str,
        delta_path: str,
        pair_warp_dir: str,
        relax: float,
        resolution_list: list,
        resolution: float,
        cached_pair_warps: dict,
        base_itr: int = 30,
        clobber: bool = False,
):
    # Two registrations + weighted-average increment.
    # mid -> prev: mid is moving, prev is fixed. The forward warp
    # `fwd_pm`, applied to mid, brings mid onto prev's grid. Used
    # directly in the increment formula: applied to the original mid
    # section it produces something in prev's coordinate frame, which
    # is the desired direction of correction (toward neighbours). The
    # opposite direction (prev->mid) would give a wrong-sign warp.
    pair_pm_dir = os.path.join(pair_warp_dir, f"pm_{y_p}_{y_m}")
    pm_key = (y_m, y_p)
    fwd_pm, inv_pm = cached_pair_warps.get(pm_key, (None, None))
    fwd_pm, inv_pm = compute_ants_alignment(
        prefixdir=pair_pm_dir,
        sec0_path=mid_resampled_path,    # moving
        sec1_path=prev_resampled_path,   # fixed
        ymin=y_p, ymax=y_m,
        fwd_tfm_path=fwd_pm, inv_tfm_path=inv_pm,
        resolution_list=resolution_list, resolution=resolution,
        base_itr=base_itr,
        clobber=clobber,
    )
    out.pair_warps[pm_key] = (fwd_pm, inv_pm)

    pair_mn_dir = os.path.join(pair_warp_dir, f"mn_{y_m}_{y_n}")
    mn_key = (y_m, y_n)
    fwd_mn, inv_mn = cached_pair_warps.get(mn_key, (None, None))
    fwd_mn, inv_mn = compute_ants_alignment(
        prefixdir=pair_mn_dir,
        sec0_path=mid_resampled_path,
        sec1_path=next_resampled_path,
        ymin=y_m, ymax=y_n,
        fwd_tfm_path=fwd_mn, inv_tfm_path=inv_mn,
        base_itr=base_itr,
        resolution_list=resolution_list, resolution=resolution,
        clobber=clobber,
    )
    out.pair_warps[mn_key] = (fwd_mn, inv_mn)

    s = float(y_m - y_p) / float(y_n - y_p)  # uniform spacing -> 0.5
    _compute_midway_increment(
        phi_m_to_p_path=fwd_pm, phi_m_to_n_path=fwd_mn,
        s=s, relax=relax,
        output_path=delta_path, clobber=clobber,
    )

    return out


def _compute_forward_backward_refinement(
        out,
        y_m: int,
        y_p: int,
        y_n: int,
        mid_resampled_path: str,
        prev_resampled_path: str,
        next_resampled_path: str,
        delta_path: str,
        pair_warp_dir: str,
        relax: float,
        resolution_list: list,
        resolution: float,
        cached_pair_warps: dict,
        iteration_index: int = 1,
        clobber: bool = False,
    ):
    # Single-neighbour registration; sweep direction alternates by
    # iteration. Even (or first) iterations are "forward sweeps":
    # every section registers against its prev neighbour. Odd
    # iterations are "backward sweeps": every section registers
    # against its next neighbour. Across paired iterations the
    # directional bias of single-neighbour propagation cancels.
    is_forward = (iteration_index % 2 == 1)
    if is_forward:
        y_neighbour = y_p
        neighbour_path = prev_resampled_path
        tag = "fwd"
    else:
        y_neighbour = y_n
        neighbour_path = next_resampled_path
        tag = "bwd"

    pair_dir = os.path.join(
        pair_warp_dir, f"{tag}_{y_m}_{y_neighbour}"
    )
    pair_key = (y_m, y_neighbour)
    fwd, inv = cached_pair_warps.get(pair_key, (None, None))
    fwd, inv = compute_ants_alignment(
        prefixdir=pair_dir,
        sec0_path=mid_resampled_path,    # moving
        sec1_path=neighbour_path,        # fixed
        ymin=min(y_m, y_neighbour),
        ymax=max(y_m, y_neighbour),
        fwd_tfm_path=fwd, inv_tfm_path=inv,
        resolution_list=resolution_list, resolution=resolution,
        clobber=clobber,
    )
    out.pair_warps[pair_key] = (fwd, inv)

    # Increment = relax * fwd_warp. No (1-s)/(s) weighting because
    # there's only one neighbour per call.
    if not os.path.exists(delta_path) or clobber:
        warp_img = ants.image_read(fwd)
        delta = warp_img * float(relax)
        ants.image_write(delta, delta_path)
    
    return out

# Note for future maintainers: the *non*-linearized version of the above
# (matching the original Perl's prev<->next half-step construction) would
# build phi_p_to_n = phi_m_to_n o phi_p_to_m and phi_n_to_p as composite
# fields, half-step each (with `s` and `1-s` respectively for the
# asymmetric case), and compose back through phi_m_to_p / phi_m_to_n.
# The result agrees with the formula above to first order in displacement
# magnitude, which is the regime this pipeline is designed for. If you
# ever need the exact form, ANTsPy's `apply_transforms(..., compose=...)`
# can build the composite fields, then `scale_displacement_field` halves
# them, then a second `apply_transforms` composes back. Implementing
# this is left as a flag (not currently exposed) because in our regime
# the linearized version is faster and as accurate.


def _refine_one_section(
    y_m: int,
    y_p: int,
    y_n: int,
    section_orig: np.ndarray,
    mid_resampled_path: str,
    prev_resampled_path: str,
    next_resampled_path: str,
    cumulative_warp_path: Optional[str],
    iter_dir: str,
    pair_warp_dir: str,
    origin: tuple,
    spacing: tuple,
    relax: float,
    resolution_list: list,
    resolution: float,
    cached_pair_warps: dict,
    clobber: bool = False,
    refinement_type: str = "midway",
    iteration_index: int = 1,
    base_itr: int = 30,
) -> _SectionResult:
    """Refine a single interior section.

    Inputs:
        y_m, y_p, y_n
            y-indices of the mid section and its valid neighbours.
        section_orig
            the *original*, un-warped 2-D section at y_m. Re-resampling
            from the original is what preserves anatomy across iterations.
        mid_resampled_path, prev_resampled_path, next_resampled_path
            paths to the current-iteration resampled images for mid, prev,
            next. These are what we register against. Some strategies
            (forward / backward) only use one neighbour and ignore the
            other; this is the deliberate choice to keep the function
            signature stable across strategies.
        cumulative_warp_path
            path to the cumulative warp for mid as of the start of this
            iteration. None on the first iteration of the first call,
            in which case we initialize it to a zero field.
        iter_dir
            directory into which this iteration's outputs are written.
        pair_warp_dir
            directory caching pair warps; survives across iterations so
            that pair_warps from earlier iterations are reused only via
            tfm_dict (we do *not* reuse stale pair warps across iterations
            because the resampled images change every iteration).
        cached_pair_warps
            dict {(y_a, y_b): (fwd, inv)} of pre-computed pair warps from
            previous calls. Same key convention as morphint.tfm_dict.
        refinement_type
            Which per-section algorithm to use:
              "midway"
                The colleague's algorithm: register mid->prev and
                mid->next, take a distance-weighted average of the two
                forward warps, scaled by relax, as the increment. Best
                when section spacing is dense relative to the scale of
                anatomical variation between sections.
              "forward_backward"
                Adjacent-pair only. Even iterations register mid->prev
                ("forward sweep"); odd iterations register mid->next
                ("backward sweep"). The increment is the single forward
                warp scaled by relax. Better when adjacent sections
                contain noticeably different anatomy (e.g. wide gaps
                through cortex) -- the prev/next averaging that "midway"
                does is then less meaningful, while one-neighbour
                registration only asks "does mid look like its
                neighbour?" which remains well-posed at the smoothing
                scales ANTs uses internally in its multi-resolution
                schedule.
        iteration_index
            1-based iteration counter from the driver. Used by
            "forward_backward" to alternate sweep direction. Ignored by
            "midway".

    Output:
        _SectionResult bundling the new cumulative warp path, the new
        resampled section path, and any pair warps we computed (for
        upstream caching).
    """
    out = _SectionResult(y_m=y_m, new_warp_path=None,
                         new_section_path="", pair_warps={})

    # ---- 1-3) compute the increment, dispatch by strategy.
    delta_path = os.path.join(iter_dir, f"delta_{y_m}.nii.gz")

    if refinement_type == "midway":
        out = _compute_midway_refinement(
                out,
                y_m,
                y_p,
                y_n,
                mid_resampled_path,
                prev_resampled_path,
                next_resampled_path,
                delta_path,
                pair_warp_dir,
                relax,
                resolution_list,
                resolution,
                cached_pair_warps,
                base_itr=base_itr,
                clobber=clobber,
        )
    elif refinement_type == "forward_backward":
        out = _compute_forward_backward_refinement(
                out,
                y_m,
                y_p,
                y_n,
                mid_resampled_path,
                prev_resampled_path,
                next_resampled_path,
                delta_path,
                pair_warp_dir,
                relax,
                resolution_list,
                resolution,
                cached_pair_warps,
                iteration_index=iteration_index,
                clobber=clobber,
        )
    else:
        raise ValueError(
            f"unknown refinement_type {refinement_type!r}; "
            f"expected 'midway' or 'forward_backward'"
        )

    # ---- 4) update cumulative warp (strategy-independent).
    # The cumulative warp lives in the same coordinate frame as the section
    # template. On the first iteration, when no cumulative warp exists yet,
    # delta IS the cumulative warp. On subsequent iterations we need to
    # combine W_old with delta. To first order in displacement, addition of
    # the two displacement fields is equivalent to composition; for the
    # small per-iteration corrections this stage produces, it's accurate
    # enough and avoids a second ANTs round-trip.
    new_warp_path = os.path.join(iter_dir, f"cumulative_{y_m}.nii.gz")

    if (cumulative_warp_path is not None
            and os.path.exists(cumulative_warp_path)):
        W_old = ants.image_read(cumulative_warp_path)
        delta = ants.image_read(delta_path)
        if W_old.shape != delta.shape:
            # Shapes will differ if the template grid changed between iters.
            # Resample W_old onto delta's grid before adding.
            W_old = ants.resample_image_to_target(
                W_old, delta, interp_type="linear"
            )
        W_new = W_old + delta
        ants.image_write(W_new, new_warp_path)
    else:
        # First-iteration shortcut: cumulative := delta. Copy rather than
        # rename so re-runs with `clobber=True` remain idempotent.
        if not os.path.exists(new_warp_path) or clobber:
            ants.image_write(ants.image_read(delta_path), new_warp_path)

    out.new_warp_path = new_warp_path

    # ---- 5) re-resample the *original* section through the new cumulative warp
    # (also strategy-independent).
    orig_path = os.path.join(iter_dir, f"orig_{y_m}.nii.gz")
    _write_section(section_orig, origin, spacing, orig_path)

    new_section_path = os.path.join(iter_dir, f"section_{y_m}.nii.gz")
    if not os.path.exists(new_section_path) or clobber:
        ref = ants.image_read(mid_resampled_path)
        moving = ants.image_read(orig_path)
        warped = ants.apply_transforms(
            fixed=ref,
            moving=moving,
            interpolator="linear",
            transformlist=[new_warp_path],
            verbose=False,
        )
        ants.image_write(warped, new_section_path)

    out.new_section_path = new_section_path
    return out


# ---------------------------------------------------------------------------
# Iteration driver
# ---------------------------------------------------------------------------

def _initialize_iter0(
    vol: np.ndarray,
    valid_idx: np.ndarray,
    iter_dir: str,
    origin: tuple,
    spacing: tuple,
) -> dict:
    """Write each valid input section to disk as the iteration-0 'resampled'
    image. Returns a dict {y: path_to_section_i0}.

    On iteration 0 the resampled image *is* the input section, because no
    refinement has been applied yet. The cumulative warp is implicitly the
    identity / a zero displacement field, so we don't need to materialize it.
    """
    os.makedirs(iter_dir, exist_ok=True)
    section_paths = {}
    for y in valid_idx:
        path = os.path.join(iter_dir, f"section_{int(y)}.nii.gz")
        if not os.path.exists(path):
            arr = vol[:, int(y), :]
            _slice_to_ants(arr, origin, spacing).to_filename(path)
        section_paths[int(y)] = path
    return section_paths


def refine_volume(
    vol: np.ndarray,
    output_dir: str,
    origin: tuple,
    spacing: tuple,
    relax: float = 0.5,
    resolution_list: list = (4, 2, 1, 0.5),
    num_jobs: int = -1,
    tfm_dict: Optional[dict] = None,
    base_itr: int = 30,
    iterations_per_resolution: int = 1,
    clobber: bool = False,
    refinement_type: str = "midway",
) -> tuple:
    """Refine the inter-section alignment of an MRI-aligned volume.

    Parameters
    ----------
    vol
        3-D numpy array, sections along axis=1. Sections that are entirely
        zero are treated as missing and skipped.
    output_dir
        Working directory for transforms and intermediate sections.
    origin, spacing
        2-D origin/spacing for the in-plane geometry of each section
        (matching morphint's `affine[[0, 2], 3]` and `affine[[0, 2], [0, 2]]`
        conventions).
    n_iterations
        Number of refinement sweeps. Each iteration updates every interior
        section in parallel using the previous iteration's snapshot, then
        swaps the new sections in. For `refinement_type="forward_backward"`
        you typically want this to be even so forward and backward sweeps
        balance.
    relax
        Per-iteration damping (default 0.5).
    resolution_list, resolution
        Multi-resolution schedule passed through to morphint's
        `compute_ants_alignment`. ANTs runs its own internal coarse-to-fine
        descent within a single registration call using
        `--shrink-factors` and `--smoothing-sigmas` -- we don't pre-smooth
        the inputs ourselves.
    num_jobs
        joblib worker count (-1 means all cores).
    tfm_dict
        Optional cache of pair warps from a previous run. Keys are
        (y_a, y_b) tuples; values are (fwd_path, inv_path).
    clobber
        Overwrite existing intermediate files if True.
    refinement_type
        Strategy passed through to `_refine_one_section` -- see its
        docstring. "midway" (default) is the colleague's algorithm;
        "forward_backward" alternates single-neighbour sweeps each
        iteration and is generally better for sparse data.
    base_itr
        Base number of iterations for ANTs registration.

    Returns
    -------
    (refined_vol, tfm_dict)
        The refined volume (same shape as `vol`) and the updated
        pair-warp cache.
    """
    out_vol = vol.copy()

    valid_idx = np.where(np.max(vol, axis=(0, 2)) > 0)[0]
    if len(valid_idx) == 0:
        return out_vol, (tfm_dict or {})

    # Iteration 0: each valid section becomes its own "resampled" image.
    iter0_dir = os.path.join(output_dir, "iter_0")
    section_paths = _initialize_iter0(vol, valid_idx, iter0_dir, origin, spacing)

    # Pair warps live in their own dir, separate from per-iter scratch.
    # We do NOT reuse stale pair warps across iterations: the resampled
    # mid image changes every iteration. The user-provided `tfm_dict` is
    # used only to seed the *first* iteration (e.g. for resuming a run).
    pair_warp_dir = os.path.join(output_dir, "pair_warps")
    os.makedirs(pair_warp_dir, exist_ok=True)

    # Cumulative warps per section; None at iteration 0.
    cumulative_warps = {int(y): None for y in valid_idx}

    # Normalize the user-provided seed cache.
    if tfm_dict is None:
        seed_cache = {}
    else:
        seed_cache = {
            tuple(int(x) for x in (k if isinstance(k, tuple) else
                                   tuple(int(z) for z in k.split(","))))
            : v
            for k, v in tfm_dict.items()
        }
    latest_pair_warps: dict = {}

    # Identify interior sections. The endpoints are not refined (they
    # have no two-sided neighbour pair).
    interior_y = [int(y) for y in valid_idx[1:-1]]
    valid_set = set(int(y) for y in valid_idx)

    # Build (y_p, y_m, y_n) triples for each interior section. y_p / y_n
    # are the nearest valid neighbours on each side. With uniform sampling
    # they are immediate neighbours; with sparse sampling they may be
    # farther away.
    triples = []
    sorted_valid = sorted(valid_set)
    for y_m in interior_y:
        idx = sorted_valid.index(y_m)
        y_p = sorted_valid[idx - 1]
        y_n = sorted_valid[idx + 1]
        triples.append((y_p, y_m, y_n))

    if refinement_type == "forward_backward" or iterations_per_resolution > 1 : 
        itr_resolution_list = np.repeat(resolution_list, iterations_per_resolution)
    else :
        itr_resolution_list = resolution_list 
    
    # ----------- iteration loop -----------
    for it, curr_resolution in enumerate(itr_resolution_list):
        iter_dir = os.path.join(output_dir, f"iter_{it}_{curr_resolution}mm")
        os.makedirs(iter_dir, exist_ok=True)

        # Snapshot of section paths going INTO this iteration. The new
        # paths produced this iteration are only promoted to
        # `section_paths` once the whole sweep completes (mirrors the
        # Perl's `new/` staging dir).
        snapshot_paths = dict(section_paths)
        snapshot_cumulative = dict(cumulative_warps)

        # Only the first iteration consults the user-supplied seed cache;
        # subsequent iterations need fresh registrations.
        cache_for_iter = seed_cache if it == 0 else {}

        num_jobs = 1
        results = Parallel(n_jobs=num_jobs)(
            delayed(_refine_one_section)(
                y_m=y_m,
                y_p=y_p,
                y_n=y_n,
                section_orig=vol[:, y_m, :],
                mid_resampled_path=snapshot_paths[y_m],
                prev_resampled_path=snapshot_paths[y_p],
                next_resampled_path=snapshot_paths[y_n],
                cumulative_warp_path=snapshot_cumulative[y_m],
                iter_dir=iter_dir,
                pair_warp_dir=os.path.join(pair_warp_dir, f"iter_{it}_{curr_resolution}mm"),
                origin=origin,
                spacing=spacing,
                relax=relax,
                resolution_list=list(resolution_list),
                resolution=curr_resolution,
                cached_pair_warps=cache_for_iter,
                clobber=clobber,
                base_itr=base_itr,
                refinement_type=refinement_type,
                iteration_index=it,
            )
            for (y_p, y_m, y_n) in triples
        )

        # Promote results into the live state.
        latest_pair_warps = {}
        for r in results:
            section_paths[r.y_m] = r.new_section_path
            cumulative_warps[r.y_m] = r.new_warp_path
            latest_pair_warps.update(r.pair_warps)

    # ----------- pack refined sections back into the volume -----------
    for y_m in interior_y:
        refined = ants.image_read(section_paths[y_m]).numpy()
        out_vol[:, y_m, :] = refined
    # Endpoints retain their original (non-refined) values.

    tfm_dict_out = {f"{a},{b}": v for (a, b), v in latest_pair_warps.items()}
    return out_vol, tfm_dict_out


# ---------------------------------------------------------------------------
# File-in / file-out wrappers, mirroring morphint's nii / top-level functions
# ---------------------------------------------------------------------------

def refine_nii(
    in_fin: str,
    out_fin: str,
    output_dir: str,
    relax: float = 0.5,
    resolution_list: list = (4, 2, 1, 0.5),
    base_itr: int = 30,
    iterations_per_resolution: int = 1,
    num_jobs: int = -1,
    tfm_dict: Optional[dict] = None,
    clobber: bool = False,
    refinement_type: str = "midway",
) -> dict:
    """Refine an MRI-aligned NIfTI volume in place. Mirrors
    `morphint.nl_deformation_flow_nii` in interface and side effects.
    """
    tfm_json = out_fin.replace(".nii.gz", "") + "_refine_tfm.json"

    if (not os.path.exists(out_fin)
            or not os.path.exists(tfm_json)
            or clobber):
        in_img = ants.image_read(in_fin)

        in_vol = in_img.numpy()
        in_vol[in_vol < 0] = 0

        origin = tuple(in_img.origin[[0, 2]])
        spacing = tuple(in_img.spacing[[0, 2]])

        refined_vol, tfm_dict_out = refine_volume(
            in_vol,
            output_dir + "/section_refine/",
            origin=origin,
            spacing=spacing,
            relax=relax,
            resolution_list=resolution_list,
            base_itr=base_itr,
            iterations_per_resolution=iterations_per_resolution,
            num_jobs=num_jobs,
            tfm_dict=tfm_dict,
            clobber=clobber,
            refinement_type=refinement_type,
        )


        ants.image_write(
            ants.from_numpy(refined_vol, origin=in_img.origin, spacing=in_img.spacing, direction=in_img.direction),
            out_fin
        )

        with open(tfm_json, "w") as f:
            json.dump(tfm_dict_out, f)
    else:
        with open(tfm_json, "r") as f:
            tfm_dict_out = json.load(f)
    
    correlations_orig = inter_section_similarity(in_fin)
    correlations_refined = inter_section_similarity(out_fin)
    mean_sim_orig = np.mean(correlations_orig)
    mean_sim_refined = np.mean(correlations_refined)

    print(f"Improvement: {mean_sim_refined - mean_sim_orig:.3f} (mean)")

    return tfm_dict_out, {'correlations_orig': correlations_orig, 'correlations_refined': correlations_refined}


def section_refine(
    in_fin: str,
    out_fin:str,
    output_dir: str,
    n_resolutions: int = 3,
    iterations_per_resolution: int = 1,
    relax: float = 0.5,
    resolution_list: list = None,
    resolution: float = 0.5,
    num_jobs: int = -1,
    tfm_dict: Optional[dict] = None,
    base_itr: int = 30,
    clobber: bool = False,
    refinement_type: str = "midway",
) -> tuple:
    """Top-level entry point analogous to ``morphint.morphint``.

    Returns
    -------
    (out_fin, tfm_dict)
        Path to the refined NIfTI and the pair-warp dict.
    """
    if resolution_list is None:
        resolution_list = [resolution * i for i in range(1, n_resolutions + 1)]

    tfm_dict_out, corr_dict_out = refine_nii(
        in_fin=in_fin,
        out_fin=out_fin,
        output_dir=output_dir,
        relax=relax,
        base_itr=base_itr,
        iterations_per_resolution=iterations_per_resolution,
        resolution_list=resolution_list,
        num_jobs=num_jobs,
        tfm_dict=tfm_dict,
        clobber=clobber,
        refinement_type=refinement_type,
    )
    return tfm_dict_out, corr_dict_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Section-to-section refinement of an MRI-aligned "
                    "histology volume (sections along axis=1)."
    )
    parser.add_argument("in_fin", help="Input NIfTI volume.")
    parser.add_argument("out_fin", help="Output NIfTI path.")
    parser.add_argument("output_dir", help="Working directory for transforms.")
    parser.add_argument(
        "--refinement-type", default="midway",
        choices=("midway", "forward_backward"),
        help="Per-section algorithm. 'midway' (default) is the "
             "weighted-average algorithm; 'forward_backward' alternates "
             "single-neighbour sweeps.",
    )
    parser.add_argument("--n-resolutions", type=int, default=3)
    parser.add_argument("--iterations-per-resolution", type=int, default=1)
    parser.add_argument("--base-itr", type=int, default=30,
                        help="Base number of iterations for ANTs registration.")
    parser.add_argument("--relax", type=float, default=0.5)
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--resolution-list", type=float, nargs="+",
                        default=None)
    parser.add_argument("--num-jobs", type=int, default=-1)
    parser.add_argument("--clobber", action="store_true")
    args = parser.parse_args()

    refine_nii(
        in_fin=args.in_fin,
        out_fin=args.out_fin,
        output_dir=args.output_dir,
        n_resolutions=args.n_resolutions,
        iterations_per_resolution=args.iterations_per_resolution,
        relax=args.relax,
        resolution=args.resolution,
        base_itr=args.base_itr,
        num_jobs=args.num_jobs,
        clobber=args.clobber,
        refinement_type=args.refinement_type,
    )