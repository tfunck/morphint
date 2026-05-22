import os
import uuid

import ants
import nibabel as nb


def _is_nonempty_file(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def _is_readable_nifti(path: str) -> bool:
    """Fast sanity check: exists, non-empty, and nibabel can read the header."""
    if not _is_nonempty_file(path):
        return False
    try:
        nb.load(path)
        return True
    except Exception:
        return False


def _atomic_ants_image_write(image: "ants.core.ants_image.ANTsImage", out_path: str) -> None:
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.basename(out_path)
    if out_path.endswith(".nii.gz"):
        suffix = ".nii.gz"
    else:
        suffix = os.path.splitext(out_path)[1] or ".nii.gz"

    tmp_path = os.path.join(
        out_dir,
        f".{base}.tmp_{os.getpid()}_{uuid.uuid4().hex}{suffix}",
    )

    try:
        ants.image_write(image, tmp_path)
        os.replace(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise