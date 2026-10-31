"""Read in a nii.gz volume and calculate the pearson correlation between each pair of adjacent coronal sections across axis 1.
"""

import argparse
import numpy as np
import ants
from scipy.stats import kendalltau, pearsonr, spearmanr

from brainbuilder.utils.axis_utils import (
    DEFAULT_SECTION_AXIS,
    get_section,
    section_profile,
)
from brainbuilder.utils.utils import get_logger

logger = get_logger("morphint")

def inter_section_similarity(
    in_path, similarity_metric="kendalltau", axis: int = DEFAULT_SECTION_AXIS
) -> tuple[float, float]:
    """Calculate the Similarity correlation between each pair of adjacent sections across the sectioning axis.
        First, read in the .nii.gz using ants and then identify the valid sections (those with non-zero pixels). 
        Then, for each pair of adjacent valid sections, calculate the similarity and 
        print the mean and standard deviation list of these correlations.
    """

    # Read in the .nii.gz using ants
    img = ants.image_read(in_path)
    data = img.numpy()

    # Identify valid sections (those with non-zero pixels)
    valid_idx = np.where(section_profile(data, axis, reduction=np.sum) > 0)[0]

    # Calculate similarity between adjacent valid sections
    correlations = []
    for i in range(len(valid_idx) - 1):
        section1 = get_section(data, valid_idx[i], axis)
        section2 = get_section(data, valid_idx[i + 1], axis)
        if similarity_metric == "kendalltau":
            corr = kendalltau(section1.flatten(), section2.flatten())[0]
        elif similarity_metric == "pearson":
            corr = pearsonr(section1.flatten(), section2.flatten())[0]
        elif similarity_metric == "spearman":
            corr = spearmanr(section1.flatten(), section2.flatten())[0]
        correlations.append(corr)

    # Print mean and standard deviation of correlations
    mean_corr = np.mean(correlations)
    std_corr = np.std(correlations)
    logger.info(f"Similarity ({similarity_metric}): {mean_corr:.3f} +/- {std_corr:.3f}")



    return correlations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate inter-section similarity for a .nii.gz volume.")
    parser.add_argument("in_path", type=str, help="Path to the input .nii.gz file.")
    parser.add_argument("--metric", type=str, default="kendalltau", choices=["kendalltau", "pearson", "spearman"], help="Similarity metric to use.")
    args = parser.parse_args()
    inter_section_similarity(args.in_path, similarity_metric=args.metric)
