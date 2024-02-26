import faiss
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--num_NNs",
    type=int,
    default=20,
    help="",
)
parser.add_argument(
    "--dim",
    type=int,
    default=512,
    help="",
)
parser.add_argument(
    "--centroids_file",
    type=int,
    default="/path/to/centroids.npy",
    help="",
)
parser.add_argument(
    "--mean_centroid_distances_save_path",
    type=str,
    default="/path/to/mean_distances.npy",
    help="",
)


def compute_centroid_distances(args):
    """
    Compute the mean distances between a centroid and its <num_NNs> nearest centroids using Faiss indexing.

    Args:
        args (Namespace): A namespace object containing the following attributes:
            - num_NNs (int): Number of nearest neighbors to consider.
            - centroids_file (str): File path to the kmeans .npy centroids array.
            - mean_centroid_distances_save_path (str): .npy file path to save the mean distances between centroids.
            - dim (int): Dimensionality of the centroids.

    Returns:
        None: This function does not return any value. It saves the computed mean distances
        between centroids to the specified file path.

    Raises:
        AssertionError: If the length of the computed mean distances does not match the number of centroids.

    Notes:
        The computation involves loading the centroids from a file, building an IndexFlatIP Faiss index,
        adding the centroids to the index, performing a search to find nearest neighbors, calculating the
        mean distances between centroids while excluding self-distances, and finally saving the mean distances
        to a file.

        The mean distances between centroids are computed using the formula: mean_dist = np.mean(1 - Dist[:, 1:], axis=1),
        where Dist is the distance matrix obtained from the Faiss index search.

        If the directory specified by mean_centroid_distances_save_path does not exist, it will be created.

    Example:
        # Define arguments
        args = Namespace(
            num_NNs=10,
            centroids_file="/path/to/centroids.npy",
            mean_centroid_distances_save_path="/path/to/mean_distances.npy",
            dim=128
        )

        # Call the function
        compute_centroid_distances(args)
    """
    num_NNs = args.num_NNs
    centroids_file = args.centroids_file
    mean_centroid_distances_save_path = args.mean_centroid_distances_save_path
    dim = args.dim

    # Load centroids
    centroids = np.load(centroids_file)

    # Build Faiss index
    index = faiss.IndexFlatIP(dim)

    # Add centroids to the index
    index.add(centroids)

    # Perform search to find nearest neighbors
    Dist, _ = index.search(centroids, num_NNs + 1)

    # Compute mean distances between centroids (excluding self-distances)
    mean_dist = np.mean(1 - Dist[:, 1:], axis=1)

    # Ensure length of mean_dist matches number of centroids
    assert len(mean_dist) == len(centroids)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(mean_centroid_distances_save_path), exist_ok=True)

    # Save mean distances to file
    np.save(mean_centroid_distances_save_path, mean_dist)
    print("Done")


if __name__ == "__main__":
    args = parser.parse_args()
    compute_centroid_distances(args)