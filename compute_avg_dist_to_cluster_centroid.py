import numpy as np
import os
from tqdm import tqdm
import argparse
from constants import CLUSTER_SCHEMA

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--num_clusters",
    type=int,
    default=30000,
    help="",
)
parser.add_argument(
    "--sorted_clusters_path",
    type=str,
    default="",
    help="",
)
parser.add_argument(
    "--avg_distance_to_cent_save_path",
    type=str,
    default="",
    help="",
)


def compute_avg_dist_to_cluster_centroid(args):
    """
    Compute the average distance to cluster centroids and save the results.

    Args:
        args (Namespace): A namespace object containing the following attributes:
            - num_clusters (int): Number of clusters.
            - sorted_clusters_path (str): Directory path where individual cluster files are stored.
            - avg_distance_to_cent_save_path (str): File path to save the average distances to centroids.

    Returns:
        None: This function does not return any value. It saves the computed average distances
        to centroids to the specified file path.

    Notes:
        This function assumes the cluster files named as 'cluster_{i}.npy' where 'i' ranges from 0 to num_clusters-1.
        Each cluster file is expected to contain data with a schema defined by CLUSTER_SCHEMA.

        The computation involves loading each cluster file, extracting the distance to centroid information
        based on the specified schema, calculating the average distance to centroid for each cluster,
        and finally saving the average distances to centroids in a list format to a file.

        If the directory specified by avg_distance_to_cent_save_path does not exist, it will be created.

    Example:
        # Define arguments
        args = Namespace(
            num_clusters=100,
            sorted_clusters_path="/path/to/sorted_clusters",
            avg_distance_to_cent_save_path="/path/to/avg_distances_to_centroids.npy"
        )

        # Call the function
        compute_avg_dist_to_cluster_centroid(args)
    """
    num_clusters = args.num_clusters
    sorted_clusters_path = args.sorted_clusters_path
    avg_distance_to_cent_save_path = args.avg_distance_to_cent_save_path
    
    # Ensure the number of cluster files matches num_clusters
    assert len(os.listdir(sorted_clusters_path)) == num_clusters

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(avg_distance_to_cent_save_path), exist_ok=True)

    avg_distance_to_cent_list = []
    
    # Iterate over each cluster
    for i in tqdm(range(num_clusters)):
        # Load cluster data
        cluster_i = np.load(f"{sorted_clusters_path}/cluster_{i}.npy")
        
        # Extract distance to centroid information based on schema
        avg_distance_to_cent_list.append((cluster_i[:, CLUSTER_SCHEMA['distance_to_centroid']['id']].astype("float32")).mean())

    # Save average distances to centroids to file
    np.save(avg_distance_to_cent_save_path, avg_distance_to_cent_list)
    print("Done")




if __name__ == "__main__":
    args = parser.parse_args()
    compute_avg_dist_to_cluster_centroid(args)
