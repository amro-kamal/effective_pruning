import numpy as np
import pickle
import random
import os
import argparse
import torch
from qpsolvers import solve_qp
from tqdm import tqdm
import pygtrie
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from utils import get_logger
import logging
from cls_balance_pruning import ssp_pruning, random_pruning, load_arrays
from utils import seed_everything

METRIC_ID = 2
# python prune.py --prune-ratio 0.8 --sorted-clusters-path "/fsx-labs/amroabbas/pruned/sorted_clusters/" --savedir "/fsx-labs/amroabbas/openclip-for-density-based-pruning-short-training/new_methods_pruning_files/random_pruning_0.8/"


def pruning_qp(
    probabilites,
    number_of_items_in_cluster,
    pruned_dataset_size,
    num_centroids,
    print_fn,
    cls_bal_ratio,
):
    assert (
        cls_bal_ratio >= 0.0 and cls_bal_ratio <= 1.0
    ), "cls_bal_ratio should be >0.0 and <=1.0"
    if cls_bal_ratio > 0.0:
        min_samples = int((pruned_dataset_size / num_centroids) * cls_bal_ratio)
        print_fn(
            f"class/cluster balanced pruning, pruned_data_size: {pruned_dataset_size}, num_centroids: {num_centroids}, collecting min_samples ({min_samples}) from each class/cluster to acheive cls_balance_ratio of {cls_bal_ratio}"
        )
    else:
        min_samples = 1

    P = np.eye(num_centroids)
    q = -probabilites * pruned_dataset_size
    A = np.array([1.0] * num_centroids)
    b = np.array([pruned_dataset_size])
    bounds = np.array(
        [
            (
                min(min_samples, number_of_items_in_cluster[i] - 1),
                number_of_items_in_cluster[i],
            )
            for i in range(num_centroids)
        ]
    )

    x = solve_qp(P=P, q=q, A=A, b=b, lb=bounds[:, 0], ub=bounds[:, 1], solver="osqp")
    x = np.rint(x).astype(int)
    # assert sum((x < 0)) == 0

    return x


def get_paths(args, number_of_points_to_keep):

    all_paths_pruned = list()
    all_distances = list()
    pruned_distances = list()

    for i, cluster_i in enumerate(tqdm(args.sorted_clusters)):
        if i == 0:
            args.print_fn(
                f"{i} get_paths,  keep: {args.which_to_keep}"
            )
        cluster_i = np.load(os.path.join(args.sorted_clusters_path, f"cluster_{i}.npy"))

        cluster_size = len(cluster_i)
        cluster_all_paths = cluster_i[:, 0].astype("<U32").tolist()
        distances = cluster_i[:, 2].astype(float)
        all_distances.append(np.mean(distances))

        number_of_points_to_keep_from_cluster_i = number_of_points_to_keep[i]
        if args.which_to_keep == "random":

            random_ids = random.sample(
                range(len(cluster_all_paths)),
                min(cluster_size, int(number_of_points_to_keep_from_cluster_i)),
            )
            ids_to_remove = list(set(random_ids) - set(range(cluster_size)))
            paths = np.array(cluster_all_paths)[random_ids]
            removed_paths = np.array(cluster_all_paths)[ids_to_remove]
            removed_distances = distances[ids_to_remove]
            distances = distances[random_ids]

        elif args.which_to_keep == "hard":
            paths = np.array(cluster_all_paths)[
                : int(number_of_points_to_keep_from_cluster_i)
            ]
            removed_paths = np.array(cluster_all_paths)[
                int(number_of_points_to_keep_from_cluster_i) :
            ]
            removed_distances = distances[
                int(number_of_points_to_keep_from_cluster_i) :
            ]
            distances = distances[: int(number_of_points_to_keep_from_cluster_i)]

        elif args.which_to_keep == "easy":
            paths = np.array(cluster_all_paths)[
                -int(number_of_points_to_keep_from_cluster_i) :
            ]
            removed_paths = np.array(cluster_all_paths)[
                : -int(number_of_points_to_keep_from_cluster_i)
            ]
            removed_distances = distances[
                : -int(number_of_points_to_keep_from_cluster_i)
            ]
            distances = distances[-int(number_of_points_to_keep_from_cluster_i) :]

        all_paths_pruned.extend(paths)
        pruned_distances.append(np.mean(distances))

    return all_paths_pruned, all_distances, pruned_distances



def get_distances(args):
    args.print_fn(f"get_distances {args.density}")
    assert args.density in [
        "clustersize",
        "uniform",
        "dinter*dintra", 
        "dintra*dinter",
    ]
    args.print_fn(f"get_distances, density: {args.density}")

    # calculate complexity
    if args.density in ["clustersize"]:
        number_of_items_in_cluster = list()
        for cluster_i in enumerate(tqdm(args.sorted_clusters)):
            number_of_items_in_cluster.append(cluster_i.shape[0])

        d_i = number_of_items_in_cluster

    elif args.density == "uniform":
        d_i = np.ones(args.num_centroids)
        return d_i

    elif args.density in ["dinter*dintra", "dintra*dinter"]:
        ## load d_inter and d_intra first
        d_intra = np.load(args.avg_distance_to_centroid_file)
        args.print_fn("avg_distance_to_centroid_file loaded")

        d_inter = np.load(args.NNs_centroids_distances_dir)
        args.print_fn("centroids_distances loaded")

        assert d_inter.shape == d_intra.shape

        d_i = d_inter * d_intra

        # remove nans. nans can arise if there are only 1-2 items per cluster
        indices_nan = np.argwhere(np.isnan(d_i))
        assert sum((d_i < 0)) < 10

        for item in indices_nan:
            d_i[item] = [np.nanmean(d_i)]

    return d_i


def t_or_f(arg):
    ua = str(arg).upper()
    if "TRUE".startswith(ua):
        return True
    elif "FALSE".startswith(ua):
        return False
    else:
        pass  # error condition maybe?


def main(args=None):

    parser = argparse.ArgumentParser(description="Density Based Pruning")
    parser.add_argument(
        "--num-centroids",
        default=10000,
        type=int,
        help="number of cluster centroids in kmeans clustering",
    )
    parser.add_argument(
        "--temperature",
        default=0.5,
        type=float,
        help="temperature for softmax to turn the inter-cluster distances into a probability distribution",
    )
    parser.add_argument(
        "--distances-dir",
        default="/private/home/evgeniarusak/SemDeDup/SemDeDup/clustering/results_spp_recalc_5k/distances/",
        type=str,
        help="location of the intra- vs inter-class distances. Output when running calculate_distances.sh",
    )
    parser.add_argument(
        "--NNs-centroids-distances-dir",
        default="",
        type=str,
        help="location of NNs_centroids_distances_dir distances. Output when running calculate_distances.sh",
    )
    parser.add_argument(
        "--avg-distance-to-centroid-file",
        default="",
        type=str,
        help="location of avg_distance_to_centroid_file distances. Output when running compute_avg_distance_to_centroid.sh",
    )
    parser.add_argument(
        "--sorted-clusters-path",
        type=str,
        default="/private/home/evgeniarusak/SemDeDup/SemDeDup/clustering/results_spp_recalc_5k/0.5_cluster_bal/sorted_clusters/OpenCLIP_SSP_5000clusters_cosine_SphNormkmeansIndex_cls_bal_prn/",
        help="location of the sorted kmeans clusters",
    )
    parser.add_argument("--total-dataset-size", default=50749149, type=int)
    parser.add_argument(
        "--which-to-keep",
        default="False",
        choices=["hard", "easy", "random"],
        type=str,
        help="keep hard/easy/random examples from cluster",
    )
    parser.add_argument(
        "--prune-ratio", default=0.8, type=float, help="how much of the dataset to keep"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/checkpoint/evgeniarusak/clustering_50M_niters100_recalculated_emb_img_only/points_to_remove/",
    )
    parser.add_argument(
        "--density",
        type=str,
        default="uniform",
        help="density method",
        choices=[
            "clustersize",
            "uniform",
            "dintra*dinter",
            "dinter*dintra",
            "None",
        ],
    )
    parser.add_argument(
        "--pruning-method",
        default="qp",
        type=str,
        choices=["qp", "cls_bal_prn", "random", "None"],
        help="pruning method",
    )
    parser.add_argument(
        "--cls-bal-ratio",
        default=0.5,
        type=float,
        help="class/cluster balance ratio for ssp pruning",
    )
    parser.add_argument(
        "--save-output",
        default="True",
        type=str,
        help="save output files",
    )

    # fix seed
    seed_everything()
    if args is None:
        args = parser.parse_args()
        
    args.save_output = t_or_f(args.save_output)

    args.name = (
        "qp_"
        + str(args.density)
        + str(args.prune_ratio)
        + "_temperature_"
        + str(args.temperature)
        + "_num_centroids_"
        + str(args.num_centroids)
    )
    
    if args.cls_bal_ratio > 0.0:
        args.name += "_" + str(args.cls_bal_ratio) + "cls_bal"

    if args.which_to_keep == "random":
        args.name += "_keep_random_True"
    elif args.which_to_keep == "hard":
        args.name += "_keep_hard_True"
    elif args.which_to_keep == "easy":
        args.name += "_keep_hard_False"

    args.save_dir = os.path.join(args.save_dir, args.name)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger = get_logger(
        file_name=os.path.join(args.save_dir, "logs.log"),
        level=logging.INFO,
        stdout=True,
    )
    print_fn = logger.info
    args.print_fn = print_fn
    
    print_fn(f"Output will be saved to: {args.save_dir}")

    # Open a text file and write all the parameters
    file_path = os.path.join(args.save_dir, "params.txt")
    with open(file_path, "w") as file:
        # Write all the arguments to the file
        for arg in vars(args):
            value = getattr(args, arg)
            file.write(f"{arg}: {value}\n")

    args.sorted_clusters = load_arrays(
        args.sorted_clusters_path,
        args.num_centroids,
        logger=logger,
    )
    assert len(args.sorted_clusters) == args.num_centroids

    # compute density
    print_fn("get_distances")
    d_i = get_distances(args)
    print_fn("get_distances DONE")

    # turn d_i into a probability:
    temperature = args.temperature
    softmax = torch.nn.Softmax()
    probs = torch.Tensor(d_i) / temperature
    probs = softmax(probs)

    number_of_items_in_cluster = list()
    for i, cluster_i in enumerate(tqdm(args.sorted_clusters)):
        number_of_items_in_cluster.append(cluster_i.shape[0])
        
    print_fn(f"Total number of samples: {sum(number_of_items_in_cluster)}")
    
    # calculate the number of items per cluster according to QP pruning
    num_of_items_qp = list()
    pruned_dataset_size = int(args.prune_ratio * args.total_dataset_size)

    number_of_items_in_cluster_pruned = pruning_qp(
        probs.data.numpy(),
        number_of_items_in_cluster,
        pruned_dataset_size,
        args.num_centroids,
        args.print_fn,
        args.cls_bal_ratio,
    )

    np.save(
        os.path.join(
            args.save_dir,
            f"number_of_points_to_keep_per_cluster_density_{args.density}.npy",
        ),
        number_of_items_in_cluster_pruned,
    )

    # plot the final distribution
    plt.figure()
    indices_sort = np.argsort(probs)
    probs_s = probs[indices_sort]
    plt.plot(number_of_items_in_cluster_pruned[indices_sort], label="pruned clusters")
    plt.plot(probs_s * pruned_dataset_size, label="probs")
    plt.xlabel("Cluster indices")
    plt.ylabel("Number of samples per cluster")
    plt.title("keep ratio: " + str(args.prune_ratio))
    plt.legend()
    plt.show()
    plt.savefig(
        os.path.join(args.save_dir, "distribution_of_sampled_points_with_qp.jpg")
    )

    # plot the final distribution
    plt.figure()
    plt.scatter(
        range(len(number_of_items_in_cluster_pruned)),
        number_of_items_in_cluster_pruned[indices_sort],
        s=1.2,
        label="pruned clusters",
    )
    plt.plot(probs_s * pruned_dataset_size, c="orange", label="probs")
    plt.xlabel("Cluster indices")
    plt.ylabel("Number of samples per cluster")
    plt.title("keep ratio: " + str(args.prune_ratio))
    plt.legend()
    plt.show()
    plt.savefig(
        os.path.join(
            args.save_dir, "distribution_of_sampled_points_with_qp_scatter_plot.jpg"
        )
    )

    # get the pruned paths and build the trie file:
    pruning_fn = get_paths

    all_paths_pruned, all_distances, pruned_distances = pruning_fn(
        args, number_of_items_in_cluster_pruned
    )

    if not args.save_output:
        return all_paths_pruned

    if all_distances and pruned_distances:
        fname = os.path.join(args.save_dir, "distances_stats.npy")
        np.save(fname, [all_distances, pruned_distances])

    print_fn(
        f"Number of final paths, fraction of total dataset: , {len(all_paths_pruned)}, {len(all_paths_pruned) / args.total_dataset_size}",
    )
    with open(os.path.join(args.save_dir, args.name + "_paths.txt"), "w") as fp:
        fp.write("\n".join(all_paths_pruned))



if __name__ == "__main__":
    main()