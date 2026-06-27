# dbscan_cluster.py
import numpy as np
from sld_cuda_wrapper import get_epsilon_neighbor_cuda
from config import EPS, W_SH, MIN_PTS

def dbscan_sh_sld(sh_mat: np.ndarray, sld_arr: np.ndarray) -> np.ndarray:
    n = sh_mat.shape[0]
    visited = np.zeros(n, dtype=bool)
    cluster_label = np.full(n, -1, dtype=np.int32)
    current_cluster = 0

    for p_idx in range(n):
        if visited[p_idx]:
            continue
        visited[p_idx] = True
        neighbors = get_epsilon_neighbor_cuda(sh_mat, sld_arr, p_idx, EPS, W_SH)
        n_nei = len(neighbors)

        if n_nei < MIN_PTS:
            cluster_label[p_idx] = -1
        else:
            current_cluster += 1
            cluster_label[p_idx] = current_cluster
            seed_list = neighbors.tolist()
            seed_ptr = 0
            while seed_ptr < len(seed_list):
                q_idx = seed_list[seed_ptr]
                if not visited[q_idx]:
                    visited[q_idx] = True
                    m_neighbors = get_epsilon_neighbor_cuda(sh_mat, sld_arr, q_idx, EPS, W_SH)
                    if len(m_neighbors) >= MIN_PTS:
                        seed_list.extend(m_neighbors.tolist())
                if cluster_label[q_idx] == -1:
                    cluster_label[q_idx] = current_cluster
                seed_ptr += 1
    return cluster_label

def map_crown_branch(cluster_ids: np.ndarray, sld_arr: np.ndarray) -> np.ndarray:
    valid_clusters = np.unique(cluster_ids[cluster_ids != -1])
    if len(valid_clusters) != 2:
        raise ValueError(f"聚类得到{len(valid_clusters)}簇")
    
    mean_sld_list = []
    for cid in valid_clusters:
        mask = cluster_ids == cid
        mean_sld_list.append(sld_arr[mask].mean())
    
    crown_cluster_id = valid_clusters[np.argmax(mean_sld_list)]
    branch_cluster_id = valid_clusters[np.argmin(mean_sld_list)]

    cl_label = np.full_like(cluster_ids, -1, dtype=np.int32)
    cl_label[cluster_ids == crown_cluster_id] = 1
    cl_label[cluster_ids == branch_cluster_id] = 0
    return cl_label