__all__ = ['greedy_batch_requests', 'naive_batch_requests', 'greedy_grouping_mini_batch']

import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np


def calculate_intersection(
        clusters: list, device: torch.device | None = None
) -> tuple[Tensor, Tensor]:
    """Calculate the number of clusters that overlap between each pair
    of data points.

    The function will calculate two values:
    1. The number of clusters that overlap between each pair of data points.
    2. The indices of the data points sorted by the number of overlapping
       clusters in descending order.

    Args:
        clusters (list): A list of clusters for each data point.
        device (torch.device | None): The device to use for the calculations.
         Defaults to None.
    """
    n_data = len(clusters)
    intersect = torch.zeros(n_data * n_data)
    if device is not None:
        intersect = intersect.to(device)
    for i in range(n_data):
        for j in range(n_data):
            if i == j: continue
            intersect[i * n_data + j] = torch.isin(clusters[i], clusters[j]) \
                .sum().item()
    return intersect, torch.argsort(intersect, descending=True)


def greedy_batch_requests(clusters: list, size: int, start_idx=0) -> list:
    """
    Return a list of lists, each containing size elements from the data.

    Algorithm:
    1. Use the calculate_intersection function to calculate the number of
       clusters that overlap between each pair of data points.
    2. From the biggest overlap to the smallest to batch the data points
       together.
    3. If the batch size is 2, end; otherwise, recursively call the do this
       process with batched ones as one request.

    This algorithm is 1/2-optimal.
    """
    return batch_recursive(clusters, size, [[i + start_idx] for i in range(len(clusters))])


def batch_recursive(cluster: list, size: int, batched: list) -> list:
    """
    Recursively batch the data points together.
    """
    if size == 1:
        return batched
    assert size % 2 == 0
    intersect, sorted = calculate_intersection(cluster)
    num = len(cluster)
    assert num == len(batched)


    in_batch = [False] * num
    result = []
    new_cluster = []
    for i in range(len(sorted)):
        request1 = sorted[i] // num
        request2 = sorted[i] % num
        if request1 == request2:
            continue
        if in_batch[request1] or in_batch[request2]:
            continue
        in_batch[request1] = True
        in_batch[request2] = True
        batch = batched[request1] + batched[request2]
        result.append(batch)
        new_cluster.append(torch.unique(torch.cat((cluster[request1], cluster[request2]))))
    remain = []
    remain_cluster = None
    for i, added in enumerate(in_batch):
        if not added:
            remain.append(batched[i])
            if remain_cluster is None:
                remain_cluster = cluster[i]
            else:
                remain_cluster.append(cluster[i])
    if len(remain) > 0:
        result += remain
        new_cluster.append(remain_cluster)
    if size == 2:
        return result
    return batch_recursive(new_cluster, size // 2, result)


def naive_batch_requests(total_data: int, batch_size: int, start_idx=0) -> list:
    """
    Naive batching algorithm.
    """
    if batch_size == -1:
        batch_size = total_data

    result = []
    for i in range(start_idx, total_data + start_idx, batch_size):
        result.append([j for j in range(i, min(i + batch_size, total_data + start_idx))])
    return result

def greedy_grouping_mini_batch(requests, embeddings, group_size=4):
    """
    Groups embeddings into clusters using an optimized greedy strategy.
    This implementation uses vectorized tensor operations, a boolean mask,
    and torch.topk for fast computations on GPU.
    
    Args:
      embeddings (torch.Tensor): A 2D tensor of shape (N, D) where N is the
                                 number of embeddings.
      group_size (int): Size of each group (default=4).
    
    Returns:
      groups (list[list[int]]): A list of groups, each containing indices of the embeddings.
    """
    n = embeddings.size(0)
    
    # Ensure embeddings are normalized for cosine similarity (dot product on normalized vectors)
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    
    # Compute the cosine similarity matrix (N x N)
    sim_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
    
    # Create a boolean mask for indices that are not yet grouped.
    mask = torch.ones(n, dtype=torch.bool, device=embeddings.device)
    groups = []
    
    while mask.sum() > 0:
        # Get the tensor of indices that are still available.
        remaining_indices = torch.nonzero(mask).view(-1)
        
        # If there are fewer remaining embeddings than the desired group_size, group them all.
        if remaining_indices.numel() <= group_size:
            groups.append(remaining_indices.tolist())
            break
        
        # ---- Seed selection with vectorized row-sum calculation ----
        # Extract the submatrix for the remaining indices.
        submatrix = sim_matrix[remaining_indices][:, remaining_indices].clone()
        # Zero-out self-similarity along the diagonal.
        submatrix.fill_diagonal_(0)
        # Compute the row-sum for each remaining embedding.
        row_sums = submatrix.sum(dim=1)
        # Find the position in remaining_indices with the highest total similarity.
        max_pos = torch.argmax(row_sums)
        seed_idx = remaining_indices[max_pos].item()

        # ---- Select top (group_size - 1) similar embeddings using vectorized topk ----
        # Get the similarities between the seed and all remaining indices.
        seed_sim = sim_matrix[seed_idx][remaining_indices].clone()
        # Exclude self by setting its similarity to -inf.
        seed_pos = (remaining_indices == seed_idx)
        seed_sim[seed_pos] = -float('inf')
        # Select the top (group_size - 1) embeddings for the new group.
        topk_vals, topk_indices = torch.topk(seed_sim, group_size - 1)
        group_members = remaining_indices[topk_indices].tolist()
        # Form the new group: the seed plus its top similar embeddings.
        group = [seed_idx] + group_members
        groups.append(group)
        
        # Mark these indices as grouped by setting corresponding mask positions to False.
        mask[group] = False

    if not isinstance(requests, np.ndarray):
        requests = np.array(requests)
    
    mini_batch_results = []
    for group in groups:
        mini_batch_results.append(requests[group])

    return np.array(mini_batch_results)