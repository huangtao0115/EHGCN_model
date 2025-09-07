import torch
import numpy as np
import scipy.sparse as sp
import umap

# Constant definitions
epsilon = 1e-15

k = 25  # Keep 25 most similar neighbors for each node
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def minmax_normalize(data):
    """Min-Max normalization to [0,1] range"""
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    return (data - data_min) / (data_max - data_min + 1e-8)

def hyperbolic_distance_poincare(x, y, epsilon=1e-15):
    """Efficient implementation of hyperbolic distance calculation"""
    norm_diff_squared = torch.sum((x - y) ** 2, dim=2)
    norm_x_squared = torch.sum(x ** 2, dim=2)
    norm_y_squared = torch.sum(y ** 2, dim=1)
    common_denominator = (1 - norm_x_squared) * (1 - norm_y_squared) + epsilon
    return torch.acosh(1 + 2 * norm_diff_squared / common_denominator)

def build_sparse_hyperbolic_adjacency(features, k, device):
    """Build sparse hyperbolic distance adjacency matrix"""
    num_nodes = features.shape[0]
    chunk_size = 1000  # Adjust chunk size based on memory
    adj_sparse = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)

    # Calculate distances in chunks and keep top-k neighbors
    for start in range(0, num_nodes, chunk_size):
        end = min(start + chunk_size, num_nodes)
        chunk = features[start:end].unsqueeze(1)  # (chunk_size, 1, dim)

        # Calculate hyperbolic distance from current chunk to all nodes
        dist_chunk = hyperbolic_distance_poincare(chunk, features, epsilon)

        # Handle diagonal (self-distance)
        diag_mask = torch.eye(dist_chunk.shape[0], dist_chunk.shape[1], device=device).bool()
        dist_chunk[diag_mask] = torch.inf  # Set self-distance to infinity

        # Get top-k nearest neighbors for each node (smallest distance)
        topk_vals, topk_indices = torch.topk(dist_chunk, k=k, dim=1, largest=False)

        # Convert to similarity (inverse of distance)
        sim_chunk = 1 / (topk_vals + epsilon)

        # Fill into sparse matrix
        for i in range(chunk.shape[0]):
            global_idx = start + i
            for j in range(k):
                neighbor_idx = topk_indices[i, j].item()
                sim_value = sim_chunk[i, j].item()
                adj_sparse[global_idx, neighbor_idx] = sim_value

    return adj_sparse.tocoo()

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)).flatten()
    r_inv = np.power(rowsum, -0.5)
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx).dot(r_mat_inv)

# ------------------------- Main processing pipeline -------------------------
print("Loading features...")
name = 'TREC'  # snippets  StackOverflow   ohsumed    TagMyNews   ohsumed   mr   Twitter   R52   mr    TREC
features = torch.load(f'../glove/embedding_{name}.pt').float()
features_np = features.cpu().detach().numpy()  # snippets

print("Min-Max normalization...")
features_np = minmax_normalize(features_np)

print("UMAP dimensionality reduction...")
reducer = umap.UMAP(
    n_components=30,
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42,
    low_memory=True
)
features_reduced = reducer.fit_transform(features_np)
features = torch.from_numpy(features_reduced).float().to(device)

print("Feature normalization...")
max_norms = torch.max(torch.abs(features), dim=1, keepdim=True)[0]
max_norms[max_norms == 0] = epsilon
features = features / max_norms

print("Building sparse hyperbolic distance adjacency matrix...")
adj_sparse = build_sparse_hyperbolic_adjacency(features, k, device)

print("Normalization processing...")
adj_normalized = normalize(adj_sparse)

print("Converting to PyTorch tensor and saving...")
adj_dense = adj_normalized.toarray()
adj_tensor = torch.FloatTensor(adj_dense)
torch.save(adj_tensor, f'H_adj_{name}.pt')