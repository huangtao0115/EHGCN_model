import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix, lil_matrix
from umap import UMAP
import gc

k = 25  # Keep 25 most similar neighbors for each node

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx).dot(r_mat_inv)

name = 'TREC'  # mr, snippets, Twitter   StackOverflow  ohsumed   TagMyNews     Twitter   R52    mr   TREC
# 1. Load features and normalize
features = torch.load(f'../glove/embedding_{name}.pt').float()      #TagMyNews
print('Number of features:', len(features))
max_norms = torch.max(torch.abs(features), dim=1, keepdim=True)[0]
max_norms[max_norms == 0] = 1
features = features / max_norms

# 2. UMAP dimensionality reduction
print("Starting UMAP dimensionality reduction...")
# features_np = features.cpu().numpy()
features_np = features.cpu().detach().numpy().copy()
del features
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

umap_reducer = UMAP(
    n_components=30,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    random_state=42,
    low_memory=True
)

features_reduced = umap_reducer.fit_transform(features_np)
features = torch.tensor(features_reduced, dtype=torch.float32)
del features_np, features_reduced
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
print(f"Dimensionality reduction completed, new feature dimension: {features.shape}")

# 3. Calculate global minimum non-zero distance
num_nodes = features.shape[0]
chunk_size = 1000
print(f"Calculating global minimum non-zero distance, chunk size: {chunk_size}")

min_non_zero = float('inf')
for i in range(0, num_nodes, chunk_size):
    chunk_end = min(i + chunk_size, num_nodes)
    chunk = features[i:chunk_end]

    # Calculate distance from current chunk to all nodes
    dist_chunk = torch.cdist(chunk, features, p=2)

    # Create mask to identify zero-distance positions
    zero_mask = (dist_chunk == 0)

    # Exclude zero-distance values
    non_zero_distances = dist_chunk[~zero_mask]

    if non_zero_distances.numel() > 0:
        chunk_min = torch.min(non_zero_distances).item()
        if chunk_min < min_non_zero:
            min_non_zero = chunk_min

if min_non_zero == float('inf'):
    min_non_zero = 1.0

max_similarity = 1 / min_non_zero
special_value = max_similarity * 10
print(f"Global minimum non-zero distance: {min_non_zero:.6f}, Special value: {special_value:.2f}")

# 4. Sparsification strategy - Only keep top-k similarities
print(f"Applying sparsification strategy, keeping top-{k} neighbors for each node")

# Use LIL format for row-by-row operations
adj_sparse = lil_matrix((num_nodes, num_nodes), dtype=np.float32)

for i in range(0, num_nodes, chunk_size):
    chunk_end = min(i + chunk_size, num_nodes)
    chunk = features[i:chunk_end]

    # Calculate distance from current chunk to all nodes
    dist_chunk = torch.cdist(chunk, features, p=2)

    # Create mask to identify zero-distance positions
    zero_mask = (dist_chunk == 0)

    # Build similarity matrix chunk
    sim_chunk = torch.where(
        zero_mask,
        torch.full_like(dist_chunk, special_value),
        1 / dist_chunk
    )

    # Convert to NumPy array
    sim_chunk_np = sim_chunk.cpu().numpy()

    # For each node in current chunk, keep only top-k similarity indices
    for j in range(sim_chunk_np.shape[0]):
        global_idx = i + j

        # Get all similarity values for this node
        row_sims = sim_chunk_np[j]

        # Exclude self-connections
        row_sims[global_idx] = 0

        # Find top-k similarity indices
        topk_indices = np.argpartition(row_sims, -k)[-k:]

        # Keep only these connections
        for col_idx in topk_indices:
            if row_sims[col_idx] > 0:  # Only keep positive similarities
                adj_sparse[global_idx, col_idx] = row_sims[col_idx]

    # Free memory
    del dist_chunk, sim_chunk, sim_chunk_np
    gc.collect()

print(f"Sparsification completed, number of non-zero elements: {adj_sparse.nnz}")

# 5. Symmetrize adjacency matrix (optional)
print("Symmetrizing adjacency matrix...")
adj_sparse = adj_sparse.maximum(adj_sparse.T)

# 6. Normalization processing
print("Normalization processing...")
adj_normalized = normalize(adj_sparse.tocoo())

print("Saving adjacency matrix in chunks...")
torch.save(adj_normalized, f'D_adj_{name}.pt')

# Get coordinates and data in COO format
adj_coo = adj_normalized.tocoo()
rows = adj_coo.row
cols = adj_coo.col
values = adj_coo.data

# Create PyTorch sparse tensor
indices = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
values = torch.tensor(values, dtype=torch.float)
shape = torch.Size(adj_coo.shape)
sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)

# Save sparse tensor
output_file = f'D_adj_{name}.pt'
torch.save(sparse_tensor, output_file)
print(f"File saved as {output_file}")