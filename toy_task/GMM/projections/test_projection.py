from toy_task.GMM.projections.split_kl_projection import split_kl_projection
import torch as ch

# test split_kl_projection
batch_size = 640
n_components = 15
dz = 2

eps_mean = 1e-6
eps_cov = 1e-6

mean = ch.randn(batch_size, n_components, dz)
A = ch.randn(batch_size, n_components, dz, dz)
chol = ch.matmul(A, A.transpose(-2, -1)) + ch.eye(dz).unsqueeze(0).unsqueeze(0) * 1e-3
chol = ch.linalg.cholesky(chol)

old_mean = ch.randn(batch_size, n_components, dz)
B = ch.randn(batch_size, n_components, dz, dz)
old_chol = ch.matmul(B, B.transpose(-2, -1)) + ch.eye(dz).unsqueeze(0).unsqueeze(0) * 1e-3
old_chol = ch.linalg.cholesky(old_chol)


start_event = ch.cuda.Event(enable_timing=True)
end_event = ch.cuda.Event(enable_timing=True)

start_event.record()

mean_proj_flatten, chol_proj_flatten = split_kl_projection(mean.view(-1, dz).cuda(), chol.view(-1, dz, dz).cuda(),
                                                           old_mean.view(-1, dz).cuda(),
                                                           old_chol.view(-1, dz, dz).cuda(),
                                                           eps_mean, eps_cov)

mean_proj = mean_proj_flatten.view(batch_size, n_components, dz)
chol_proj = chol_proj_flatten.view(batch_size, n_components, dz, dz)
end_event.record()

# Wait for everything to finish running on the GPU
ch.cuda.synchronize()

elapsed_time = start_event.elapsed_time(end_event)  # in milliseconds
print(f"Time taken for split_kl_projection: {elapsed_time / 1000:.4f} seconds")

assert mean_proj.shape == mean.shape, f"Expected shape {mean.shape}, but got {mean_proj.shape}"
assert chol_proj.shape == chol.shape, f"Expected shape {chol.shape}, but got {chol_proj.shape}"
print("\nTest passed: Shapes are correct.")
