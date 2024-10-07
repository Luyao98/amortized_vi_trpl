from toy_task.GMM.projections.split_kl_projection import split_kl_projection

import torch as ch


# test split_kl_projection
batch_size = 4
n_components = 3
dz = 5

eps_mean = 0.1
eps_cov = 0.5

mean = ch.randn(batch_size, n_components, dz)
chol = ch.randn(batch_size, n_components, dz, dz)
chol = ch.tril(chol)

old_mean = ch.randn(batch_size, n_components, dz)
old_chol = ch.randn(batch_size, n_components, dz, dz)
old_chol = ch.tril(old_chol)

mean_proj, chol_proj = split_kl_projection(mean, chol, old_mean, old_chol, eps_mean, eps_cov)

print("Projected Mean:")
print(mean_proj)

print("\nProjected Cholesky Decomposition:")
print(chol_proj)

assert mean_proj.shape == mean.shape, f"Expected shape {mean.shape}, but got {mean_proj.shape}"
assert chol_proj.shape == chol.shape, f"Expected shape {chol.shape}, but got {chol_proj.shape}"

print("\nTest passed: Shapes are correct.")