import numpy as np
import torch
from scipy.optimize import minimize

def optimize_lambda(epsilon, q, r):
    device = q.device  # Get the device of q

    # Define the function g(lambda) using PyTorch
    def g_torch(lambda_, epsilon, q, r):
        term1 = -lambda_ * epsilon
        term2 = -(1 + lambda_) * torch.log(torch.sum(q ** (1 / (1 + lambda_)) * r ** (lambda_ / (1 + lambda_))))
        return term1 + term2

    # Convert g to a function that can be used with scipy.optimize.minimize
    def g_numpy(lambda_numpy, epsilon, q, r):
        lambda_torch = torch.tensor(lambda_numpy, dtype=torch.float32, device=device)
        q_torch = torch.tensor(q, dtype=torch.float32, device=device)
        r_torch = torch.tensor(r, dtype=torch.float32, device=device)
        g_value = g_torch(lambda_torch, epsilon, q_torch, r_torch)
        return g_value.item()

    # Initial guess for lambda
    lambda_initial = np.array([0.1], dtype=np.float32)

    # Bounds for lambda (lambda >= 0)
    bounds = [(0, None)]

    # Perform the optimization for the given sample
    with torch.no_grad():
        q_numpy = q.cpu().numpy()
        r_numpy = r.cpu().numpy()
        result = minimize(g_numpy, lambda_initial, args=(epsilon, q_numpy, r_numpy), bounds=bounds, method='L-BFGS-B')

    # Return the optimal lambda
    return result.x[0]

def calculate_p_i_star(optimal_lambda, q, r):
    numerator =  (q ** (1 / (1 + optimal_lambda))) * (r ** (optimal_lambda / (1 + optimal_lambda)))
    denominator = torch.sum((q ** (1 / (1 + optimal_lambda))) * (r ** (optimal_lambda / (1 + optimal_lambda))))
    log_p = torch.log(numerator) - torch.log(denominator)
    return log_p

def get_optimal_p_batch(q_batch, r_batch, epsilon):
    device = q_batch.device
    batch_size = q_batch.size(0)
    p_i_star_batch = []

    for i in range(batch_size):
        # Optimize lambda for each sample in the batch
        q = q_batch[i]
        r = r_batch[i]
        optimal_lambda = optimize_lambda(epsilon, q, r)

        # Convert optimal lambda to PyTorch tensor
        optimal_lambda_torch = torch.tensor(optimal_lambda, dtype=torch.float32, device=device)

        # Calculate p_i_star using the optimal lambda
        p_i_star = calculate_p_i_star(optimal_lambda_torch, q, r)
        p_i_star_batch.append(p_i_star)

    # Stack results into a batch
    p_i_star_batch = torch.stack(p_i_star_batch)
    return p_i_star_batch

# Example usage
batch_size = 4
n_components = 3
epsilon = 0.00001

q_batch = torch.tensor([[1, 2, 3], [0.3, 0.4, 0.3], [0.4, 0.3, 0.3], [0.1, 0.6, 0.3]], requires_grad=True, device='cuda')
r_batch = torch.tensor([[4, 5, 6], [0.2, 0.5, 0.3], [0.3, 0.4, 0.3], [0.4, 0.3, 0.3]], device='cuda')

p_i_star_batch = get_optimal_p_batch(q_batch, r_batch, epsilon)

print("Calculated p_i^* values for the batch:", p_i_star_batch)
print("p_i^* requires grad:", p_i_star_batch.requires_grad)
p_i_star_batch.sum().backward()
print("Gradients of q_batch:", q_batch.grad)
