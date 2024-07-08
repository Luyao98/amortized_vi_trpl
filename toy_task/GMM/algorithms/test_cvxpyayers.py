import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer


n, m = 2, 3
x = cp.Variable(n)
A = cp.Parameter((m, n))
b = cp.Parameter(m)
constraints = [x >= 0]
objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
problem = cp.Problem(objective, constraints)
assert problem.is_dpp()

cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])

A_tch = torch.randn(m, n, requires_grad=True)
b_tch = torch.randn(m, requires_grad=True)

optimizer = torch.optim.SGD([A_tch, b_tch], lr=0.01)

for i in range(100):
    optimizer.zero_grad()
    solution, = cvxpylayer(A_tch, b_tch)
    loss = solution.sum()
    loss.backward()
    optimizer.step()

    print(f"Iteration {i}, Loss: {loss.item()}")

print("Optimized A:", A_tch)
print("Optimized b:", b_tch)
