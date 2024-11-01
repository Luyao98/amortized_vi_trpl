import torch as ch
import time

from toy_task.GMM.projections.gate_projecion import kl_projection_gate, create_kl_projection_problem, kl_projection_gate_non_batch


batch_size = 1000
n_components = 15
a = ch.rand(batch_size, n_components, dtype=ch.float32, requires_grad=True)
b = ch.rand(batch_size, n_components, dtype=ch.float32, requires_grad=True)

b_gate_old = ch.softmax(a, dim=1).cuda()
gate_pred = ch.softmax(b, dim=1).cuda()

epsilon = 0.01

n_components = n_components
cvxpylayer = create_kl_projection_problem(n_components)

start_time = time.time()

projected_gates = kl_projection_gate(b_gate_old, gate_pred, epsilon, cvxpylayer)
# projected_gates_non_batch = kl_projection_gate_non_batch(b_gate_old, gate_pred, epsilon)

end_time = time.time()


print("Sum along each row:", projected_gates.sum(dim=1))
# print("Sum along each row:", projected_gates_non_batch.sum(dim=1))

print(f"Time taken for kl_projection_gate: {end_time - start_time:.6f} seconds")


# result: 1000*15, batched takes 5.242s, non-batched can not be calculated,caused warning, requires more memory

