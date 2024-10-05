from typing import Callable, Tuple, Union

import torch


def eval_fn_grad(
    fn: Callable, z: torch.Tensor, compute_grad: bool
) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
    # check input
    n_samples = z.shape[0]
    batch_dim = z.shape[1:-1]
    assert len(batch_dim) <= 1
    d_z = z.shape[-1]

    if not compute_grad:
        f_z, f_z_grad, f_z_hess = _eval_fn(fn=fn, z=z), None, None
    else:
        z.requires_grad = True
        f_z = _eval_fn(fn=fn, z=z)
        f_z_grad = torch.zeros_like(z)
        for i in range(n_samples):
            if len(batch_dim) > 0:
                for j in range(batch_dim[0]):
                    f_z_grad[i, j] = torch.autograd.grad(
                        outputs=f_z[i, j], inputs=z, retain_graph=True
                    )[0][i, j]
            else:
                f_z_grad[i] = torch.autograd.grad(outputs=f_z[i], inputs=z, retain_graph=True)[0][i]
        f_z_grad = f_z_grad.detach()

    return f_z, f_z_grad


def _eval_fn(fn, z: torch.Tensor) -> torch.Tensor:
    # compute fn(z)
    f_z = fn(z)
    return f_z


if __name__ == "__main__":
    # autograd example
    z = torch.arange(2).reshape(1, 1, 2).float() + 1
    f_z, f_z_grad = eval_fn_grad(fn=lambda z: torch.sum(3 * z * z, dim=-1), z=z, compute_grad=True)
    print(f_z)
    print(f_z_grad)
