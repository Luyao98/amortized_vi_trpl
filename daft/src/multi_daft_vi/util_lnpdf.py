import math
from typing import Callable, Tuple, Union

import torch


def mini_batch_function_no_grad(
        function: Callable[[torch.Tensor, bool], Tuple[torch.Tensor, Union[torch.Tensor, None]]],
        z: torch.Tensor,
        mini_batch_size: int,
) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
    """
    Computes the function where the function returns its evaluation only.
    z: (num_samples, batch_dim, d_z)
    mini_batch_size: size of the mini batches to split up the num_samples dimension
    Returns:
        function_result: (num_samples, batch_dim)
        grad: None
    """
    num_samples = z.shape[0]
    num_batches = math.ceil(num_samples / mini_batch_size)
    function_result = []
    for i in range(num_batches):
        z_batch = z[i * mini_batch_size: (i + 1) * mini_batch_size]
        function_result_batch, grad_batch = function(z_batch)
        function_result.append(function_result_batch)
    function_result = torch.cat(function_result, dim=0)
    return function_result, None


def mini_batch_function_grad(
        function_with_grad_return: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        z: torch.Tensor,
        mini_batch_size: int,
) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
    """
    Computes the function where the function returns its evaluation and its gradient wrt the input.
    z: (num_samples, batch_dim, d_z)
    mini_batch_size: size of the mini batches to split up the num_samples dimension
    Returns:
        function_result: (num_samples, batch_dim)
        grad: (num_samples, batch_dim, d_z)
    """
    num_samples = z.shape[0]
    num_batches = math.ceil(num_samples / mini_batch_size)
    function_result = []
    grad = []
    for i in range(num_batches):
        z_batch = z[i * mini_batch_size: (i + 1) * mini_batch_size]
        function_result_batch, grad_batch = function_with_grad_return(z_batch)
        function_result.append(function_result_batch)
        grad.append(grad_batch)
    function_result = torch.cat(function_result, dim=0)
    grad = torch.cat(grad, dim=0)
    return function_result, grad

