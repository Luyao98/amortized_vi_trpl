from typing import Tuple, Union

import torch


def weighted_logsumexp(
    logx: torch.Tensor,
    w: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    return_sign: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    This is a Pytorch port of the Tensorflow function `reduce_weighted_logsumexp` from
    https://www.tensorflow.org/probability/api_docs/python/tfp/math/reduce_weighted_logsumexp
    Computes log(abs(sum(weight * exp(elements across tensor dimensions)))) in a numerically stable way.
    Right now, it only supports to perform the operation over 1 dimension. (mandatory parameter)
    :param logx: Tensor to reduce
    :param w: weights
    :param dim: dimension to reduce
    :param keep_dim: if True, retains reduced dimensions with length 1
    :param return_sign: if True, return the sign of weight * exp(elements across tensor dimensions)))
    :return: Either the reduced tensor or a tuple of the reduced tensor and the sign
    """
    log_absw_x = logx + torch.log(torch.abs(w))
    max_log_absw_x = torch.amax(log_absw_x, dim=dim, keepdim=True)
    max_log_absw_x = torch.where(
        torch.isinf(max_log_absw_x),
        torch.zeros(torch.Size([]), dtype=max_log_absw_x.dtype, device=max_log_absw_x.device),
        max_log_absw_x)
    wx_over_absw_x = torch.sign(w) * torch.exp(log_absw_x - max_log_absw_x)
    sum_wx_over_max_absw_x = torch.sum(wx_over_absw_x, dim=dim, keepdim=keepdim)
    if not keepdim:
        max_log_absw_x = torch.squeeze(max_log_absw_x, dim=dim)
    sgn = torch.sign(sum_wx_over_max_absw_x)
    lswe = max_log_absw_x + torch.log(sgn * sum_wx_over_max_absw_x)
    if return_sign:
        return lswe, sgn
    else:
        return lswe


if __name__ == "__main__":
    x = torch.tensor([[0.0, 0, 0], [0, 0, 0]])

    w = torch.tensor([[-1.0, 1, 1], [1, 0., 1]])

    print(weighted_logsumexp(x, w, dim=1))
