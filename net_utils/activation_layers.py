import torch
import torch.nn as nn


class Swish(torch.autograd.Function):
    """
    code from: https://github.com/AntanasKascenas/DenoisingAE/

    License:
        GNU GENERAL PUBLIC LICENCE Version 2, June 1991
    """
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class CustomSwish(nn.Module):
    def forward(self, input_tensor):
        return Swish.apply(input_tensor)