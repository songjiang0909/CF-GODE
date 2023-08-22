

# import numpy as np
# import torch
# from torch.autograd import Function

# """
# credits: code from https://github.com/jvanvugt/pytorch-domain-adaptation/
# """

# class GradientReversalFunction(Function):
#     """
#     Gradient Reversal Layer from:
#     Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
#     Forward pass is the identity function. In the backward pass,
#     the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
#     """

#     @staticmethod
#     def forward(ctx, x, lambda_):
#         ctx.lambda_ = lambda_
#         return x.clone()

#     @staticmethod
#     def backward(ctx, grads):
#         lambda_ = ctx.lambda_
#         lambda_ = grads.new_tensor(lambda_)
#         dx = -lambda_ * grads
#         return dx, None


# class GradientReversal(torch.nn.Module):
#     def __init__(self, lambda_=1):
#         super(GradientReversal, self).__init__()
#         self.lambda_ = lambda_

#     def forward(self, x):
#         return GradientReversalFunction.apply(x, self.lambda_)


import torch
from torch import nn
from torch.autograd import Function

class GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, *grad_output):
        return grad_output * -ctx.lambd, None