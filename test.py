import torch

import torch_autograd_solver as S
from torch.autograd import gradcheck

def test_runtime():
    """test that there are no runtime errors"""
    from torch.autograd import Variable
    import torch.nn as nn
    import torch.nn.functional as F
    x = torch.randn(30,10)
    w = nn.Parameter(torch.rand(30,10), requires_grad=True)
    xw = F.linear(x, w)
    a, b = S.symeig(xw)
    asum = a.sum()
    asum.backward()

def test_gradcheck():
    """test gradcheck"""
    input = torch.randn(5,5).double()
    input.requires_grad=True
    for upper in (True, False):
        assert gradcheck(S.BasicSymeig(upper=upper), (input,), eps=1e-6, atol=1e-4)

def test_symeig():
    # NOTE need pytorch 0.5.0 or 1.0
    a = torch.tensor([[ 1.96,  0.00,  0.00,  0.00,  0.00],
                      [-6.49,  3.80,  0.00,  0.00,  0.00],
                      [-0.47, -6.39,  4.17,  0.00,  0.00],
                      [-7.20,  1.50, -1.51,  5.70,  0.00],
                      [-0.65, -6.34,  2.67,  1.80, -7.10]]).t()
    a.requires_grad = True
    w, v = torch.symeig(a, eigenvectors=True)
    v.sum().backward()
    print(v.grad)


# test_symeig()
test_runtime()
test_gradcheck()
