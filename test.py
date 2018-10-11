import torch

import torch_autograd_solver as S
from torch.autograd import gradcheck

def test_runtime():
    """test that there are no runtime errors"""
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


def test_batch_symeig_forward():
    xs = torch.randn(3, 5, 5).double()
    ws, vs = S.symeig(xs)
    for i in range(xs.shape[0]):
        w, v = S.symeig(xs[i])
        torch.testing.assert_allclose(ws[i], w)
        torch.testing.assert_allclose(vs[i], v)

def test_batch_symeig_backward():
    input = torch.randn(2, 3, 3).double()
    input1 = input.clone()
    input.requires_grad = True
    w, v = S.symeig(input)
    (w.sum() + v.sum()).backward()
    # print(input.grad)
    for i in range(input1.size(0)):
        in1 = input1[i]
        in1.requires_grad = True
        wi, vi = S.symeig(in1)
        (wi.sum() + vi.sum()).backward()
        # print(in1.grad)
        torch.testing.assert_allclose(input.grad[i], in1.grad)

def test_generalized_symeig_forward():
    a = torch.randn(3, 3).double()
    a = a.t().mm(a)
    b = torch.randn(3, 3).double()
    b = b.t().mm(b)
    w, v = S.symeig(a, b)
    torch.testing.assert_allclose(a.mm(v), w * b.mm(v))

def test_generalized_symeig_backward():
    a = torch.randn(3, 3).double()
    a = a.t().mm(a)
    a.requires_grad=True
    b = torch.randn(3, 3).double()
    b = b.t().mm(b)
    b.requires_grad=True
    assert gradcheck(S.GeneralizedSymeig(), (a, b), eps=1e-6, atol=1e-4)


# test_symeig() # need pytorch 0.5.0 or later
test_runtime()
test_gradcheck()
test_batch_symeig_forward()
test_batch_symeig_backward()
test_generalized_symeig_forward()
test_generalized_symeig_backward()
