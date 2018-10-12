# coding: utf-8
import numpy
import torch
import thxx


def test_batch_eigh():
    if not torch.cuda.is_available():
        return
    for dtype in [torch.float32, torch.float64]:
        A = torch.rand(2, 5, 5, dtype=dtype, device=torch.device("cuda"))
        A = A.transpose(1, 2).matmul(A)
        w, V = thxx.backend.batch_eigh(A,
                                       False,
                                       True,
                                       1e-7,
                                       100,
                                       False)
        for i in range(A.shape[0]):
            a = A[i]
            e = V[i].t().matmul(w[i].diag()).matmul(V[i])
            torch.testing.assert_allclose(a, e)

def test_generalized_eigh():
    if not torch.cuda.is_available():
        return
    # A = torch.rand(3, 3).cuda()
    # A = A.transpose(0, 1).matmul(A)
    # B = torch.rand(3, 3).cuda()
    # B = thxx_backend_cuda.transpose(0, 1).matmul(B)
    # example from https://docs.nvidia.com/cuda/cusolver/index.html#sygvd-example1
    a = torch.cuda.FloatTensor(
        [[3.5, 0.5, 0.0],
         [0.5, 3.5, 0.0],
         [0.0, 0.0, 2.0]])
    b = torch.cuda.FloatTensor(
        [[10, 2, 3],
         [2, 10, 5],
         [3, 5, 10]])
    w_expect = torch.cuda.FloatTensor([0.158660256604, 0.370751508101882, 0.6])
    for upper in [True, False]:
        for jacob in [True, False]:
            w, V, L = thxx.backend.generalized_eigh(a, False, b, False, upper, jacob, 1e-7, 100)
            torch.testing.assert_allclose(w, w_expect)
            torch.testing.assert_allclose(V.mm(b).mm(V.t()), torch.eye(a.shape[0], device=a.device))
            for i in range(3):
                torch.testing.assert_allclose(a.matmul(V[i]), b.matmul(V[i]) * w[i])


def test_batch_svd():
    if not torch.cuda.is_available():
        return
    # example from https://docs.nvidia.com/cuda/cusolver/index.html#batchgesvdj-example1
    A = torch.cuda.FloatTensor(
        [[[ 1, -1],
          [-1,  2],
          [ 0,  0]],
         [[3, 4],
          [4, 7],
          [0, 0]]]) # .transpose(1, 2).contiguous()
    s_expect = torch.cuda.FloatTensor(
        [[2.6180, 0.382],
         [9.4721, 0.5279]])
    U, s, V = thxx.backend.batch_svd(A, False, 0.0, 100)

    # FIXME not matched
    print(s_expect)
    print(s)

    # # s (2, 2) -> (2, 3)
    for i in range(A.shape[0]):
        spad = torch.zeros(3, 2, device=A.device)
        spad.diagonal()[:2] = s[i]
        print(i)
        # FIXME not matched
        print(A[i])
        print(U[i].t().mm(spad).mm(V[i]))


def test_batch_matinv():
    if not torch.cuda.is_available():
        return

    for dtype in [torch.float32, torch.float64]:
        _a = torch.randn(2, 3, 3, dtype=dtype, device=torch.device("cuda"))
        for a in [_a, _a.transpose(1, 2)]:
            ai = thxx.backend.batch_matinv(a)
            for i in range(a.shape[0]):
                torch.testing.assert_allclose(
                    a[i].mm(ai[i]),
                    torch.eye(a.shape[1], dtype=dtype, device=a.device))


def test_batch_complex_matinv():
    if not torch.cuda.is_available():
        return

    for dtype in [torch.float32, torch.float64]:
        _a = torch.randn(2, 3, 3, 2, dtype=dtype, device=torch.device("cuda"))
        for a in [_a, _a.transpose(1, 2)]:
            ai = thxx.backend.batch_complex_matinv(a)
            id = thxx.backend.batch_complex_mm(a, ai)
            for i in range(a.shape[0]):
                torch.testing.assert_allclose(
                    id[i, :, :, 0],
                    torch.eye(a.shape[1], dtype=dtype, device=a.device))
                torch.testing.assert_allclose(
                    id[i, :, :, 1],
                    torch.zeros(a.shape[1], dtype=dtype, device=a.device))


def test_complex_mm():
    for d in ["cpu", "cuda"]:
        if d == "cuda":
            if not torch.cuda.is_available():
                continue
        dev = torch.device(d)
        for dtype in [torch.float32, torch.float64]:
            ab = [
                (torch.randn(4, 3, 2, device=dev, dtype=dtype),
                 torch.randn(3, 2, 2, device=dev, dtype=dtype)),
                (torch.randn(3, 4, 2, device=dev, dtype=dtype).transpose(0, 1),
                 torch.randn(3, 2, 2, device=dev, dtype=dtype)),
                (torch.randn(4, 3, 2, device=dev, dtype=dtype),
                 torch.randn(2, 3, 2, device=dev, dtype=dtype).transpose(0, 1)),
                (torch.randn(3, 4, 2, device=dev, dtype=dtype).transpose(0, 1),
                 torch.randn(2, 3, 2, device=dev, dtype=dtype).transpose(0, 1)),
            ]
            for a, b in ab:
                c = thxx.backend.complex_mm(a, b).cpu()
                a = a.cpu()
                b = b.cpu()
                for i in range(c.shape[0]):
                    for j in range(c.shape[1]):
                        ai = a[i, :].t()
                        bj = b[:, j].t()
                        # ar * br - ai * bi
                        cr = sum(ai[0] * bj[0] - ai[1] * bj[1])
                        # ai * br + ar * bi
                        ci = sum(ai[1] * bj[0] + ai[0] * bj[1])
                        torch.testing.assert_allclose(c[i, j, 0], cr)
                        torch.testing.assert_allclose(c[i, j, 1], ci)


def test_batch_complex_mm():
    for d in ["cpu", "cuda"]:
        if d == "cuda":
            if not torch.cuda.is_available():
                continue
        dev = torch.device(d)
        for dtype in [torch.float32, torch.float64]:
            ab = [
                (torch.randn(5, 4, 3, 2, device=dev, dtype=dtype),
                 torch.randn(5, 3, 2, 2, device=dev, dtype=dtype)),
                (torch.randn(5, 3, 4, 2, device=dev, dtype=dtype).transpose(1, 2),
                 torch.randn(5, 3, 2, 2, device=dev, dtype=dtype)),
                (torch.randn(5, 4, 3, 2, device=dev, dtype=dtype),
                 torch.randn(5, 2, 3, 2, device=dev, dtype=dtype).transpose(1, 2)),
                (torch.randn(5, 3, 4, 2, device=dev, dtype=dtype).transpose(1, 2),
                 torch.randn(5, 2, 3, 2, device=dev, dtype=dtype).transpose(1, 2)),
            ]
            for a, b in ab:
                c = thxx.backend.batch_complex_mm(a, b).cpu()
                a = a.cpu()
                b = b.cpu()
                for k in range(c.shape[0]):
                    for i in range(c.shape[1]):
                        for j in range(c.shape[2]):
                            ai = a[k, i, :].t()
                            bj = b[k, :, j].t()
                            # ar * br - ai * bi
                            cr = sum(ai[0] * bj[0] - ai[1] * bj[1])
                            # ai * br + ar * bi
                            ci = sum(ai[1] * bj[0] + ai[0] * bj[1])
                            torch.testing.assert_allclose(c[k, i, j, 0], cr)
                            torch.testing.assert_allclose(c[k, i, j, 1], ci)


if __name__ == "__main__":
    # wip
    test_batch_svd()

    # float only
    test_generalized_eigh()

    # float/double done
    test_batch_eigh()
    test_batch_matinv()
    test_batch_complex_matinv()
    test_complex_mm()
    test_batch_complex_mm()
