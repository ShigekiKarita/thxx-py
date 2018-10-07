#include <torch/torch.h>

#include <iostream>

using at::Tensor;

// forked from pytorch 0.5
// https://github.com/sethah/pytorch/blob/81b61db9219ffeb8fc0c8ab3abe0f0b5a7edf4f4/tools/autograd/templates/Functions.cpp#L1514
// http://eprints.maths.ox.ac.uk/1079/1/NA-08-01.pdf
Tensor symeig_backward(
    // backward variables
    const Tensor& grad_loss_wrt_eigenvalues, // [m]
    const Tensor& grad_loss_wrt_eigenvectors, // [m, m]
    // forward variables
    const Tensor& x, // [m, m]
    const Tensor& eigenvalues, // [m]
    const Tensor& eigenvectors, // [m, m]
    // config
    bool upper)
{
    Tensor gx; // [m, m]
    auto vt = eigenvectors.t();
    if (grad_loss_wrt_eigenvectors.defined())
    {
        Tensor F = eigenvalues.unsqueeze(0).expand_as(x).clone(); // [m, m]
        F.sub_(at::unsqueeze(eigenvalues, 1));
        // auto F = at::zeros_like(x);
        F.diagonal().fill_(INFINITY);
        F.pow_(-1);
        F.mul_(vt.mm(grad_loss_wrt_eigenvectors));
        gx = eigenvectors.mm(F.mm(vt));
    }
    if (grad_loss_wrt_eigenvalues.defined())
    {
        auto gx_gw = (eigenvectors * grad_loss_wrt_eigenvalues).mm(vt);
        if (gx.defined()) {
            gx.add_(gx_gw);
        } else {
            gx = gx_gw;
        }
    }
    if (upper)
    {
        auto gxu = at::triu(gx.t(), 1);
        gx.triu_().add_(gxu);
    }
    else
    {
        auto gxl = at::tril(gx.t(), -1);
        gx.tril_().add_(gxl);
    }
    return gx;
}

std::tuple<Tensor, Tensor> batch_symeig_forward(const Tensor& input, bool upper)
{
    auto batch_size = input.size(0);
    auto n = input.size(1);
    auto w = at::empty({batch_size, n}, input.type());
    auto v = at::empty({batch_size, n, n}, input.type());
#pragma omp for
    for (int64_t i = 0; i < batch_size; ++i)
    {
        at::Tensor wi, vi;
        // FIXME use syev directly to avoid fragmented memory alloc https://github.com/pytorch/pytorch/blob/695465915a88f4803dfae152151bb56be5c99410/aten/src/TH/generic/THTensorLapack.cpp#L361
        std::tie(wi, vi) = at::symeig(input.select(0, i), true, upper);
        w.select(0, i).copy_(wi);
        v.select(0, i).copy_(vi);
    }
    return {w, v};
}

Tensor batch_symeig_backward(
    // backward variables
    const Tensor& grad_loss_wrt_eigenvalues, // [b, m]
    const Tensor& grad_loss_wrt_eigenvectors, // [b, m, m]
    // forward variables
    const Tensor& x, // [b, m, m]
    const Tensor& eigenvalues, // [b, m]
    const Tensor& eigenvectors, // [b, m, m]
    // config
    bool upper)
{
    Tensor gx; // [b, m, m]
    auto batch_size = x.size(0);
    auto m = x.size(1);
    auto vt = eigenvectors.transpose(1, 2);
    if (grad_loss_wrt_eigenvectors.defined())
    {
        auto F = eigenvalues.unsqueeze(1).expand_as(x).clone();
        F.sub_(at::unsqueeze(eigenvalues, 2));
        F.diagonal(0, 1, 2).fill_(INFINITY);
        F.reciprocal_();
        F.mul_(vt.bmm(grad_loss_wrt_eigenvectors));
        gx = eigenvectors.bmm(F.bmm(vt));
    }
    if (grad_loss_wrt_eigenvalues.defined())
    {
        auto gw_gx = (eigenvectors * grad_loss_wrt_eigenvalues.unsqueeze(-1)).bmm(vt);
        if (gx.defined()) {
            gx.add_(gw_gx);
        } else {
            gx = gw_gx;
        }
    }

    // TODO implemnt batch_triu/l
    if (upper)
    {
#pragma omp for
        for (int64_t i = 0; i < batch_size; ++i) {
            auto&& gxi = gx.select(0, i);
            auto gxu = at::triu(gxi.t(), 1);
            gxi.triu_().add_(gxu);
        }
    }
    else
    {
#pragma omp for
        for (int64_t i = 0; i < batch_size; ++i) {
            auto&& gxi = gx.select(0, i);
            auto gxl = at::tril(gxi.t(), -1);
            gxi.tril_().add_(gxl);
        }
    }
    return gx;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("symeig_backward", &symeig_backward, "basic symeig backward");
    m.def("batch_symeig_forward", &batch_symeig_forward, "batch symeig forward");
    m.def("batch_symeig_backward", &batch_symeig_backward, "batch symeig backward");
}
