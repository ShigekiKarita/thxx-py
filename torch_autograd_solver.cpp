#include <torch/torch.h>

#include <iostream>

using at::Tensor;

// forked from pytorch 0.5
// https://github.com/sethah/pytorch/blob/81b61db9219ffeb8fc0c8ab3abe0f0b5a7edf4f4/tools/autograd/templates/Functions.cpp#L1514
// http://eprints.maths.ox.ac.uk/1079/1/NA-08-01.pdf
Tensor symeig_backward(
    Tensor& gx,
    const Tensor& grad_loss_wrt_eigenvalues,
    const Tensor& grad_loss_wrt_eigenvectors,
    const Tensor& x,
    const Tensor& eigenvalues,
    const Tensor& eigenvectors,
    bool upper) {
    auto vt = eigenvectors.t();

    if (!gx.defined()) {
        gx = at::zeros_like(x);
    }
    if (grad_loss_wrt_eigenvectors.defined()) {
        Tensor F = eigenvalues.unsqueeze(0).expand_as(x).clone();
        F.sub_(at::unsqueeze(eigenvalues, 1));
        F.diagonal().fill_(INFINITY);
        F.pow_(-1);
        F.mul_(vt.mm(grad_loss_wrt_eigenvectors));
        gx.add_(eigenvectors.mm(F.mm(vt)));
    }
    if (grad_loss_wrt_eigenvalues.defined()) {
        gx.add_((eigenvectors * grad_loss_wrt_eigenvalues).mm(vt));
    }
    if (upper) {
        auto gxu = at::triu(gx.t(), 1);
        gx.triu_().add_(gxu);
    } else {
        auto gxl = at::tril(gx.t(), -1);
        gx.tril_().add_(gxl);
    }
    return gx;
}

void basic_symeig_backward(
    const Tensor& grad_loss_wrt_eigenvalue,
    const Tensor& grad_loss_wrt_eigenvector,
    Tensor& grad_loss_wrt_input, // result
    const Tensor& eigenvalue, const Tensor& eigenvector,
    const Tensor& input,
    bool upper)
{
    auto& result = grad_loss_wrt_input;
    if (!result.defined()) {
        result = at::zeros_like(input);
    }
    // accumulate two gradients if backproped
    auto vt = eigenvector.t();
    if (grad_loss_wrt_eigenvector.defined()) {
        Tensor F = eigenvalue.unsqueeze(0).expand_as(input).clone();
        F.sub_(at::unsqueeze(eigenvalue, 1));
        F.diagonal().fill_(INFINITY);
        F.pow_(-1);
        F.mul_(vt.mm(grad_loss_wrt_eigenvector));
        result.add_(eigenvector.mm(F.mm(vt)));
    }
    if (grad_loss_wrt_eigenvalue.defined()) {
        result.add_((eigenvector * grad_loss_wrt_eigenvalue).mm(vt));
    }

    if (upper) {
        result = at::triu(result) + at::triu(result.t(), 1);
    } else {
        result = at::tril(result) + at::tril(result.t(), -1);
    }
}


void batch_symeig_backward(
    const Tensor& grad_loss_wrt_eigenvalue,
    const Tensor& grad_loss_wrt_eigenvector,
    Tensor& grad_loss_wrt_input, // result
    const Tensor& eigenvalue, const Tensor& eigenvector,
    const Tensor& input,
    bool upper)
{
    auto batch_size = input.size(0);
    if (!grad_loss_wrt_input.defined()) {
        grad_loss_wrt_input = at::zeros_like(input);
    }
#pragma omp for
    for (int64_t i = 0; i < batch_size; ++i) {

    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("symeig_backward", &symeig_backward, "basic symeig backward");
    m.def("basic_symeig_backward", &basic_symeig_backward, "basic symeig backward");
    m.def("batch_symeig_backward", &batch_symeig_backward, "batch symeig backward");
}
