#include <torch/torch.h>
#include <mkl.h>

namespace thxx
{
    namespace mkl
    {
        at::Tensor complex_mm(const at::Tensor& a, const at::Tensor& b)
        {
            AT_CHECK(a.dtype() == at::kFloat || a.dtype() == at::kDouble,
                     "only float and double are supported");
            AT_CHECK(a.dtype() == b.dtype(), "a.dtype() != b.dtype()");

            AT_CHECK(a.dim() == 3, "3-dim complex matrix is supported but a.dim() == ", a.dim());
            AT_CHECK(a.size(2) == 2, "complex matrix a should be a.size(2) == 2 but ", a.size(2));
            AT_CHECK(a.stride(2) == 1, "complex matrix a should be a.stride(2) == 1 but ", a.stride(2));

            AT_CHECK(b.dim() == 3, "3-dim complex matrix is supported but b.dim() == ", b.dim());
            AT_CHECK(b.size(2) == 2, "complex matrix b should be a.size(2) == 2 but ", b.size(2));
            AT_CHECK(b.stride(2) == 1, "complex matrix a should be b.stride(2) == 1 but ", b.stride(2));

            AT_CHECK(a.size(1) == b.size(0), "complex matrix is not matched:",
                     "a.size(1) {", a.size(1), "} != b.size(0) {", b.size(0), "}");
            AT_CHECK(!a.is_cuda() && !b.is_cuda(), "device is not matched");

            auto c = at::empty({a.size(0), b.size(1), 2}, a.type());

            const auto transa = a.stride(1) == 2;
            const auto transb = b.stride(1) == 2;
            auto ta = transa ? 'N' : 'T';
            auto tb = transb ? 'N' : 'T';
            int m = b.size(1);
            int n = a.size(0);
            int k = b.size(0);
            int ldb = b.stride(transb ? 0 : 1) / 2;
            int lda = a.stride(transa ? 0 : 1) / 2;
            int ldc = c.stride(0) / 2;

            if (a.dtype() == at::kFloat)
            {
                const float alpha[2] = {1.0, 0.0};
                const float beta[2] = {0.0, 0.0};
                cgemm(
                    &tb, &ta,
                    &m, &n, &k,
                    (MKL_Complex8*) &alpha,
                    (const MKL_Complex8*) b.data_ptr(), &ldb,
                    (const MKL_Complex8*) a.data_ptr(), &lda,
                    (MKL_Complex8*) &beta,
                    (MKL_Complex8*) c.data_ptr(), &ldc
                    );
            }
            else if (a.dtype() == at::kDouble)
            {
                const double alpha[2] = {1.0, 0.0};
                const double beta[2] = {0.0, 0.0};
                zgemm(
                    &tb, &ta,
                    &m, &n, &k,
                    (MKL_Complex16*) &alpha,
                    (const MKL_Complex16*) b.data_ptr(), &ldb,
                    (const MKL_Complex16*) a.data_ptr(), &lda,
                    (MKL_Complex16*) &beta,
                    (MKL_Complex16*) c.data_ptr(), &ldc
                    );
            }
            else
            {
                AT_CHECK(false);
            }
            return c;
        }

        at::Tensor batch_complex_mm(const at::Tensor& a, const at::Tensor& b)
        {
            AT_CHECK(a.dtype() == at::kFloat || a.dtype() == at::kDouble,
                     "only float and double are supported");
            AT_CHECK(a.dtype() == b.dtype(), "a.dtype() != b.dtype()");

            AT_CHECK(a.dim() == 4, "4-dim complex matrix is supported but a.dim() == ", a.dim());
            AT_CHECK(a.size(3) == 2, "complex matrix a should be a.size(3) == 2 but ", a.size(3));
            AT_CHECK(a.stride(3) == 1, "complex matrix a should be a.stride(3) == 1 but ", a.stride(3));

            AT_CHECK(b.dim() == 4, "4-dim complex matrix is supported but b.dim() == ", b.dim());
            AT_CHECK(b.size(3) == 2, "complex matrix a should be b.size(3) == 2 but ", b.size(3));
            AT_CHECK(b.stride(3) == 1, "complex matrix a should be b.stride(3) == 1 but ", b.stride(3));

            AT_CHECK(a.size(0) == b.size(0), "complex matrix is not matched:",
                     "a.size(0) {", a.size(0), "} != b.size(0) {", b.size(0), "}");
            AT_CHECK(a.size(2) == b.size(1), "complex matrix is not matched:",
                     "a.size(2) {", a.size(2), "} != b.size(1) {", b.size(1), "}");
            AT_CHECK(!a.is_cuda() && !b.is_cuda(), "device is not matched");

            int batch_size = a.size(0);
            auto c = at::zeros({batch_size, a.size(1), b.size(2), 2}, a.type());
            std::vector<void*> ap, bp, cp;

            ap.reserve(batch_size);
            bp.reserve(batch_size);
            cp.reserve(batch_size);

            const auto transa = a.stride(2) == 2;
            const auto transb = b.stride(2) == 2;
            const int sa = a.stride(transa ? 1 : 2) / 2;
            const int sb = b.stride(transb ? 1 : 2) / 2;
            const int sc = c.stride(1) / 2;
            const CBLAS_TRANSPOSE _ta = transa ? CblasNoTrans : CblasTrans; // 'N' : 'T';
            const CBLAS_TRANSPOSE _tb = transb ? CblasNoTrans : CblasTrans; // 'N' : 'T';

            for (int i = 0; i < batch_size; ++i)
            {
                ap.push_back(a.select(0, i).data_ptr());
                bp.push_back(b.select(0, i).data_ptr());
                cp.push_back(c.select(0, i).data_ptr());
            }

            int m = b.size(2);
            int n = a.size(1);
            int k = b.size(1);

            if (a.dtype() == at::kFloat)
            {
                float alpha[2] = {1.0, 0.0};
                float beta[2] = {0.0, 0.0};
                cblas_cgemm_batch(
                    CblasColMajor,
                    &_tb, &_ta,
                    &m, &n, &k,
                    &alpha,
                    (const void**) bp.data(), &sb,
                    (const void**) ap.data(), &sa,
                    &beta,
                    (void**) cp.data(), &sc,
                    1,
                    &batch_size
                    );
            }
            else if (a.dtype() == at::kDouble)
            {
                double alpha[2] = {1.0, 0.0};
                double beta[2] = {0.0, 0.0};
                cblas_zgemm_batch(
                    CblasColMajor,
                    &_tb, &_ta,
                    &m, &n, &k,
                    &alpha,
                    (const void**) bp.data(), &sb,
                    (const void**) ap.data(), &sa,
                    &beta,
                    (void**) cp.data(), &sc,
                    1,
                    &batch_size
                    );
            }
            else
            {
                AT_CHECK(false);
            }
            return c;
        }
    } // namespace mkl
} // namespace thxx


// generate wrappers
// FIXME do not use legacy preprocessor macro
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("complex_mm", &thxx::mkl::complex_mm,
          "MKL based complex matrix multiplication implementation");
    m.def("batch_complex_mm", &thxx::mkl::batch_complex_mm,
          "MKL based batch complex matrix multiplication implementation");
}
