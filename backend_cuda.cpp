/// TODO: support double?

#include <torch/torch.h>
#include <THC/THC.h>
//#undef NDEBUG

#include <cstdio>
#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverDn.h>


namespace thxx
{
    template<int success = CUSOLVER_STATUS_SUCCESS, class T, class Status>
    std::unique_ptr<T, Status(*)(T*)> unique_allocate(Status(allocator)(T**),  Status(deleter)(T*))
    {
        T* ptr;
        auto stat = allocator(&ptr);
        AT_CHECK(stat == success);
        return {ptr, deleter};
    }

    template <class T>
    using cuda_ptr = std::unique_ptr<T, decltype(&cudaFree)>;

    template <class T>
    cuda_ptr<T> unique_cuda_allocate(size_t len)
    {
        T* ptr;
        auto stat = cudaMalloc(&ptr, sizeof(T) * len);
        AT_CHECK(stat == cudaSuccess);
        return {ptr, cudaFree};
    }

    namespace cublas
    {
        cublasHandle_t getCurrentCUDABlasHandle()
        {
            return THCState_getCurrentBlasHandle(at::globalContext().getTHCState());
        }

        static const char *cudaGetErrorEnum(cublasStatus_t error)
        {
            switch (error)
            {
            case CUBLAS_STATUS_SUCCESS:
                return "CUBLAS_STATUS_SUCCESS";

            case CUBLAS_STATUS_NOT_INITIALIZED:
                return "CUBLAS_STATUS_NOT_INITIALIZED";

            case CUBLAS_STATUS_ALLOC_FAILED:
                return "CUBLAS_STATUS_ALLOC_FAILED";

            case CUBLAS_STATUS_INVALID_VALUE:
                return "CUBLAS_STATUS_INVALID_VALUE";

            case CUBLAS_STATUS_ARCH_MISMATCH:
                return "CUBLAS_STATUS_ARCH_MISMATCH";

            case CUBLAS_STATUS_MAPPING_ERROR:
                return "CUBLAS_STATUS_MAPPING_ERROR";

            case CUBLAS_STATUS_EXECUTION_FAILED:
                return "CUBLAS_STATUS_EXECUTION_FAILED";

            case CUBLAS_STATUS_INTERNAL_ERROR:
                return "CUBLAS_STATUS_INTERNAL_ERROR";

            case CUBLAS_STATUS_NOT_SUPPORTED:
                return "CUBLAS_STATUS_NOT_SUPPORTED";

            case CUBLAS_STATUS_LICENSE_ERROR:
                return "CUBLAS_STATUS_LICENSE_ERROR";
            }

            return "<unknown>";
        }

        cuda_ptr<void*> to_batch_pointers(at::Tensor a)
        {
            const auto batch_size = a.size(0);
            std::vector<void*> a_pointers;
            a_pointers.reserve(batch_size);
            for (int i = 0; i < batch_size; ++i)
            {
                a_pointers.push_back(a.select(0, i).data_ptr());
            }
            auto dev_a_ptrs = unique_cuda_allocate<void*>(batch_size);
            auto status_memcpy = cudaMemcpy(dev_a_ptrs.get(), a_pointers.data(),
                                            sizeof(void*) * batch_size, cudaMemcpyHostToDevice);
            AT_CHECK(cudaSuccess == status_memcpy);
            return std::move(dev_a_ptrs);
        }

        void test_cucomplex()
        {
            static_assert(sizeof(cuComplex) == sizeof(float[2]), "unexpected cuComplex size");
            float _A[2] = {1.0, 1.0};
            cuComplex* _a = (cuComplex*) &_A;
            AT_CHECK(_a->x == 1.0);
            AT_CHECK(_a->y == 1.0);
        }

        at::Tensor complex_mm(const at::Tensor& a, const at::Tensor& b)
        {
            // TODO optional arg
            at::Tensor c={};
            AT_CHECK(a.dtype() == at::kFloat || a.dtype() == at::kDouble, "only float and double are supported");
            AT_CHECK(b.dtype() == a.dtype(), "type mismatch between a and b");

            AT_CHECK(a.dim() == 3, "3-dim complex matrix is supported but a.dim() == ", a.dim());
            AT_CHECK(a.size(2) == 2 && a.stride(2) == 1,
                     "complex matrix a should be a.size(2) == 2 but ", a.size(2));

            AT_CHECK(b.dim() == 3, "3-dim complex matrix is supported but b.dim() == ", b.dim());
            AT_CHECK(b.size(2) == 2 && b.stride(2) == 1,
                     "complex matrix b should be a.size(2) == 2 but ", b.size(2));

            AT_CHECK(a.size(1) == b.size(0), "complex matrix is not matched:",
                     "a.size(1) {", a.size(1), "} != b.size(0) {", b.size(0), "}");
            AT_CHECK(a.is_cuda() && b.is_cuda(), "device is not matched");

            if (!c.defined())
            {
                c = at::empty({a.size(0), b.size(1), 2}, a.type());
            }

            // NOTE: cublas only supports fortran order (transposed A x B -> B^T x A^T)
            const auto transa = a.stride(1) == 2;
            const auto transb = b.stride(1) == 2;
            if (a.dtype() == at::kFloat)
            {
                const cuComplex alpha = {1.0, 0.0};
                const cuComplex beta = {0.0, 0.0};
                auto status = cublasCgemm(
                    getCurrentCUDABlasHandle(),
                    transb ? CUBLAS_OP_N : CUBLAS_OP_T,
                    transa ? CUBLAS_OP_N : CUBLAS_OP_T,
                    b.size(1), a.size(0), b.size(0),
                    &alpha,
                    (const cuComplex*) b.data_ptr(), b.stride(transb ? 0 : 1) / 2,
                    (const cuComplex*) a.data_ptr(), a.stride(transa ? 0 : 1) / 2,
                    &beta,
                    (cuComplex*) c.data_ptr(), c.stride(0) / 2
                    );
                AT_CHECK(status == CUBLAS_STATUS_SUCCESS, cudaGetErrorEnum(status));
            }
            else if (a.dtype() == at::kDouble)
            {
                const cuDoubleComplex alpha = {1.0, 0.0};
                const cuDoubleComplex beta = {0.0, 0.0};
                auto status = cublasZgemm(
                    getCurrentCUDABlasHandle(),
                    transb ? CUBLAS_OP_N : CUBLAS_OP_T,
                    transa ? CUBLAS_OP_N : CUBLAS_OP_T,
                    b.size(1), a.size(0), b.size(0),
                    &alpha,
                    (const cuDoubleComplex*) b.data_ptr(), b.stride(transb ? 0 : 1) / 2,
                    (const cuDoubleComplex*) a.data_ptr(), a.stride(transa ? 0 : 1) / 2,
                    &beta,
                    (cuDoubleComplex*) c.data_ptr(), c.stride(0) / 2
                    );
                AT_CHECK(status == CUBLAS_STATUS_SUCCESS, cudaGetErrorEnum(status));
            }
            else
            {
                AT_CHECK(false);
            }
            return c;
        }

        at::Tensor batch_complex_mm(const at::Tensor& a, const at::Tensor& b)
        {
            AT_CHECK(a.dtype() == at::kFloat || a.dtype() == at::kDouble, "only float and double are supported");
            AT_CHECK(a.dim() == 4, "3-dim batch complex matrix is supported but a.dim() == ", a.dim());
            AT_CHECK(a.size(3) == 2 && a.stride(3) == 1,
                     "complex matrix a should be a.size(3) == 2 but ", a.size(3));

            AT_CHECK(b.dim() == 4, "4-dim batch complex matrix is supported but b.dim() == ", b.dim());
            AT_CHECK(b.size(3) == 2 && b.stride(3) == 1,
                     "complex matrix b should be a.size(3) == 2 but ", b.size(3));

            AT_CHECK(b.dtype() == a.dtype(), "type mismatch between a and b");
            AT_CHECK(a.size(0) == b.size(0), "batch size is not matched:",
                     "a.size(0) {", a.size(0), "} != b.size(0) {", b.size(0), "}");
            AT_CHECK(a.size(2) == b.size(1), "complex matrix is not matched:",
                     "a.size(2) {", a.size(2), "} != b.size(1) {", b.size(1), "}");
            AT_CHECK(a.is_cuda(), "a is not cuda");
            AT_CHECK(b.is_cuda(), "b is not cuda");

            const auto batch_size = a.size(0);
            auto c = at::empty({batch_size, a.size(1), b.size(2), 2}, a.type());

            // NOTE: cublas only supports fortran order (transposed A x B -> B^T x A^T)
            const auto transa = a.stride(2) == 2;
            const auto transb = b.stride(2) == 2;
            auto a_ptrs = to_batch_pointers(a);
            auto b_ptrs = to_batch_pointers(b);
            auto c_ptrs = to_batch_pointers(c);

            if (a.dtype() == at::kFloat)
            {
                const cuComplex alpha = {1.0, 0.0};
                const cuComplex beta = {0.0, 0.0};
                auto status = cublasCgemmBatched(
                    getCurrentCUDABlasHandle(),
                    transb ? CUBLAS_OP_N : CUBLAS_OP_T,
                    transa ? CUBLAS_OP_N : CUBLAS_OP_T,
                    b.size(2), a.size(1), b.size(1),
                    &alpha,
                    const_cast<const cuComplex**>(reinterpret_cast<cuComplex**>(b_ptrs.get())),
                    // b.data_ptr(),
                    b.stride(transb ? 1 : 2) / 2,
                    const_cast<const cuComplex**>(reinterpret_cast<cuComplex**>(a_ptrs.get())),
                    a.stride(transa ? 1 : 2) / 2,
                    &beta,
                    reinterpret_cast<cuComplex**>(c_ptrs.get()),
                    c.stride(1) / 2,
                    batch_size
                    );
                AT_CHECK(status == CUBLAS_STATUS_SUCCESS, cudaGetErrorEnum(status));
            }
            else if (a.dtype() == at::kDouble)
            {
                const cuDoubleComplex alpha = {1.0, 0.0};
                const cuDoubleComplex beta = {0.0, 0.0};
                auto status = cublasZgemmBatched(
                    getCurrentCUDABlasHandle(),
                    transb ? CUBLAS_OP_N : CUBLAS_OP_T,
                    transa ? CUBLAS_OP_N : CUBLAS_OP_T,
                    b.size(2), a.size(1), b.size(1),
                    &alpha,
                    const_cast<const cuDoubleComplex**>(reinterpret_cast<cuDoubleComplex**>(b_ptrs.get())),
                    // b.data_ptr(),
                    b.stride(transb ? 1 : 2) / 2,
                    const_cast<const cuDoubleComplex**>(reinterpret_cast<cuDoubleComplex**>(a_ptrs.get())),
                    a.stride(transa ? 1 : 2) / 2,
                    &beta,
                    reinterpret_cast<cuDoubleComplex**>(c_ptrs.get()),
                    c.stride(1) / 2,
                    batch_size
                    );
                AT_CHECK(status == CUBLAS_STATUS_SUCCESS, cudaGetErrorEnum(status));
            }
            else
            {
                AT_CHECK(false);
            }
            return c;
        }

        at::Tensor batch_matinv(const at::Tensor& a)
        {
            AT_CHECK(a.is_cuda(), "only cuda tensor is supported");
            AT_CHECK(a.dim() == 3, "batch matrix should be a.dim() == 3 but ", a.dim());
            AT_CHECK(a.dtype() == at::kFloat || a.dtype() == at::kDouble, "only float or double is supported");
            AT_CHECK(a.size(1) == a.size(2), "a is not square");

            // TODO wrap getrf/getri inv for large pinv
            const auto batch_size = a.size(0);
            const auto m = a.size(1);
            AT_CHECK(m <= 32, "matrix row should be <= 32");
            const auto n = a.size(2);
            AT_CHECK(n == m, "should be col == row");
            const auto trans = a.stride(2) == 1;
            const auto lda = a.stride(trans ? 1 : 2);
            auto inv = at::empty_like(a);
            auto lda_inv = inv.stride(1);
            auto info_ptr = unique_cuda_allocate<int>(batch_size);
            auto dev_a_ptrs = to_batch_pointers(a);
            auto dev_inv_ptrs = to_batch_pointers(inv);

            if (a.dtype() == at::kFloat)
            {
                auto status = cublasSmatinvBatched(
                    getCurrentCUDABlasHandle(),
                    n,
                    const_cast<const float**>(reinterpret_cast<float**>(dev_a_ptrs.get())),
                    lda,
                    reinterpret_cast<float**>(dev_inv_ptrs.get()),
                    lda_inv,
                    info_ptr.get(),
                    batch_size);
                AT_CHECK(status == CUBLAS_STATUS_SUCCESS);
            }
            else if (a.dtype() == at::kDouble)
            {
                auto status = cublasDmatinvBatched(
                    getCurrentCUDABlasHandle(),
                    n,
                    const_cast<const double**>(reinterpret_cast<double**>(dev_a_ptrs.get())),
                    lda,
                    reinterpret_cast<double**>(dev_inv_ptrs.get()),
                    lda_inv,
                    info_ptr.get(),
                    batch_size);
                AT_CHECK(status == CUBLAS_STATUS_SUCCESS);
            }
            else
            {
                AT_CHECK(false);
            }
            return trans ? inv : inv.transpose(1, 2);
        }

        at::Tensor batch_complex_matinv(const at::Tensor& a)
        {
            AT_CHECK(a.is_cuda(), "only cuda tensor is supported");
            AT_CHECK(a.dim() == 4, "batch complex matrix should be a.dim() == 4 but ", a.dim());
            AT_CHECK(a.dtype() == at::kFloat || a.dtype() == at::kDouble, "only float or double is supported");
            AT_CHECK(a.size(1) == a.size(2), "a is not square");
            AT_CHECK(a.size(3) == 2, "a.size(3) should be 2 but ", a.size(3));
            AT_CHECK(a.stride(3) == 1, "a.stride(3) should be 1 but ", a.stride(3))

            // TODO wrap getrf/getri inv for large pinv
            const auto batch_size = a.size(0);
            const auto m = a.size(1);
            AT_CHECK(m <= 32, "matrix row should be <= 32");
            const auto n = a.size(2);
            const auto trans = a.stride(2) == 2;
            const auto lda = a.stride(trans ? 1 : 2) / 2;
            auto inv = at::empty_like(a);
            auto lda_inv = inv.stride(1) / 2;
            auto info_ptr = unique_cuda_allocate<int>(batch_size);
            auto dev_a_ptrs = to_batch_pointers(a);
            auto dev_inv_ptrs = to_batch_pointers(inv);

            if (a.dtype() == at::kFloat)
            {
                auto status = cublasCmatinvBatched(
                    getCurrentCUDABlasHandle(),
                    n,
                    const_cast<const cuComplex**>(reinterpret_cast<cuComplex**>(dev_a_ptrs.get())),
                    lda,
                    reinterpret_cast<cuComplex**>(dev_inv_ptrs.get()),
                    lda_inv,
                    info_ptr.get(),
                    batch_size);
                AT_CHECK(status == CUBLAS_STATUS_SUCCESS);
            }
            else if (a.dtype() == at::kDouble)
            {
                auto status = cublasZmatinvBatched(
                    getCurrentCUDABlasHandle(),
                    n,
                    const_cast<const cuDoubleComplex**>(reinterpret_cast<cuDoubleComplex**>(dev_a_ptrs.get())),
                    lda,
                    reinterpret_cast<cuDoubleComplex**>(dev_inv_ptrs.get()),
                    lda_inv,
                    info_ptr.get(),
                    batch_size);
                AT_CHECK(status == CUBLAS_STATUS_SUCCESS);
            }
            else
            {
                AT_CHECK(false);
            }
            return trans ? inv : inv.transpose(1, 2);
        }


        // batch_getrf
        // batch_getrs
        // batch_getri
        // batch_inv from https://github.com/chainer/chainer/blob/v4.5.0/chainer/functions/math/inv.py#L129
    }

    namespace cusolver
    {
        void check_jacobi(int* info_ptr, int batch_size)
        {
            std::vector<int> hinfo(batch_size);
            auto status_memcpy = cudaMemcpy(hinfo.data(), info_ptr, sizeof(int) * batch_size, cudaMemcpyDeviceToHost);
            AT_CHECK(cudaSuccess == status_memcpy);

            for(int i = 0 ; i < batch_size; ++i)
            {
                if ( 0 == hinfo[i] )
                {
                    continue;
                }
                else if ( 0 > hinfo[i] )
                {
                    printf("Error: %d-th parameter is wrong \n", -hinfo[i]);
                    AT_CHECK(false);
                }
                else
                {
                    printf("WARNING: matrix %d, info = %d : Jacobi method does not converge \n", i, hinfo[i] );
                }
            }

        }

        // solve AV = wV  a.k.a. syevj, where A (batch, m, m), V (batch, m, m), w (batch, m)
        // see also https://docs.nvidia.com/cuda/cusolver/index.html#batchsyevj-example1
        std::tuple<at::Tensor, at::Tensor> batch_symmetric_eigenvalue_solve(
            at::Tensor a, bool in_place=false, bool use_lower=true,
            double tol=1e-7, int max_sweeps=15, bool sort_eig=false)
        {
            // TODO use singleton handler instead of ondemand handle
            // TODO check cutorch or ATen does not handle cusolver
            // https://github.com/torch/cutorch/blob/master/lib/THC/THCGeneral.h.in
            // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/CUDAContext.h

            AT_CHECK(a.is_cuda(), "only cuda tensor is supported");
            AT_CHECK(a.dtype() == at::kFloat || a.dtype() == at::kDouble, "only float/double is supported");
            AT_CHECK(a.dim() == 3, "3-dim batch matrix is supported");
            AT_CHECK(a.size(1) == a.size(2), "only symmetric matrix is supported");
            // initialization
            auto handle_ptr = unique_allocate(cusolverDnCreate, cusolverDnDestroy);
            // TODO use non blocking stream?
            auto batch_size = a.size(0);
            auto m = a.size(2);
            AT_CHECK(m <= 32, "matrix row/col should be <= 32");
            auto w = at::empty({a.size(0), a.size(2)}, a.type());
            auto V = in_place ? a.contiguous() : a.clone();
            auto lda = V.stride(1);
            auto d_V = V.data_ptr();
            auto d_W = w.data_ptr();
            auto uplo = use_lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

            // configure
            auto param_ptr = unique_allocate(cusolverDnCreateSyevjInfo, cusolverDnDestroySyevjInfo);
            auto syevj_params = param_ptr.get();
            /* default value of tolerance is machine zero */
            auto status = cusolverDnXsyevjSetTolerance(syevj_params, tol);
            AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
            /* default value of max. sweeps is 100 */
            status = cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps);
            AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
            /* disable sorting */
            status = cusolverDnXsyevjSetSortEig(syevj_params, sort_eig);
            AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
            auto info_ptr = unique_cuda_allocate<int>(batch_size);

            // query working space of syevjBatched
            if (a.dtype() == at::kFloat)
            {
                int lwork;
                status = cusolverDnSsyevjBatched_bufferSize(
                    handle_ptr.get(),
                    CUSOLVER_EIG_MODE_VECTOR,
                    uplo,
                    m,
                    (float*) d_V,
                    lda,
                    (float*) d_W,
                    &lwork,
                    syevj_params,
                    batch_size
                    );
                AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
                auto work_ptr = unique_cuda_allocate<float>(lwork);
                status = cusolverDnSsyevjBatched(
                    handle_ptr.get(),
                    CUSOLVER_EIG_MODE_VECTOR,
                    uplo,
                    m,
                    (float*) d_V,
                    lda,
                    (float*) d_W,
                    work_ptr.get(),
                    lwork,
                    info_ptr.get(),
                    syevj_params,
                    batch_size
                    );
                AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
            }
            else if (a.dtype() == at::kDouble)
            {
                int lwork;
                status = cusolverDnDsyevjBatched_bufferSize(
                    handle_ptr.get(),
                    CUSOLVER_EIG_MODE_VECTOR,
                    uplo,
                    m,
                    (double*) d_V,
                    lda,
                    (double*) d_W,
                    &lwork,
                    syevj_params,
                    batch_size
                    );
                AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
                auto work_ptr = unique_cuda_allocate<double>(lwork);
                status = cusolverDnDsyevjBatched(
                    handle_ptr.get(),
                    CUSOLVER_EIG_MODE_VECTOR,
                    uplo,
                    m,
                    (double*) d_V,
                    lda,
                    (double*) d_W,
                    work_ptr.get(),
                    lwork,
                    info_ptr.get(),
                    syevj_params,
                    batch_size
                    );
                AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
            }
            else
            {
                AT_CHECK(false);
            }
            check_jacobi(info_ptr.get(), batch_size);
            return std::make_tuple(w, V);
        }

        std::tuple<at::Tensor, at::Tensor> batch_complex_symmetric_eigenvalue_solve(
            at::Tensor a, bool in_place=false, bool use_lower=true,
            double tol=1e-7, int max_sweeps=15, bool sort_eig=false)
        {
            // TODO use singleton handler instead of ondemand handle
            // TODO check cutorch or ATen does not handle cusolver
            // https://github.com/torch/cutorch/blob/master/lib/THC/THCGeneral.h.in
            // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/CUDAContext.h

            AT_CHECK(a.is_cuda(), "only cuda tensor is supported");
            AT_CHECK(a.dtype() == at::kFloat || a.dtype() == at::kDouble, "only float/double is supported");
            AT_CHECK(a.dim() == 4, "only 4-dim batch complex matrix is supported");
            AT_CHECK(a.size(3) == 2, "complex dim should be a.size(3) == 2 but ", a.size(3));
            AT_CHECK(a.stride(3) == 1, "complex dim=3 is not contiguous");
            AT_CHECK(a.size(1) == a.size(2), "only symmetric matrix is supported");
            // initialization
            auto handle_ptr = unique_allocate(cusolverDnCreate, cusolverDnDestroy);
            // TODO use non blocking stream?
            auto batch_size = a.size(0);
            auto m = a.size(2);
            AT_CHECK(m <= 32, "matrix row/col should be <= 32");
            auto w = at::empty({m, m}, a.type());
            auto V = in_place ? a.contiguous() : a.clone();
            auto lda = V.stride(1) / 2;
            auto d_V = V.data_ptr();
            auto d_W = w.data_ptr();
            auto uplo = use_lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

            // configure
            auto param_ptr = unique_allocate(cusolverDnCreateSyevjInfo, cusolverDnDestroySyevjInfo);
            auto syevj_params = param_ptr.get();
            /* default value of tolerance is machine zero */
            auto status = cusolverDnXsyevjSetTolerance(syevj_params, tol);
            AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
            /* default value of max. sweeps is 100 */
            status = cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps);
            AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
            /* disable sorting */
            status = cusolverDnXsyevjSetSortEig(syevj_params, sort_eig);
            AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
            auto info_ptr = unique_cuda_allocate<int>(batch_size);

            /* FIXME
            // query working space of syevjBatched
            if (a.dtype() == at::kFloat)
            {
                int lwork;
                status = cusolverDnCgesvdjBatched_bufferSize(
                    handle_ptr.get(),
                    CUSOLVER_EIG_MODE_VECTOR,
                    uplo,
                    m,
                    (cuComplex*) d_V,
                    lda,
                    (float*) d_W,
                    ldv,
                    &lwork,
                    syevj_params,
                    batch_size
                    );
                AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
                auto work_ptr = unique_cuda_allocate<cuComplex>(lwork);
                status = cusolverDnCgesvdjBatched(
                    handle_ptr.get(),
                    CUSOLVER_EIG_MODE_VECTOR,
                    uplo,
                    m,
                    (cuComplex*) d_V,
                    lda,
                    (float*) d_W,
                    work_ptr.get(),
                    lwork,
                    info_ptr.get(),
                    syevj_params,
                    batch_size
                    );
                AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
            }
            else if (a.dtype() == at::kDouble)
            {
                int lwork;
                status = cusolverDnZgesvdjBatched_bufferSize(
                    handle_ptr.get(),
                    CUSOLVER_EIG_MODE_VECTOR,
                    uplo,
                    m,
                    (cuDoubleComplex*) d_V,
                    lda,
                    (double*) d_W,
                    &lwork,
                    syevj_params,
                    batch_size
                    );
                AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
                auto work_ptr = unique_cuda_allocate<cuDoubleComplex>(lwork);
                status = cusolverDnZgesvdjBatched(
                    handle_ptr.get(),
                    CUSOLVER_EIG_MODE_VECTOR,
                    uplo,
                    m,
                    (cuDoubleComplex*) d_V,
                    lda,
                    (double*) d_W,
                    work_ptr.get(),
                    lwork,
                    info_ptr.get(),
                    syevj_params,
                    batch_size
                    );
                AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
            }
            else
            {
                AT_CHECK(false);
            }
            */
            check_jacobi(info_ptr.get(), batch_size);
            return std::make_tuple(w, V);
        }


        // solve AV = wBV  a.k.a. syevj, where A (m, m), B (m, m), V (m, m), w (m)
        // see also https://docs.nvidia.com/cuda/cusolver/index.html#sygvd-example1
        std::tuple<at::Tensor, at::Tensor, at::Tensor>
        generalized_symmetric_eigenvalue_solve(
            at::Tensor a, bool in_place_a, at::Tensor b, bool in_place_b,
            bool use_upper, bool use_jacob, double tol=1e-7, int max_sweeps=100
            ) {
            AT_CHECK(a.is_cuda(), "only cuda tensor for a is supported");
            AT_CHECK(a.dtype() == at::kFloat, "only float for a is supported");
            AT_CHECK(b.is_cuda(), "only cuda tensor for b is supported");
            AT_CHECK(b.dtype() == at::kFloat, "only float for b is supported");

            auto handle_ptr = unique_allocate(cusolverDnCreate, cusolverDnDestroy);

            // step 1: copy A and B to device
            auto m = a.size(0);
            // NOTE: V will be overwritten from A to orthonormal eigenvectors
            auto V = in_place_a ? a.contiguous() : a.clone();
            auto d_A = V.data<float>();
            auto lda = V.stride(0);
            // NOTE: B_ will be overwritten from B to LU-Choresky factorization
            auto B_LU = in_place_b ? b.contiguous() : b.clone();
            auto d_B = B_LU.data<float>();
            auto ldb = B_LU.stride(0);
            // NOTE: w will be sorted
            auto w = at::empty({m}, a.type());
            auto d_W = w.data<float>();
            auto info_ptr = unique_cuda_allocate<int>(1);

            cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1; // A V = w B V
            cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors
            cublasFillMode_t uplo = use_upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

            if (use_jacob)
            {
                syevjInfo_t syevj_params;
                auto status_param = cusolverDnCreateSyevjInfo(&syevj_params);
                AT_CHECK(CUSOLVER_STATUS_SUCCESS == status_param);
                status_param = cusolverDnXsyevjSetTolerance(syevj_params, tol);
                AT_CHECK(CUSOLVER_STATUS_SUCCESS == status_param);
                status_param = cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps);
                AT_CHECK(CUSOLVER_STATUS_SUCCESS == status_param);

                int lwork;
                auto status_buffer = cusolverDnSsygvj_bufferSize(
                    handle_ptr.get(),
                    itype,
                    jobz,
                    uplo,
                    m,
                    d_A,
                    lda,
                    d_B,
                    ldb,
                    d_W,
                    &lwork,
                    syevj_params);
                AT_CHECK(status_buffer == CUSOLVER_STATUS_SUCCESS);
                auto work_ptr = unique_cuda_allocate<float>(lwork);
                auto status_compute = cusolverDnSsygvj(
                    handle_ptr.get(),
                    itype,
                    jobz,
                    uplo,
                    m,
                    d_A,
                    lda,
                    d_B,
                    ldb,
                    d_W,
                    work_ptr.get(),
                    lwork,
                    info_ptr.get(),
                    syevj_params);
                AT_CHECK(CUSOLVER_STATUS_SUCCESS == status_compute);
            }
            else
            {
                int lwork;
                auto cusolver_status = cusolverDnSsygvd_bufferSize(
                    handle_ptr.get(),
                    itype,
                    jobz,
                    uplo,
                    m,
                    d_A,
                    lda,
                    d_B,
                    ldb,
                    d_W,
                    &lwork);
                AT_CHECK (cusolver_status == CUSOLVER_STATUS_SUCCESS);
                auto work_ptr = unique_cuda_allocate<float>(lwork);
                cusolver_status = cusolverDnSsygvd(
                    handle_ptr.get(),
                    itype,
                    jobz,
                    uplo,
                    m,
                    d_A,
                    lda,
                    d_B,
                    ldb,
                    d_W,
                    work_ptr.get(),
                    lwork,
                    info_ptr.get());
                AT_CHECK(CUSOLVER_STATUS_SUCCESS == cusolver_status);
            }
            check_jacobi(info_ptr.get(), 1);
            return std::make_tuple(w, V, B_LU);
        }

        // solve U S V = svd(A)  a.k.a. syevj, where A (b, m, n), U (b, m, m), S (b, min(m, n)), V (b, n, n)
        // see also https://docs.nvidia.com/cuda/cusolver/index.html#batchgesvdj-example1
        std::tuple<at::Tensor, at::Tensor, at::Tensor>
        batch_svd(at::Tensor a, bool is_sort, double tol=1e-7, int max_sweeps=100)
        {
            AT_CHECK(a.is_cuda(), "only cuda tensor is supported");
            AT_CHECK(a.dtype() == at::kFloat, "only float is supported");

            auto handle_ptr = unique_allocate(cusolverDnCreate, cusolverDnDestroy);
            const auto A = a.contiguous();
            const auto batch_size = A.size(0);
            const auto m = A.size(1);
            AT_CHECK(m <= 32, "matrix row should be <= 32");
            const auto n = A.size(2);
            AT_CHECK(n <= 32, "matrix col should be <= 32");
            const auto lda = m;
            const auto d_A = A.data<float>();
            const auto minmn = std::min(m, n);
            auto s = at::empty({batch_size, minmn}, a.type());
            auto d_s = s.data<float>();
            auto U = at::empty({batch_size, m, m}, a.type());
            const auto d_U = U.data<float>();
            const auto ldu = m;
            auto V = at::empty({batch_size, n, n}, a.type());
            const auto d_V = V.data<float>();
            const auto ldv = n;

            auto params = unique_allocate(cusolverDnCreateGesvdjInfo, cusolverDnDestroyGesvdjInfo);
            auto status = cusolverDnXgesvdjSetTolerance(params.get(), tol);
            AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
            status = cusolverDnXgesvdjSetMaxSweeps(params.get(), max_sweeps);
            AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
            status = cusolverDnXgesvdjSetSortEig(params.get(), is_sort);
            AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);

            auto jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors
            int lwork;
            auto status_buffer = cusolverDnSgesvdjBatched_bufferSize(
                handle_ptr.get(),
                jobz,
                m,
                n,
                d_A,
                lda,
                d_s,
                d_U,
                ldu,
                d_V,
                ldv,
                &lwork,
                params.get(),
                batch_size);
            AT_CHECK(CUSOLVER_STATUS_SUCCESS == status_buffer);
            auto work_ptr = unique_cuda_allocate<float>(lwork);
            auto info_ptr = unique_cuda_allocate<int>(batch_size);
            status = cusolverDnSgesvdjBatched(
                handle_ptr.get(),
                jobz,
                m,
                n,
                d_A,
                lda,
                d_s,
                d_U,
                ldu,
                d_V,
                ldv,
                work_ptr.get(),
                lwork,
                info_ptr.get(),
                params.get(),
                batch_size
                );
            AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
            check_jacobi(info_ptr.get(), batch_size);
            return std::make_tuple(U, s, V);
        }

        // batch_potrf

        // batch_potrs

    } // namespace cusolver
} // namespace thxx

// generate wrappers
// FIXME do not use legacy preprocessor macro
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_eigh", &thxx::cusolver::batch_symmetric_eigenvalue_solve,
          "cusolver based batched eigh implementation");
    m.def("batch_complex_eigh", &thxx::cusolver::batch_complex_symmetric_eigenvalue_solve,
          "cusolver based batched complex eigh implementation");
    m.def("generalized_eigh", &thxx::cusolver::generalized_symmetric_eigenvalue_solve,
          "cusolver based generalized eigh implementation");
    m.def("batch_svd", &thxx::cusolver::batch_svd,
          "cusolver based batch svd implementation");
    m.def("batch_matinv", &thxx::cublas::batch_matinv,
          "cublas based batch matrix inverse implementation");
    m.def("batch_complex_matinv", &thxx::cublas::batch_complex_matinv,
          "cublas based batch matrix inverse implementation");
    m.def("complex_mm", &thxx::cublas::complex_mm,
          "cublas based complex matrix multiplication implementation");
    m.def("batch_complex_mm", &thxx::cublas::batch_complex_mm,
          "cublas based batch complex matrix multiplication implementation");
}
