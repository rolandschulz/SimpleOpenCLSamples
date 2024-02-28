/*
// Copyright (c) 2019-2024 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

// #include <CL/opencl.hpp>
#include <sycl.hpp>

#include <algorithm>
#include <chrono>
#include <sstream>
#include <string>
#include <random>
#include <vector>

#include "bfloat16.hpp"

#include <cute/tensor.hpp>
// #include "util.hpp"

using test_clock = std::chrono::high_resolution_clock;

bool identityData = false;
bool fixedData = false;
bool validate = false;
bool emulate = false;
bool wallclock = false;
int testIterations = 16;
float threshold = 0.01f;

std::string makeTestName(
    const std::string &func,
    size_t M, size_t N, size_t K)
{
    std::ostringstream ret;
    ret << func;
    ret << " (M=" << M << ", N=" << N << ", K=" << K << ")";
    return ret.str();
}

std::string makeTestName(
    const std::string &func,
    int tM, int tN, int tK,
    size_t M, size_t N, size_t K)
{
    std::ostringstream ret;
    ret << func;
    ret << "<tM:" << tM << ", tN:" << tN << ", tK:" << tK << ">";
    ret << " (M=" << M << ", N=" << N << ", K=" << K << ")";
    return ret.str();
}

std::string makeTestName(
    const std::string &func,
    int tM, int tN, int tK,
    int MM, int NN,
    size_t M, size_t N, size_t K)
{
    std::ostringstream ret;
    ret << func;
    ret << "<tM:" << tM << "x" << MM << ", tN:" << tN << "x" << NN << ", tK:" << tK << ">";
    ret << " (M=" << M << ", N=" << N << ", K=" << K << ")";
    return ret.str();
}

// static size_t findMinSubGroupSize(cl::Device& device)
// {
//     auto s = device.getInfo<CL_DEVICE_SUB_GROUP_SIZES_INTEL>();
//     auto it = std::min_element(std::begin(s), std::end(s));
//     if (it != std::end(s)) {
//         return *it;
//     }
//     return 0;
// }

template <typename T>
static void fill_matrix(std::vector<T>& M, size_t numRows, size_t numCols)
{
    if (identityData) {
        std::generate(std::begin(M), std::end(M), [&]{ return 1.0f; });
    } else if (fixedData) {
        for (size_t r = 0; r < numRows; r++) {
            for (size_t c = 0; c < numCols; c++) {
                M[r * numCols + c] = static_cast<float>(r + c);
            }
        }
    } else {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<float> dist(-1.0, 1.0);
        std::generate(std::begin(M), std::end(M), [&]{ return dist(rng); });
    }
}

template <typename T>
static void vnni_matrix(
    std::vector<T> &dst, const std::vector<T> &src,
    size_t numRows, size_t numCols, size_t factor)
{
    for (size_t r = 0; r < numRows / factor; r++) {
        for (size_t c = 0; c < numCols; c++) {
            for (size_t k = 0; k < factor; k++) {
                dst[r * numCols * factor + c * factor + k] =
                    src[(r * factor + k) * numCols + c];
            }
        }
    }
}

template <typename DstT, typename SrcT>
static void compute_reference(
    std::vector<DstT>& C,
    const std::vector<SrcT>& A, const std::vector<SrcT>& B,
    size_t M, size_t N, size_t K)
{
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            DstT sum = 0;
            for (size_t k = 0; k < K; k++) {
                sum = std::fma(static_cast<DstT>(A[m * K + k]),
                               static_cast<DstT>(B[k * N + n]), sum);
            }
            C[m * N + n] = sum;
        }
    }
}

template <typename T>
void check_results(
    size_t M,
    size_t N,
    const std::vector<T>& C,
    const std::vector<T>& C_ref)
{
    float err = 0.f;
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            auto index = m * N + n;
            auto localErr = std::fabs(C[index] - C_ref[index]) /
                            std::max(std::fabs(C[index]),
                                    std::fabs(C_ref[index]));
            err = std::max(localErr, err);
            if (localErr >= threshold) {
                std::cerr << "Error at m = " << m << ", n = " << n
                          << ": (local error " << localErr << "): Wanted "
                          << C_ref[index] << ", got " << C[index] << std::endl;
                // return;
            }
        }
    }
}

// static float hw_time(cl::Event& event)
// {
//     auto ns = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
//               event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
//     return ns / 1e9f;
// }

// static void go_naive(
//     cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
//     cl::Buffer& C, cl::Buffer& A, cl::Buffer& B,
//     size_t M, size_t N, size_t K,
//     const std::vector<float>& C_ref)
// {
//     printf("%80s: ", makeTestName(__FUNCTION__, M, N, K).c_str()); fflush(stdout);

//     cl::Kernel kernel{program, "bfloat16_naive"};
//     if (kernel() == nullptr) {
//         printf("unsupported.\n");
//     } else {
//         kernel.setArg(0, C);
//         kernel.setArg(1, A);
//         kernel.setArg(2, B);
//         kernel.setArg(3, static_cast<cl_int>(K));

//         queue.enqueueFillBuffer(C, 0, 0, C_ref.size());

//         float best = 999.0f;
//         for (int test = 0; test < testIterations; test++) {
//             cl::Event event;
//             auto start = test_clock::now();
//             queue.enqueueNDRangeKernel(kernel, cl::NullRange,
//                 cl::NDRange{N, M}, cl::NullRange, nullptr, &event);
//             queue.finish();
//             auto end = test_clock::now();
//             std::chrono::duration<float> sw_time = end - start;
//             auto elapsed = wallclock ? sw_time.count() : hw_time(event);
//             best = std::min(best, elapsed);
//         }
//         auto gops = 2.0 * M * N * K / best / 1e9;
//         printf("Best in %f seconds (%f gops)\n", best, gops);

//         if (validate) {
//             printf("Checking results... "); fflush(stdout);
//             std::vector<float> C_check(C_ref.size());
//             queue.enqueueReadBuffer(C, CL_TRUE, 0, C_check.size() * sizeof(C_check[0]), C_check.data());
//             check_results(M, N, C_check, C_ref);
//             printf(" done!\n");
//         }
//     }
// }

// template<int tM, int tN, int tK>
// static void go_dpas_rowmajor(
//     cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
//     cl::Buffer& C, cl::Buffer& A, cl::Buffer& B,
//     size_t M, size_t N, size_t K,
//     const std::vector<float>& C_ref)
// {
//     printf("%80s: ", makeTestName(__FUNCTION__, tM, tN, tK, M, N, K).c_str()); fflush(stdout);

//     std::string kernelName = "bfloat16_dpas_rowmajor";
//     kernelName += "_m" + std::to_string(tM);
//     kernelName += "_n" + std::to_string(tN);
//     cl::Kernel kernel{program, kernelName.c_str()};
//     if (kernel() == nullptr) {
//         printf("unsupported.\n");
//     } else {
//         kernel.setArg(0, C);
//         kernel.setArg(1, A);
//         kernel.setArg(2, B);
//         kernel.setArg(3, static_cast<cl_int>(K));

//         queue.enqueueFillBuffer(C, 0, 0, C_ref.size());

//         float best = 999.0f;
//         for (int test = 0; test < testIterations; test++) {
//             cl::Event event;
//             auto start = test_clock::now();
//             queue.enqueueNDRangeKernel(kernel, cl::NullRange,
//                 cl::NDRange{N, M/tM}, cl::NullRange, nullptr, &event);
//             queue.finish();
//             auto end = test_clock::now();
//             std::chrono::duration<float> sw_time = end - start;
//             auto elapsed = wallclock ? sw_time.count() : hw_time(event);
//             best = std::min(best, elapsed);
//         }
//         auto gops = 2.0 * M * N * K / best / 1e9;
//         printf("Best in %f seconds (%f gops)\n", best, gops);

//         if (validate) {
//             printf("Checking results... "); fflush(stdout);
//             std::vector<float> C_check(C_ref.size());
//             queue.enqueueReadBuffer(C, CL_TRUE, 0, C_check.size() * sizeof(C_check[0]), C_check.data());
//             check_results(M, N, C_check, C_ref);
//             printf(" done!\n");
//         }
//     }
// }

// template<int tM, int tN, int tK, int MM, int NN>
// static void go_dpas_rowmajor_tiled(
//     cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
//     cl::Buffer& C, cl::Buffer& A, cl::Buffer& B,
//     size_t M, size_t N, size_t K,
//     const std::vector<float>& C_ref)
// {
//     printf("%80s: ", makeTestName(__FUNCTION__, tM, tN, tK, MM, NN, M, N, K).c_str()); fflush(stdout);

//     std::string kernelName = "bfloat16_dpas_rowmajor_tiled";
//     kernelName += "_m" + std::to_string(tM);
//     kernelName += "_n" + std::to_string(tN);
//     kernelName += "_" + std::to_string(MM);
//     kernelName += "x" + std::to_string(NN);
//     cl::Kernel kernel{program, kernelName.c_str()};
//     if (kernel() == nullptr) {
//         printf("unsupported.\n");
//     } else if (tM * MM > M) {
//         printf("M is too small.\n");
//     } else if (tN * NN > N) {
//         printf("N is too small.\n");
//     } else {
//         kernel.setArg(0, C);
//         kernel.setArg(1, A);
//         kernel.setArg(2, B);
//         kernel.setArg(3, static_cast<cl_int>(K));

//         queue.enqueueFillBuffer(C, 0, 0, C_ref.size());

//         float best = 999.0f;
//         for (int test = 0; test < testIterations; test++) {
//             cl::Event event;
//             auto start = test_clock::now();
//             queue.enqueueNDRangeKernel(kernel, cl::NullRange,
//                 cl::NDRange{N/NN, M/tM/MM}, cl::NullRange, nullptr, &event);
//             queue.finish();
//             auto end = test_clock::now();
//             std::chrono::duration<float> sw_time = end - start;
//             auto elapsed = wallclock ? sw_time.count() : hw_time(event);
//             best = std::min(best, elapsed);
//         }
//         auto gops = 2.0 * M * N * K / best / 1e9;
//         printf("Best in %f seconds (%f gops)\n", best, gops);

//         if (validate) {
//             printf("Checking results... "); fflush(stdout);
//             std::vector<float> C_check(C_ref.size());
//             queue.enqueueReadBuffer(C, CL_TRUE, 0, C_check.size() * sizeof(C_check[0]), C_check.data());
//             check_results(M, N, C_check, C_ref);
//             printf(" done!\n");
//         }
//     }
// }

// template<int tM, int tN, int tK>
// static void go_dpas_vnni(
//     cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
//     cl::Buffer& C, cl::Buffer& A, cl::Buffer& B,
//     size_t M, size_t N, size_t K,
//     const std::vector<float>& C_ref)
// {
//     printf("%80s: ", makeTestName(__FUNCTION__, tM, tN, tK, M, N, K).c_str()); fflush(stdout);

//     std::string kernelName = "bfloat16_dpas_vnni";
//     kernelName += "_m" + std::to_string(tM);
//     kernelName += "_n" + std::to_string(tN);
//     cl::Kernel kernel{program, kernelName.c_str()};
//     if (kernel() == nullptr) {
//         printf("unsupported.\n");
//     } else {
//         kernel.setArg(0, C);
//         kernel.setArg(1, A);
//         kernel.setArg(2, B);
//         kernel.setArg(3, static_cast<cl_int>(K));

//         queue.enqueueFillBuffer(C, 0, 0, C_ref.size());

//         float best = 999.0f;
//         for (int test = 0; test < testIterations; test++) {
//             cl::Event event;
//             auto start = test_clock::now();
//             queue.enqueueNDRangeKernel(kernel, cl::NullRange,
//                 cl::NDRange{N, M/tM}, cl::NullRange, nullptr, &event);
//             queue.finish();
//             auto end = test_clock::now();
//             std::chrono::duration<float> sw_time = end - start;
//             auto elapsed = wallclock ? sw_time.count() : hw_time(event);
//             best = std::min(best, elapsed);
//         }
//         auto gops = 2.0 * M * N * K / best / 1e9;
//         printf("Best in %f seconds (%f gops)\n", best, gops);

//         if (validate) {
//             printf("Checking results... "); fflush(stdout);
//             std::vector<float> C_check(C_ref.size());
//             queue.enqueueReadBuffer(C, CL_TRUE, 0, C_check.size() * sizeof(C_check[0]), C_check.data());
//             check_results(M, N, C_check, C_ref);
//             printf(" done!\n");
//         }
//     }
// }

// template<int tM, int tN, int tK, int MM, int NN>
// static void go_dpas_vnni_tiled(
//     cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
//     cl::Buffer& C, cl::Buffer& A, cl::Buffer& B,
//     size_t M, size_t N, size_t K,
//     const std::vector<float>& C_ref)
// {
//     printf("%80s: ", makeTestName(__FUNCTION__, tM, tN, tK, MM, NN, M, N, K).c_str()); fflush(stdout);

//     std::string kernelName = "bfloat16_dpas_vnni_tiled";
//     kernelName += "_m" + std::to_string(tM);
//     kernelName += "_n" + std::to_string(tN);
//     kernelName += "_" + std::to_string(MM);
//     kernelName += "x" + std::to_string(NN);
//     cl::Kernel kernel{program, kernelName.c_str()};
//     if (kernel() == nullptr) {
//         printf("unsupported.\n");
//     } else if (tM * MM > M) {
//         printf("M is too small.\n");
//     } else if (tN * NN > N) {
//         printf("N is too small.\n");
//     } else {
//         kernel.setArg(0, C);
//         kernel.setArg(1, A);
//         kernel.setArg(2, B);
//         kernel.setArg(3, static_cast<cl_int>(K));

//         queue.enqueueFillBuffer(C, 0, 0, C_ref.size());

//         float best = 999.0f;
//         for (int test = 0; test < testIterations; test++) {
//             cl::Event event;
//             auto start = test_clock::now();
//             queue.enqueueNDRangeKernel(kernel, cl::NullRange,
//                 cl::NDRange{N/NN, M/tM/MM}, cl::NullRange, nullptr, &event);
//             queue.finish();
//             auto end = test_clock::now();
//             std::chrono::duration<float> sw_time = end - start;
//             auto elapsed = wallclock ? sw_time.count() : hw_time(event);
//             best = std::min(best, elapsed);
//         }
//         auto gops = 2.0 * M * N * K / best / 1e9;
//         printf("Best in %f seconds (%f gops)\n", best, gops);

//         if (validate) {
//             printf("Checking results... "); fflush(stdout);
//             std::vector<float> C_check(C_ref.size());
//             queue.enqueueReadBuffer(C, CL_TRUE, 0, C_check.size() * sizeof(C_check[0]), C_check.data());
//             check_results(M, N, C_check, C_ref);
//             printf(" done!\n");
//         }
//     }
// }

// template<int tM, int tN, int tK>
// static void go_dpas_blockread_rowmajor(
//     cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
//     cl::Buffer& C, cl::Buffer& A, cl::Buffer& B,
//     size_t M, size_t N, size_t K,
//     const std::vector<float>& C_ref)
// {
//     printf("%80s: ", makeTestName(__FUNCTION__, tM, tN, tK, M, N, K).c_str()); fflush(stdout);

//     std::string kernelName = "bfloat16_dpas_blockread_rowmajor";
//     kernelName += "_m" + std::to_string(tM);
//     kernelName += "_n" + std::to_string(tN);
//     cl::Kernel kernel{program, kernelName.c_str()};
//     if (kernel() == nullptr) {
//         printf("unsupported.\n");
//     } else {
//         kernel.setArg(0, C);
//         kernel.setArg(1, A);
//         kernel.setArg(2, B);
//         kernel.setArg(3, static_cast<cl_int>(K));

//         queue.enqueueFillBuffer(C, 0, 0, C_ref.size());

//         float best = 999.0f;
//         for (int test = 0; test < testIterations; test++) {
//             cl::Event event;
//             auto start = test_clock::now();
//             queue.enqueueNDRangeKernel(kernel, cl::NullRange,
//                 cl::NDRange{N, M/tM}, cl::NullRange, nullptr, &event);
//             queue.finish();
//             auto end = test_clock::now();
//             std::chrono::duration<float> sw_time = end - start;
//             auto elapsed = wallclock ? sw_time.count() : hw_time(event);
//             best = std::min(best, elapsed);
//         }
//         auto gops = 2.0 * M * N * K / best / 1e9;
//         printf("Best in %f seconds (%f gops)\n", best, gops);

//         if (validate) {
//             printf("Checking results... "); fflush(stdout);
//             std::vector<float> C_check(C_ref.size());
//             queue.enqueueReadBuffer(C, CL_TRUE, 0, C_check.size() * sizeof(C_check[0]), C_check.data());
//             check_results(M, N, C_check, C_ref);
//             printf(" done!\n");
//         }
//     }
// }

// template<int tM, int tN, int tK, int MM, int NN>
// static void go_dpas_blockread_rowmajor_tiled(
//     cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
//     cl::Buffer& C, cl::Buffer& A, cl::Buffer& B,
//     size_t M, size_t N, size_t K,
//     const std::vector<float>& C_ref)
// {
//     printf("%80s: ", makeTestName(__FUNCTION__, tM, tN, tK, MM, NN, M, N, K).c_str()); fflush(stdout);

//     std::string kernelName = "bfloat16_dpas_blockread_rowmajor_tiled";
//     kernelName += "_m" + std::to_string(tM);
//     kernelName += "_n" + std::to_string(tN);
//     kernelName += "_" + std::to_string(MM);
//     kernelName += "x" + std::to_string(NN);
//     cl::Kernel kernel{program, kernelName.c_str()};
//     if (kernel() == nullptr) {
//         printf("unsupported.\n");
//     } else if (tM * MM > M) {
//         printf("M is too small.\n");
//     } else if (tN * NN > N) {
//         printf("N is too small.\n");
//     } else {
//         kernel.setArg(0, C);
//         kernel.setArg(1, A);
//         kernel.setArg(2, B);
//         kernel.setArg(3, static_cast<cl_int>(K));

//         queue.enqueueFillBuffer(C, 0, 0, C_ref.size());

//         float best = 999.0f;
//         for (int test = 0; test < testIterations; test++) {
//             cl::Event event;
//             auto start = test_clock::now();
//             queue.enqueueNDRangeKernel(kernel, cl::NullRange,
//                 cl::NDRange{N/NN, M/tM/MM}, cl::NullRange, nullptr, &event);
//             queue.finish();
//             auto end = test_clock::now();
//             std::chrono::duration<float> sw_time = end - start;
//             auto elapsed = wallclock ? sw_time.count() : hw_time(event);
//             best = std::min(best, elapsed);
//         }
//         auto gops = 2.0 * M * N * K / best / 1e9;
//         printf("Best in %f seconds (%f gops)\n", best, gops);

//         if (validate) {
//             printf("Checking results... "); fflush(stdout);
//             std::vector<float> C_check(C_ref.size());
//             queue.enqueueReadBuffer(C, CL_TRUE, 0, C_check.size() * sizeof(C_check[0]), C_check.data());
//             check_results(M, N, C_check, C_ref);
//             printf(" done!\n");
//         }
//     }
// }

// template<int tM, int tN, int tK>
// static void go_dpas_blockread_vnni(
//     cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
//     cl::Buffer& C, cl::Buffer& A, cl::Buffer& B,
//     size_t M, size_t N, size_t K,
//     const std::vector<float>& C_ref)
// {
//     printf("%80s: ", makeTestName(__FUNCTION__, tM, tN, tK, M, N, K).c_str()); fflush(stdout);

//     std::string kernelName = "bfloat16_dpas_blockread_vnni";
//     kernelName += "_m" + std::to_string(tM);
//     kernelName += "_n" + std::to_string(tN);
//     cl::Kernel kernel{program, kernelName.c_str()};
//     if (kernel() == nullptr) {
//         printf("unsupported.\n");
//     } else {
//         kernel.setArg(0, C);
//         kernel.setArg(1, A);
//         kernel.setArg(2, B);
//         kernel.setArg(3, static_cast<cl_int>(K));

//         queue.enqueueFillBuffer(C, 0, 0, C_ref.size());

//         float best = 999.0f;
//         for (int test = 0; test < testIterations; test++) {
//             cl::Event event;
//             auto start = test_clock::now();
//             queue.enqueueNDRangeKernel(kernel, cl::NullRange,
//                 cl::NDRange{N, M/tM}, cl::NullRange, nullptr, &event);
//             queue.finish();
//             auto end = test_clock::now();
//             std::chrono::duration<float> sw_time = end - start;
//             auto elapsed = wallclock ? sw_time.count() : hw_time(event);
//             best = std::min(best, elapsed);
//         }
//         auto gops = 2.0 * M * N * K / best / 1e9;
//         printf("Best in %f seconds (%f gops)\n", best, gops);

//         if (validate) {
//             printf("Checking results... "); fflush(stdout);
//             std::vector<float> C_check(C_ref.size());
//             queue.enqueueReadBuffer(C, CL_TRUE, 0, C_check.size() * sizeof(C_check[0]), C_check.data());
//             check_results(M, N, C_check, C_ref);
//             printf(" done!\n");
//         }
//     }
// }

// from cutlass 3.4. current port is on 3.2 and needs to be rebased
template <class Tuple>
CUTE_HOST_DEVICE constexpr
auto
make_inttuple_iter(Tuple const& t) {
  return ArithmeticTupleIterator(as_arithmetic_tuple(t));
}

#ifdef __SYCL_DEVICE_ONLY__ 
template<class T, class F> T vec_as(const F& x) { return sycl::bit_cast<T>(x); } 
#else 
template<class T, class F> T vec_as(const F& x) { return x.template as<T>(); }
#endif

template<class T>
using global_decorated = typename sycl::decorated_global_ptr<std::remove_pointer_t<T>>::pointer;
template<class T> long as_long(const T &x) { return sycl::bit_cast<long>(x); }
template<class T> short8 as_short8(const T& x) { return vec_as<short8>(x); }
template<class T> int8 as_int8(const T& x)   { return vec_as<int8>(x); }
template<class T> uint8 as_uint8(const T& x)  { return vec_as<uint8>(x); }

static void intel_subgroup_block_write_u32_m8k16v1(global_decorated<void*> base_address, int width, int height, int pitch, int2_ coord, uint8 data)
{
    __builtin_IB_subgroup_block_write_flat_u32_m8k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord, data);
}
static ushort8 intel_subgroup_block_read_u16_m8k16(global_decorated<const void*> base_address, int width, int height, int pitch, int2_ coord)
{
    return __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}
inline uint8 intel_subgroup_block_read_u32_m8k16(global_decorated<const void*> base_address, int width, int height, int pitch, int2_ coord)
{
    return __builtin_IB_subgroup_block_read_flat_u32_m8k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}

template<int tM, int tN, int tK, int MM, int NN>
static void go_dpas_blockread_vnni_tiled(
    sycl::queue queue,
    std::vector<float>& c_vec, sycl::buffer<bfloat16> a, sycl::buffer<bfloat16> b,
    // cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
    // cl::Buffer& C, cl::Buffer& A, cl::Buffer& B,
    size_t M, size_t N, size_t K,
    const std::vector<float>& C_ref)
{
    printf("%80s: ", makeTestName(__FUNCTION__, tM, tN, tK, MM, NN, M, N, K).c_str()); fflush(stdout);

    // std::string kernelName = "bfloat16_dpas_blockread_vnni_tiled";
    // kernelName += "_m" + std::to_string(tM);
    // kernelName += "_n" + std::to_string(tN);
    // kernelName += "_" + std::to_string(MM);
    // kernelName += "x" + std::to_string(NN);
    // cl::Kernel kernel{program, kernelName.c_str()};
    // if (kernel() == nullptr) {
    //     printf("unsupported.\n");
    // } else 
    if (tM * MM > M) {
        printf("M is too small.\n");
    } else if (tN * NN > N) {
        printf("N is too small.\n");
    } else {
        // kernel.setArg(0, C);
        // kernel.setArg(1, A);
        // kernel.setArg(2, B);
        // kernel.setArg(3, static_cast<cl_int>(K));
        // queue.enqueueFillBuffer(C, 0, 0, C_ref.size());

        float best = 999.0f;
        for (int test = 0; test < testIterations; test++) {
            sycl::buffer c{c_vec};
            queue.submit([&](sycl::handler& cgh) {
                sycl::accessor accA { a, cgh, sycl::read_only };
                sycl::accessor accB { b, cgh, sycl::read_only };
                sycl::accessor accC { c, cgh, sycl::write_only };
                cgh.parallel_for/*<dpas_blockread_vnni_tiled<tM, tN, tK, MM, NN>>*/(sycl::nd_range<2>{{ M/tM/MM, N/NN }, { 1, 16}},
                 [=](sycl::nd_item<2> id) [[sycl::reqd_sub_group_size(16)]] {
    // const int tM = 8;
    // const int tN = 16;
    // const int M = get_global_size(1) * tM;
    // const int N = get_global_size(0) * NN;
    // const int m = get_group_id(1) * tM * MM;
    // const int n = get_group_id(0) * tN * NN;
    const int M = id.get_global_range(0) * tM * MM;
    const int N = id.get_global_range(1) * NN;
    const int m = id.get_group().get_group_id(0) * tM * MM;
    const int n = id.get_group().get_group_id(1) * tN * NN;

    auto A = accA.get_multi_ptr<sycl::access::decorated::yes>().get();
    auto B = accB.get_multi_ptr<sycl::access::decorated::yes>().get();
    auto C = accC.get_multi_ptr<sycl::access::decorated::yes>().get();

    float8 sum[MM][NN];
    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            sum[mm][nn] = 0;
        }
    }

    using namespace cute;
    // TiledMMA<MMA_Atom<XE_8x16x16_BF16BF16F32F32_NN>,
    //          Layout<Shape<_1, _1, _1>>, Layout<Shape<Int<MM>, Int<NN>, _1>>>
    //     tiled_mma;
    auto A_copy = make_xe_2d_copy(make_tensor(make_gmem_ptr((ushort*)A), make_shape(M, K))); //cast should probably not be here
    auto B_copy = make_xe_2d_copy(make_tensor(make_gmem_ptr((uint*)B), make_shape(K, N)));
    for (int k = 0; k < K; k += tK) {
        short8  aData[MM];
        for (int mm = 0; mm < MM; mm++) {
            aData[mm] = as_short8(intel_subgroup_block_read_u16_m8k16(A, K * sizeof(ushort), M, K * sizeof(ushort), int2_{k, m + mm * tM}));
        }

        int8    bData[NN];
        for (int nn = 0; nn < NN; nn++) {
            bData[nn] = as_int8(intel_subgroup_block_read_u32_m8k16(B, N * sizeof(uint), K, N * sizeof(uint), int2_{n + nn * tN, k / 2}));
        }
        Tensor aT = make_tensor(make_rmem_ptr((bfloat16*)&aData), Layout<Shape<_1, Int<MM>>,Stride<_1, _8>>{});
        Tensor bT = make_tensor(make_rmem_ptr((bfloat16*)&bData), Layout<Shape<_1, Int<NN>>,Stride<_1, _16>>{});
        Tensor cT = make_tensor(make_rmem_ptr((float*)&sum), Layout<Shape<_1, Int<MM>, Int<NN>>,Stride<_1, Int<8*NN>, _8>>{});
        gemm(MMA_Atom<XE_8x16x16_BF16BF16F32F32_NN>(), aT, bT, cT);
    }

    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            intel_subgroup_block_write_u32_m8k16v1(C, N * sizeof(float), M, N * sizeof(float), int2_{n + nn * tN, m + mm * tM}, as_uint8(sum[mm][nn]));
        }
    }
});}).wait_and_throw();
        //     cl::Event event;
        //     auto start = test_clock::now();
        //     queue.enqueueNDRangeKernel(kernel, cl::NullRange,
        //         cl::NDRange{N/NN, M/tM/MM}, cl::NullRange, nullptr, &event);
        //     queue.finish();
        //     auto end = test_clock::now();
        //     std::chrono::duration<float> sw_time = end - start;
        //     auto elapsed = wallclock ? sw_time.count() : hw_time(event);
        //     best = std::min(best, elapsed);
        }
        auto gops = 2.0 * M * N * K / best / 1e9;
        printf("Best in %f seconds (%f gops)\n", best, gops);

        if (validate) {
            printf("Checking results... "); fflush(stdout);
            // std::vector<float> C_check(C_ref.size());
            // queue.enqueueReadBuffer(C, CL_TRUE, 0, C_check.size() * sizeof(C_check[0]), C_check.data());
            check_results(M, N, c_vec, C_ref);
            printf(" done!\n");
        }
    }
}

int main(int argc, char** argv)
{
    int platformIndex = 0;
    int deviceIndex = 0;

    // std::string fileName("matrix_kernels.cl");
    std::string buildOptions;
    size_t matrixSize = 512;

    {
        popl::OptionParser op("Supported Options");
        // op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        // op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        // op.add<popl::Value<std::string>>("", "file", "Kernel File Name", fileName, &fileName);
        // op.add<popl::Value<std::string>>("", "options", "Program Build Options", buildOptions, &buildOptions);
        op.add<popl::Value<size_t>>("m", "matrixsize", "Matrix Size", matrixSize, &matrixSize);
        op.add<popl::Value<int>>("i", "iterations", "Test Iterations", testIterations, &testIterations);
        op.add<popl::Switch>("", "validate", "Validate Results", &validate);
        op.add<popl::Switch>("", "identity", "Use Identity Data", &identityData);
        op.add<popl::Switch>("", "fixed", "Use Fixed Data", &fixedData);
        op.add<popl::Switch>("", "emulate", "Unconditionally Emulate dpas", &emulate);
        op.add<popl::Switch>("", "wallclock", "Measure Wallclock Time", &wallclock);
        op.add<popl::Value<float>>("", "threshold", "Local Error Threshold", threshold, &threshold);
        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: matrixexperiments [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    // std::vector<cl::Platform> platforms;
    // cl::Platform::get(&platforms);

    // printf("Running on platform: %s\n",
    //     platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str() );

    // std::vector<cl::Device> devices;
    // platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    // cl::Device& device = devices[deviceIndex];
    // printf("Running on device: %s\n",
    //     device.getInfo<CL_DEVICE_NAME>().c_str() );

    // auto minSubGroupSize = findMinSubGroupSize(device);
    size_t minSubGroupSize = 16;

    bool has_simd8 = minSubGroupSize == 8;
    bool emulate_tN8 = true;
    bool emulate_tN16 = true;
    // if (!emulate && checkDeviceForExtension(device, "cl_intel_subgroup_matrix_multiply_accumulate")) {
    if (!emulate && 1) {
        printf("Found support for cl_intel_subgroup_matrix_multiply_accumulate, min sub-group size is: %zu\n", minSubGroupSize);
        switch(minSubGroupSize) {
            case 8: emulate_tN8 = false; break;
            case 16: emulate_tN16 = false; break;
            default: break;
        }
    }

    buildOptions += " -DHAS_SIMD8=" + std::to_string(has_simd8);
    buildOptions += " -DEMULATE_tN8=" + std::to_string(emulate_tN8);
    buildOptions += " -DEMULATE_tN16=" + std::to_string(emulate_tN16);

    printf("Config:\n");
    printf("\tTest Iterations: %d\n", testIterations);
    printf("\tValidating data?: %s\n", validate ? "true" : "false");
    printf("\tFixed data?: %s\n", fixedData ? "true" : "false");
    printf("\tWallclock time?: %s\n", wallclock ? "true" : "false");
    printf("\tEmulate dpas for tN=8?: %s\n", emulate_tN8 ? "true" : "false");
    printf("\tEmulate dpas for tN=16?: %s\n", emulate_tN16 ? "true" : "false");

    sycl::queue queue;
    // cl::Context context{device};
    // cl::CommandQueue queue{context, device, CL_QUEUE_PROFILING_ENABLE};

    // printf("Reading program source from file: %s\n", fileName.c_str() );
    // std::string kernelString = readStringFromFile(fileName.c_str());

    // printf("Building program with build options: %s\n",
    //     buildOptions.empty() ? "(none)" : buildOptions.c_str() );
    // cl::Program program{ context, kernelString };
    // program.build(buildOptions.c_str());
    // for( auto& device : program.getInfo<CL_PROGRAM_DEVICES>() )
    // {
    //     printf("Program build log for device %s:\n",
    //         device.getInfo<CL_DEVICE_NAME>().c_str() );
    //     printf("%s\n",
    //         program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device).c_str() );
    // }

    const auto M = matrixSize;
    const auto N = matrixSize;
    const auto K = matrixSize;

    std::vector<bfloat16> A_vec(M * K);
    std::vector<bfloat16> B_vec(K * N);
    std::vector<bfloat16> Bvnni_vec(K * N);
    std::vector<float> C_vec(M * N);
    std::vector<float> C_ref(M * N);

    printf("Initializing source matrices...\n");
    fill_matrix(A_vec, M, K);
    fill_matrix(B_vec, K, N);

    vnni_matrix(Bvnni_vec, B_vec, K, N, 2);

    if (validate) {
        printf("Computing reference...\n");
        compute_reference(C_ref, A_vec, B_vec, M, N, K);
    }

    printf("Creating source buffers...\n");
    sycl::buffer A{A_vec};
    sycl::buffer B{B_vec};
    sycl::buffer Bvnni{Bvnni_vec};
    // sycl::buffer<float> C{C_ref.size()};
    // cl::Buffer A{context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, A_vec.size() * sizeof(A_vec[0]), A_vec.data()};
    // cl::Buffer B{context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, B_vec.size() * sizeof(B_vec[0]), B_vec.data()};
    // cl::Buffer Bvnni{context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Bvnni_vec.size() * sizeof(Bvnni_vec[0]), Bvnni_vec.data()};
    // cl::Buffer C{context, CL_MEM_WRITE_ONLY, C_ref.size() * sizeof(C_ref[0])};

    printf("Running tests...\n");

    // go_naive(context, program, queue, C, A, B, M, N, K, C_ref);

    // go_dpas_rowmajor<1, 8, 16>(context, program, queue, C, A, B, M, N, K, C_ref);
    // go_dpas_rowmajor<2, 8, 16>(context, program, queue, C, A, B, M, N, K, C_ref);
    // go_dpas_rowmajor<4, 8, 16>(context, program, queue, C, A, B, M, N, K, C_ref);
    // go_dpas_rowmajor<8, 8, 16>(context, program, queue, C, A, B, M, N, K, C_ref);

    // go_dpas_rowmajor_tiled<8, 8, 16, 2, 1>(context, program, queue, C, A, B, M, N, K, C_ref);
    // go_dpas_rowmajor_tiled<8, 8, 16, 1, 2>(context, program, queue, C, A, B, M, N, K, C_ref);
    // go_dpas_rowmajor_tiled<8, 8, 16, 2, 2>(context, program, queue, C, A, B, M, N, K, C_ref);
    // go_dpas_rowmajor_tiled<8, 8, 16, 4, 2>(context, program, queue, C, A, B, M, N, K, C_ref);
    // go_dpas_rowmajor_tiled<8, 8, 16, 2, 4>(context, program, queue, C, A, B, M, N, K, C_ref);
    // go_dpas_rowmajor_tiled<8, 8, 16, 4, 4>(context, program, queue, C, A, B, M, N, K, C_ref);

    // go_dpas_vnni<1, 8, 16>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
    // go_dpas_vnni<2, 8, 16>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
    // go_dpas_vnni<4, 8, 16>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
    // go_dpas_vnni<8, 8, 16>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);

    // go_dpas_vnni_tiled<8, 8, 16, 2, 1>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
    // go_dpas_vnni_tiled<8, 8, 16, 1, 2>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
    // go_dpas_vnni_tiled<8, 8, 16, 2, 2>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
    // go_dpas_vnni_tiled<8, 8, 16, 4, 2>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
    // go_dpas_vnni_tiled<8, 8, 16, 2, 4>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
    // go_dpas_vnni_tiled<8, 8, 16, 4, 4>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);

    // go_dpas_rowmajor<1, 16, 16>(context, program, queue, C, A, B, M, N, K, C_ref);
    // go_dpas_rowmajor<2, 16, 16>(context, program, queue, C, A, B, M, N, K, C_ref);
    // go_dpas_rowmajor<4, 16, 16>(context, program, queue, C, A, B, M, N, K, C_ref);
    // go_dpas_rowmajor<8, 16, 16>(context, program, queue, C, A, B, M, N, K, C_ref);

    // go_dpas_rowmajor_tiled<8, 16, 16, 2, 1>(context, program, queue, C, A, B, M, N, K, C_ref);
    // go_dpas_rowmajor_tiled<8, 16, 16, 1, 2>(context, program, queue, C, A, B, M, N, K, C_ref);
    // go_dpas_rowmajor_tiled<8, 16, 16, 2, 2>(context, program, queue, C, A, B, M, N, K, C_ref);
    // go_dpas_rowmajor_tiled<8, 16, 16, 4, 2>(context, program, queue, C, A, B, M, N, K, C_ref);
    // go_dpas_rowmajor_tiled<8, 16, 16, 2, 4>(context, program, queue, C, A, B, M, N, K, C_ref);
    // go_dpas_rowmajor_tiled<8, 16, 16, 4, 4>(context, program, queue, C, A, B, M, N, K, C_ref);

    // go_dpas_vnni<1, 16, 16>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
    // go_dpas_vnni<2, 16, 16>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
    // go_dpas_vnni<4, 16, 16>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
    // go_dpas_vnni<8, 16, 16>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);

    // go_dpas_vnni_tiled<8, 16, 16, 2, 1>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
    // go_dpas_vnni_tiled<8, 16, 16, 1, 2>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
    // go_dpas_vnni_tiled<8, 16, 16, 2, 2>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
    // go_dpas_vnni_tiled<8, 16, 16, 4, 2>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
    // go_dpas_vnni_tiled<8, 16, 16, 2, 4>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
    // go_dpas_vnni_tiled<8, 16, 16, 4, 4>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);

    // go_dpas_blockread_rowmajor<1, 16, 16>(context, program, queue, C, A, B, M, N, K, C_ref);
    // go_dpas_blockread_rowmajor<2, 16, 16>(context, program, queue, C, A, B, M, N, K, C_ref);
    // go_dpas_blockread_rowmajor<4, 16, 16>(context, program, queue, C, A, B, M, N, K, C_ref);
    // go_dpas_blockread_rowmajor<8, 16, 16>(context, program, queue, C, A, B, M, N, K, C_ref);

    // go_dpas_blockread_rowmajor_tiled<8, 16, 16, 2, 1>(context, program, queue, C, A, B, M, N, K, C_ref);
    // go_dpas_blockread_rowmajor_tiled<8, 16, 16, 1, 2>(context, program, queue, C, A, B, M, N, K, C_ref);
    // go_dpas_blockread_rowmajor_tiled<8, 16, 16, 2, 2>(context, program, queue, C, A, B, M, N, K, C_ref);
    // go_dpas_blockread_rowmajor_tiled<8, 16, 16, 4, 2>(context, program, queue, C, A, B, M, N, K, C_ref);
    // go_dpas_blockread_rowmajor_tiled<8, 16, 16, 2, 4>(context, program, queue, C, A, B, M, N, K, C_ref);
    // go_dpas_blockread_rowmajor_tiled<8, 16, 16, 4, 4>(context, program, queue, C, A, B, M, N, K, C_ref);

    // go_dpas_blockread_vnni<1, 16, 16>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
    // go_dpas_blockread_vnni<2, 16, 16>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
    // go_dpas_blockread_vnni<4, 16, 16>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
    // go_dpas_blockread_vnni<8, 16, 16>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);

    go_dpas_blockread_vnni_tiled<8, 16, 16, 1, 1>(queue, C_vec, A, Bvnni, M, N, K, C_ref);
    go_dpas_blockread_vnni_tiled<8, 16, 16, 2, 1>(queue, C_vec, A, Bvnni, M, N, K, C_ref);
    go_dpas_blockread_vnni_tiled<8, 16, 16, 1, 2>(queue, C_vec, A, Bvnni, M, N, K, C_ref);
    go_dpas_blockread_vnni_tiled<8, 16, 16, 2, 2>(queue, C_vec, A, Bvnni, M, N, K, C_ref);
    go_dpas_blockread_vnni_tiled<8, 16, 16, 4, 2>(queue, C_vec, A, Bvnni, M, N, K, C_ref);
    go_dpas_blockread_vnni_tiled<8, 16, 16, 2, 4>(queue, C_vec, A, Bvnni, M, N, K, C_ref);
    go_dpas_blockread_vnni_tiled<8, 16, 16, 4, 4>(queue, C_vec, A, Bvnni, M, N, K, C_ref);

    printf("Done.\n");

    return 0;
}
