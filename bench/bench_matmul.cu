#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); std::exit(1); } } while (0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cuBLAS %d @ %s:%d\n", (int)s, __FILE__, __LINE__); std::exit(1); } } while (0)
#define CURAND_CHECK(x) do { curandStatus_t s = (x); if (s != CURAND_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cuRAND %d @ %s:%d\n", (int)s, __FILE__, __LINE__); std::exit(1); } } while (0)

namespace vlm {
    void matmul_v1_launch(const float*, const float*, float*, int, int, int, cudaStream_t);
}

struct Shape {
    int M, N, K;
    const char* tag;
};
using LaunchFn = void(*)(const float*, const float*, float*, int, int, int, cudaStream_t);
struct Kernel {
    const char* name;
    LaunchFn launch;
};

static void cublas_sgemm_rm(cublasHandle_t h, const float* A, const float* B, float* C,
                            int M, int N, int K, cudaStream_t s) {
    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSetStream(h, s));
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha, B, N, A, K, &beta, C, N));
}   

static void cublas_tf32_rm(cublasHandle_t h, const float* A, const float* B, float* C,
                           int M, int N, int K, cudaStream_t s) {
    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSetStream(h, s));
    CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K, &alpha,
                              B, CUDA_R_32F, N,
                              A, CUDA_R_32F, K,
                              &beta, C, CUDA_R_32F, N,
                              CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT));
}

static double quantile(std::vector<double>& v, double q) {
    std::sort(v.begin(), v.end());
    size_t i = std::min(v.size() - 1, static_cast<size_t>(v.size() * q));
    return v[i];
}

template <class F>
static std::pair<double, double> time_kernel(F&& launch, int warmup, int iters, cudaStream_t s) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaStreamSynchronize(s));

    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));

    std::vector<double> ts;
    ts.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(a, s));
        launch();
        CUDA_CHECK(cudaEventRecord(b, s));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        ts.push_back(static_cast<double>(ms));
    }
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return {quantile(ts, 0.5), quantile(ts, 0.99)};
}

static float max_abs_diff(const float* a, const float* b, size_t n) {
    float m = 0.f;
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

static float tol_for_K(int K) {
    return std::max(1e-3f, 1e-3f * std::sqrt(static_cast<float>(K)) / 32.f);
}

int main(int argc, char** argv) {
    int iters = 50;
    int warmup = 10;
    unsigned long long seed = 0xC0FFEEULL;
    std::string csv_path = "bench/results/matmul.csv";
    std::string kernel_filter = "all";
    std::string shape_filter = "all";
    int sM = 0, sN = 0, sK = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing arg for %s\n", a.c_str());
                std::exit(1);
            }
            return std::string(argv[++i]);
        };
        if (a == "--iters") iters = std::stoi(next());
        else if (a == "--warmup") warmup = std::stoi(next());
        else if (a == "--seed") seed = std::stoull(next());
        else if (a == "--csv") csv_path = next();
        else if (a == "--kernel") kernel_filter = next();
        else if (a == "--shapes") shape_filter = next();
        else if (a == "--shape") {
            std::string s = next();
            if (std::sscanf(s.c_str(), "%dx%dx%d", &sM, &sN, &sK) != 3) {
                std::fprintf(stderr, "bad shape %s (want MxNxK)\n", s.c_str());
                std::exit(1);
            }
            shape_filter = "single";
        } else {
            std::fprintf(stderr, "unknown arg %s\n", a.c_str());
            std::exit(1);
        }
    }

    std::vector<Shape> shapes;
    auto add_if = [&](const char* group, std::initializer_list<Shape> ss) {
        if (shape_filter == "all" || shape_filter == group) {
            for (const auto& s : ss) shapes.push_back(s);
        }
    };
    if (shape_filter == "single") {
        shapes.push_back({sM, sN, sK, "custom"});
    } else {
        add_if("square", {
            {256, 256, 256, "sq256"},
            {512, 512, 512, "sq512"},
            {1024, 1024, 1024, "sq1024"},
            {2048, 2048, 2048, "sq2048"},
            {4096, 4096, 4096, "sq4096"},
            {8192, 8192, 8192, "sq8192"},
        });
        add_if("transformer", {
            {16384, 1152, 384, "qkv_proj"},
            {16384, 1536, 384, "ffn_up"},
            {16384, 384, 1536, "ffn_down"},
            {16384, 65, 384, "vocab_head"},
            {8192, 4096, 1024, "big_ffn"},
        });
        add_if("skinny", {
            {16384, 256, 256, "tall"},
            {256, 16384, 256, "wide"},
            {128, 4096, 4096, "tiny_M"},
        });
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("# GPU=%s SMs=%d cap=%d.%d iters=%d warmup=%d\n",
                prop.name, prop.multiProcessorCount, prop.major, prop.minor, iters, warmup);

    std::vector<Kernel> kernels = {
        {"v1", vlm::matmul_v1_launch},
    };

    auto csv_dir = std::filesystem::path(csv_path).parent_path();
    if (!csv_dir.empty()) std::filesystem::create_directories(csv_dir);
    bool csv_exists = std::filesystem::exists(csv_path);
    std::ofstream csv(csv_path, std::ios::app);
    if (!csv_exists) {
        csv << "gpu,kernel,tag,M,N,K,iters,ms_med,ms_p99,tflops,vs_sgemm_pct,vs_tf32_pct,max_abs_err\n";
    }

    std::printf("%-14s %-12s %6s %6s %6s %10s %10s %9s %9s %9s\n",
                "kernel", "tag", "M", "N", "K", "ms_med", "ms_p99", "TFLOPS", "%sgemm", "%tf32");

    curandGenerator_t rng;
    CURAND_CHECK(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetStream(rng, stream));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rng, seed));

    for (const Shape& sh : shapes) {
        const size_t na = static_cast<size_t>(sh.M) * sh.K;
        const size_t nb = static_cast<size_t>(sh.K) * sh.N;
        const size_t nc = static_cast<size_t>(sh.M) * sh.N;
        const double flops = 2.0 * sh.M * sh.N * sh.K;

        float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dCref = nullptr;
        CUDA_CHECK(cudaMalloc(&dA, na * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, nb * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, nc * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dCref, nc * sizeof(float)));

        CURAND_CHECK(curandGenerateUniform(rng, dA, na));
        CURAND_CHECK(curandGenerateUniform(rng, dB, nb));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        cublas_sgemm_rm(handle, dA, dB, dCref, sh.M, sh.N, sh.K, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<float> hC(nc), hCref(nc);
        CUDA_CHECK(cudaMemcpy(hCref.data(), dCref, nc * sizeof(float), cudaMemcpyDeviceToHost));

        auto [sg_m, sg_p] = time_kernel(
            [&] { cublas_sgemm_rm(handle, dA, dB, dC, sh.M, sh.N, sh.K, stream); },
            warmup, iters, stream);
        const double sg_tflops = flops / (sg_m * 1e-3) / 1e12;
        std::printf("%-14s %-12s %6d %6d %6d %10.4f %10.4f %9.2f %9.1f %9s\n",
                    "cublas_sgemm", sh.tag, sh.M, sh.N, sh.K, sg_m, sg_p, sg_tflops, 100.0, "-");
        csv << prop.name << ",cublas_sgemm," << sh.tag << "," << sh.M << "," << sh.N << "," << sh.K
            << "," << iters << "," << sg_m << "," << sg_p << "," << sg_tflops << ",100.0,,0\n";

        auto [tf_m, tf_p] = time_kernel(
            [&] { cublas_tf32_rm(handle, dA, dB, dC, sh.M, sh.N, sh.K, stream); },
            warmup, iters, stream);
        const double tf_tflops = flops / (tf_m * 1e-3) / 1e12;
        std::printf("%-14s %-12s %6d %6d %6d %10.4f %10.4f %9.2f %9s %9.1f\n",
                    "cublas_tf32", sh.tag, sh.M, sh.N, sh.K, tf_m, tf_p, tf_tflops, "-", 100.0);
        csv << prop.name << ",cublas_tf32," << sh.tag << "," << sh.M << "," << sh.N << "," << sh.K
            << "," << iters << "," << tf_m << "," << tf_p << "," << tf_tflops << ",,100.0,0\n";

        for (const Kernel& k : kernels) {
            if (kernel_filter != "all" && kernel_filter != k.name) continue;

            CUDA_CHECK(cudaMemsetAsync(dC, 0, nc * sizeof(float), stream));
            k.launch(dA, dB, dC, sh.M, sh.N, sh.K, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaMemcpy(hC.data(), dC, nc * sizeof(float), cudaMemcpyDeviceToHost));

            const float err = max_abs_diff(hC.data(), hCref.data(), nc);
            const float tol = tol_for_K(sh.K);
            if (err > tol) {
                std::fprintf(stderr, "[FAIL] %s @ %s err=%.4g tol=%.4g\n", k.name, sh.tag, err, tol);
                continue;
            }

            auto [m, p] = time_kernel(
                [&] { k.launch(dA, dB, dC, sh.M, sh.N, sh.K, stream); },
                warmup, iters, stream);
            const double tflops = flops / (m * 1e-3) / 1e12;
            const double vs_sg = sg_m / m * 100.0;
            const double vs_tf = tf_m / m * 100.0;
            std::printf("%-14s %-12s %6d %6d %6d %10.4f %10.4f %9.2f %9.1f %9.1f\n",
                        k.name, sh.tag, sh.M, sh.N, sh.K, m, p, tflops, vs_sg, vs_tf);
            csv << prop.name << "," << k.name << "," << sh.tag << "," << sh.M << "," << sh.N << "," << sh.K
                << "," << iters << "," << m << "," << p << "," << tflops << "," << vs_sg << "," << vs_tf << "," << err << "\n";
        }

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
        CUDA_CHECK(cudaFree(dCref));
    }

    csv.close();
    CURAND_CHECK(curandDestroyGenerator(rng));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
