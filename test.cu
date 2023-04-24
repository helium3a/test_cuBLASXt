#include <cblas.h>
#include <chrono>
#include <cublasXt.h>
#include <curand.h>
#include <iostream>
#include <string>

#define DELTA 0.00001f

void checkCuBLASXtError(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        switch (status) {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                std::cout << "CUBLAS_STATUS_NOT_INITIALIZED" << std::endl;
                break;
            case CUBLAS_STATUS_ALLOC_FAILED:
                std::cout << "CUBLAS_STATUS_ALLOC_FAILED" << std::endl;
                break;
            case CUBLAS_STATUS_INVALID_VALUE:
                std::cout << "CUBLAS_STATUS_INVALID_VALUE" << std::endl;
                break;
            case CUBLAS_STATUS_ARCH_MISMATCH:
                std::cout << "CUBLAS_STATUS_ARCH_MISMATCH" << std::endl;
                break;
            case CUBLAS_STATUS_MAPPING_ERROR:
                std::cout << "CUBLAS_STATUS_MAPPING_ERROR" << std::endl;
                break;
            case CUBLAS_STATUS_EXECUTION_FAILED:
                std::cout << "CUBLAS_STATUS_EXECUTION_FAILED" << std::endl;
                break;
            case CUBLAS_STATUS_INTERNAL_ERROR:
                std::cout << "CUBLAS_STATUS_INTERNAL_ERROR" << std::endl;
                break;
            case CUBLAS_STATUS_NOT_SUPPORTED:
                std::cout << "CUBLAS_STATUS_NOT_SUPPORTED" << std::endl;
                break;
            case CUBLAS_STATUS_LICENSE_ERROR:
                std::cout << "CUBLAS_STATUS_LICENSE_ERROR" << std::endl;
                break;
            default:
                std::cout << "Unknown error" << std::endl;
                break;
        }
    }
}

void checkCudaError(cudaError_t err, std::string message) {
        if (err != cudaSuccess) {
            std::cerr << message << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    }

void checkCurandError(curandStatus_t status, std::string message) {
    if (status != CURAND_STATUS_SUCCESS) {
        switch (status) {
            case CURAND_STATUS_VERSION_MISMATCH:
                std::cout << "CURAND_STATUS_VERSION_MISMATCH" << std::endl;
                break;
            case CURAND_STATUS_NOT_INITIALIZED:
                std::cout << "CURAND_STATUS_NOT_INITIALIZED" << std::endl;
                break;
            case CURAND_STATUS_ALLOCATION_FAILED:
                std::cout << "CURAND_STATUS_ALLOCATION_FAILED" << std::endl;
                break;
            case CURAND_STATUS_TYPE_ERROR:
                std::cout << "CURAND_STATUS_TYPE_ERROR" << std::endl;
                break;
            case CURAND_STATUS_OUT_OF_RANGE:
                std::cout << "CURAND_STATUS_OUT_OF_RANGE" << std::endl;
                break;
            case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
                std::cout << "CURAND_STATUS_LENGTH_NOT_MULTIPLE" << std::endl;
                break;
            case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
                std::cout << "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED" << std::endl;
                break;
            case CURAND_STATUS_LAUNCH_FAILURE:
                std::cout << "CURAND_STATUS_LAUNCH_FAILURE" << std::endl;
                break;
            case CURAND_STATUS_PREEXISTING_FAILURE:
                std::cout << "CURAND_STATUS_PREEXISTING_FAILURE" << std::endl;
                break;
            case CURAND_STATUS_INITIALIZATION_FAILED:
                std::cout << "CURAND_STATUS_INITIALIZATION_FAILED" << std::endl;
                break;
            case CURAND_STATUS_ARCH_MISMATCH:
                std::cout << "CURAND_STATUS_ARCH_MISMATCH" << std::endl;
                break;
            case CURAND_STATUS_INTERNAL_ERROR:
                std::cout << "CURAND_STATUS_INTERNAL_ERROR" << std::endl;
                break;
            default:
                std::cout << "Unknown error" << std::endl;
                break;
        }
    }
}

int main() {
    bool print = false;
    int cublasXtblocksize = 1024;
    int num_runs = 1;
    size_t n = 500; // number of features
    size_t k = 800*250*5*10; // number of samples
    float alpha = 1.0f, beta = 0.0f;

    // record initialization time
    auto start_time = std::chrono::high_resolution_clock::now();
    float *d_A;
    checkCudaError(cudaMalloc(&d_A, n * k * sizeof(float)), "cudaMalloc d_A failed: ");
    curandGenerator_t gen;
    checkCurandError(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT), "curandCreateGenerator failed: ");
    checkCurandError(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL), "curandSetPseudoRandomGeneratorSeed failed: ");
    checkCurandError(curandGenerateUniform(gen, d_A, n * k), "curandGenerateUniform failed: ");
    float *A = new float[n * k];
    checkCudaError(cudaMemcpy(A, d_A, n * k * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy d_A to A failed: ");
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    std::cout << "Time to initialize A: " << elapsed_time.count() << " seconds" << std::endl;

    // Initialize output matrix (C)
    float *C = new float[n * n];
    for (int i = 0; i < n * n; i++) {
        C[i] = 0.0f;
    }

    // Initialize output matrix (C1)
    float *C1 = new float[n * n];
    for (int i = 0; i < n * n; i++) {
        C1[i] = 0.0f;
    }

    if (print) {
        // Print input matrix A
        std::cout << "Matrix A:" << std::endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                printf("%15.7f ", A[i * k + j]);
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // Print output matrix C
        std::cout << "Matrix C before cublasXtSsyrk:" << std::endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%15.7f ", C[i * n + j]);
            }
            std::cout << std::endl;
        }

        // Print output matrix C1
        std::cout << "Matrix C before cblas_ssyrk:" << std::endl;    
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%15.7f ", C1[i * n + j]);
            }
            std::cout << std::endl;
        }
    }

    // Initialize cuBLASXt
    start_time = std::chrono::high_resolution_clock::now();
    cublasXtHandle_t handle;
    checkCuBLASXtError(cublasXtCreate(&handle));

    // Set the devices to be used
    int devices[1] = {0};
    // int devices[2] = {0, 1}; // doesn't improve performance
    checkCuBLASXtError(cublasXtDeviceSelect(handle, 1, devices));

    // Get and set tile size
    int tilesize;
    checkCuBLASXtError(cublasXtGetBlockDim(handle, &tilesize));
    printf("Original tile size is %d\n", tilesize);
    checkCuBLASXtError(cublasXtSetBlockDim(handle, cublasXtblocksize));
    checkCuBLASXtError(cublasXtGetBlockDim(handle, &tilesize));
    printf("New tile size is %d\n", tilesize);

    // Perform the symmetric rank-k operation: C = alpha * A * A^T + beta * C
    cublasOperation_t trans = CUBLAS_OP_T;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    for (int i = 0; i < num_runs; i++)
        checkCuBLASXtError(cublasXtSsyrk(handle, uplo, trans, n, k, &alpha, A, k, &beta, C, n));

    // Clean up
    cublasXtDestroy(handle);

    cudaDeviceSynchronize();
    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = end_time - start_time;
    std::cout << "Time to perform cublasXtSsyrk: " << elapsed_time.count() << " seconds" << std::endl;


    // Perform the symmetric rank-k operation: C = alpha * A * A^T + beta * C
    start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; i++)
        cblas_ssyrk(CblasColMajor, CblasLower, CblasTrans, n, k, alpha, A, k, beta, C1, n);
    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = end_time - start_time;
    std::cout << "Time to perform cblas_ssyrk: " << elapsed_time.count() << " seconds" << std::endl;

    if (print) {
        // Print output matrix C
        std::cout << "Matrix C after cublasXtSsyrk:" << std::endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%15.7f ", C[i * n + j]);
            }
            std::cout << std::endl;
        }

        // Print output matrix C
        std::cout << "Matrix C after cblas_ssyrk:" << std::endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%15.7f ", C1[i * n + j]);
            }
            std::cout << std::endl;
        }
    }

    // Print max relative error
    float max_error = 0.0f, val1 = 0.0f, val2 = 0.0f;
    int ii = 0, jj = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            if (std::abs(C1[i * n + j]) < DELTA) {
                continue;
            }
            float error = std::abs(C[i * n + j] - C1[i * n + j]) / std::abs(C1[i * n + j]);
            if (error > max_error) {
                max_error = error;
                val1 = C[i * n + j];
                val2 = C1[i * n + j];
                ii = i;
                jj = j;
            }
        }
    }
    std::cout << "Max relative error: " << max_error << std::endl
                << "val1: " << val1 << std::endl
                << "val2: " << val2 << std::endl
                << "ii: " << ii << std::endl
                << "jj: " << jj << std::endl;
    
    // Clean up
    delete[] A;
    delete[] C;
    delete[] C1;

    return 0;
}