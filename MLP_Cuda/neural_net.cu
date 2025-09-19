#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include "neural_net.cuh"

#define INPUT_SIZE 784
#define HIDDEN_SIZE 64
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01f

float *W1, *b1, *W2, *b2;

// Utilidades CUDA
__device__ float relu(float x) {
    return x > 0 ? x : 0;
}

__device__ float relu_deriv(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

__global__ void forward_kernel(const float* x, float* z1, float* a1, float* z2, float* output,
                               const float* W1, const float* b1, const float* W2, const float* b2) {
    int tid = threadIdx.x;

    if (tid < HIDDEN_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < INPUT_SIZE; ++i)
            sum += W1[tid * INPUT_SIZE + i] * x[i];
        z1[tid] = sum + b1[tid];
        a1[tid] = relu(z1[tid]);
    }

    __syncthreads();

    if (tid < OUTPUT_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < HIDDEN_SIZE; ++i)
            sum += W2[tid * HIDDEN_SIZE + i] * a1[i];
        z2[tid] = sum + b2[tid];
    }

    __syncthreads();

    if (tid < OUTPUT_SIZE) {
        float max_val = z2[0];
        for (int i = 1; i < OUTPUT_SIZE; ++i)
            max_val = fmaxf(max_val, z2[i]);

        float sum = 0.0f;
        for (int i = 0; i < OUTPUT_SIZE; ++i)
            sum += expf(z2[i] - max_val);
        output[tid] = expf(z2[tid] - max_val) / sum;
    }
}

void initialize_network() {
    W1 = new float[HIDDEN_SIZE * INPUT_SIZE];
    b1 = new float[HIDDEN_SIZE];
    W2 = new float[OUTPUT_SIZE * HIDDEN_SIZE];
    b2 = new float[OUTPUT_SIZE];

    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; ++i)
        W1[i] = ((float) rand() / RAND_MAX - 0.5f) * 0.1f;
    for (int i = 0; i < HIDDEN_SIZE; ++i)
        b1[i] = 0.0f;
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; ++i)
        W2[i] = ((float) rand() / RAND_MAX - 0.5f) * 0.1f;
    for (int i = 0; i < OUTPUT_SIZE; ++i)
        b2[i] = 0.0f;
}

float train_sample(const float* input, const float* target) {
    float *d_input, *d_z1, *d_a1, *d_z2, *d_output;
    float *d_W1, *d_b1, *d_W2, *d_b2;

    cudaMalloc(&d_input, INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_z1, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_a1, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_z2, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(float));

    cudaMalloc(&d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_b1, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_b2, OUTPUT_SIZE * sizeof(float));

    cudaMemcpy(d_input, input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    forward_kernel<<<1, OUTPUT_SIZE>>>(d_input, d_z1, d_a1, d_z2, d_output, d_W1, d_b1, d_W2, d_b2);

    float output[OUTPUT_SIZE];
    cudaMemcpy(output, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    float loss = 0.0f;
    for (int i = 0; i < OUTPUT_SIZE; ++i)
        loss -= target[i] * logf(output[i] + 1e-9f);

    // TODO: Añadir cálculo y aplicación de gradientes en GPU (backward + update)

    cudaFree(d_input); cudaFree(d_z1); cudaFree(d_a1); cudaFree(d_z2); cudaFree(d_output);
    cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);

    return loss;
}

void free_network() {
    delete[] W1; delete[] b1;
    delete[] W2; delete[] b2;
}
