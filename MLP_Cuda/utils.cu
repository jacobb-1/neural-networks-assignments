#include "utils.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void matrix_multiply_kernel(float* A, float* B, float* C, int m, int n, int p, bool transpose_A, bool transpose_B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < p) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            float a_val = transpose_A ? A[k * m + row] : A[row * n + k];
            float b_val = transpose_B ? B[col * n + k] : B[k * p + col];
            sum += a_val * b_val;
        }
        C[row * p + col] = sum;
    }
}

void matrix_multiply(float* d_A, float* d_B, float* d_C, int m, int n, int p, bool transpose_A, bool transpose_B) {
    dim3 blockDim(16, 16);
    dim3 gridDim((p + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);
    matrix_multiply_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, p, transpose_A, transpose_B);
}

__global__ void add_biases_kernel(float* A, float* biases, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / n;
    int col = idx % n;
    if (row < m && col < n) {
        A[idx] += biases[col];
    }
}

void add_biases(float* d_A, float* d_biases, int m, int n) {
    int size = m * n;
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    add_biases_kernel<<<blocks, threadsPerBlock>>>(d_A, d_biases, m, n);
}

__global__ void apply_relu_kernel(float* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = fmaxf(A[idx], 0.0f);
    }
}

void apply_relu(float* d_A, int size) {
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    apply_relu_kernel<<<blocks, threadsPerBlock>>>(d_A, size);
}

__global__ void apply_softmax_kernel(float* A, int batch_size, int output_size) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample < batch_size) {
        float max_val = A[sample * output_size];
        for (int i = 1; i < output_size; i++) {
            if (A[sample * output_size + i] > max_val) max_val = A[sample * output_size + i];
        }
        float sum = 0.0f;
        for (int i = 0; i < output_size; i++) {
            A[sample * output_size + i] = expf(A[sample * output_size + i] - max_val);
            sum += A[sample * output_size + i];
        }
        for (int i = 0; i < output_size; i++) {
            A[sample * output_size + i] /= sum;
        }
    }
}

void apply_softmax(float* d_A, int batch_size, int output_size) {
    int threadsPerBlock = 256;
    int blocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    apply_softmax_kernel<<<blocks, threadsPerBlock>>>(d_A, batch_size, output_size);
}

__global__ void compute_delta_output_kernel(float* output, int* labels, float* delta_output, int batch_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int y = labels[idx];
        for (int k = 0; k < output_size; k++) {
            delta_output[idx * output_size + k] = output[idx * output_size + k];
            if (k == y) delta_output[idx * output_size + k] -= 1.0f;
            delta_output[idx * output_size + k] /= (float)batch_size;
        }
    }
}

void compute_delta_output(float* d_output, int* d_labels, float* d_delta_output, int batch_size, int output_size) {
    int threadsPerBlock = 256;
    int blocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    compute_delta_output_kernel<<<blocks, threadsPerBlock>>>(d_output, d_labels, d_delta_output, batch_size, output_size);
}

__global__ void compute_biases_gradient_kernel(float* delta, float* gradient_biases, int batch_size, int size) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < size) {
        float sum = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            sum += delta[i * size + k];
        }
        gradient_biases[k] = sum;
    }
}

void compute_biases_gradient(float* d_delta, float* d_gradient_biases, int batch_size, int size) {
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    compute_biases_gradient_kernel<<<blocks, threadsPerBlock>>>(d_delta, d_gradient_biases, batch_size, size);
}

__global__ void compute_loss_per_sample_kernel(float* output, int* labels, float* loss_per_sample, int batch_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int y = labels[idx];
        float p_y = output[idx * output_size + y];
        loss_per_sample[idx] = -logf(p_y);
    }
}

void compute_loss_per_sample(float* d_output, int* d_labels, float* d_loss_per_sample, int batch_size, int output_size) {
    int threadsPerBlock = 256;
    int blocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    compute_loss_per_sample_kernel<<<blocks, threadsPerBlock>>>(d_output, d_labels, d_loss_per_sample, batch_size, output_size);
}

__global__ void sum_loss_kernel(float* loss_per_sample, float* total_loss, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        atomicAdd(total_loss, loss_per_sample[idx]);
    }
}

void sum_loss(float* d_loss_per_sample, float* d_total_loss, int batch_size) {
    int threadsPerBlock = 256;
    int blocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    sum_loss_kernel<<<blocks, threadsPerBlock>>>(d_loss_per_sample, d_total_loss, batch_size);
}

__global__ void apply_relu_derivative_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

void apply_relu_derivative(float* d_input, float* d_output, int size) {
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    apply_relu_derivative_kernel<<<blocks, threadsPerBlock>>>(d_input, d_output, size);
}

__global__ void element_wise_multiply_kernel(float* A, float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

void element_wise_multiply(float* d_A, float* d_B, float* d_C, int size) {
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    element_wise_multiply_kernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, size);
}

__global__ void update_weights_kernel(float* weights, float* gradient_weights, int size, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * gradient_weights[idx];
    }
}

__global__ void update_biases_kernel(float* biases, float* gradient_biases, int size, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        biases[idx] -= learning_rate * gradient_biases[idx];
    }
}