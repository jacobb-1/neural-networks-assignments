#ifndef UTILS_H
#define UTILS_H

void matrix_multiply(float* d_A, float* d_B, float* d_C, int m, int n, int p, bool transpose_A, bool transpose_B);
void add_biases(float* d_A, float* d_biases, int m, int n);
void apply_relu(float* d_A, int size);
void apply_softmax(float* d_A, int batch_size, int output_size);
void compute_delta_output(float* d_output, int* d_labels, float* d_delta_output, int batch_size, int output_size);
void compute_biases_gradient(float* d_delta, float* d_gradient_biases, int batch_size, int size);
void compute_loss_per_sample(float* d_output, int* d_labels, float* d_loss_per_sample, int batch_size, int output_size);
void sum_loss(float* d_loss_per_sample, float* d_total_loss, int batch_size);
void apply_relu_derivative(float* d_input, float* d_output, int size);
void element_wise_multiply(float* d_A, float* d_B, float* d_C, int size);

__global__ void update_weights_kernel(float* weights, float* gradient_weights, int size, float learning_rate);
__global__ void update_biases_kernel(float* biases, float* gradient_biases, int size, float learning_rate);

#endif