#include "network.h"
#include "utils.h"
#include <cuda_runtime.h>

void init_weights(float** weights, int rows, int cols) {
    *weights = new float[rows * cols];
    // Xavier/Glorot initialization: std = sqrt(2 / (fan_in + fan_out))
    float std_dev = sqrt(2.0f / (rows + cols));
    
    for (int i = 0; i < rows * cols; i++) {
        // Box-Muller transform to generate normal distribution
        float u1 = (std::rand() / (float)RAND_MAX);
        float u2 = (std::rand() / (float)RAND_MAX);
        if (u1 < 1e-6f) u1 = 1e-6f; // avoid log(0)
        
        float z0 = sqrt(-2.0f * log(u1)) * cos(2.0f * 3.14159f * u2);
        (*weights)[i] = z0 * std_dev;
    }
}

void init_biases(float** biases, int size) {
    *biases = new float[size]();
}

void forward_propagation(float* d_input, 
                        float* d_weights_hidden1, float* d_biases_hidden1,
                        float* d_weights_hidden2, float* d_biases_hidden2,
                        float* d_weights_output, float* d_biases_output, 
                        float* d_hidden1, float* d_hidden2, float* d_output,
                        int batch_size, int input_size, int hidden1_size, int hidden2_size, int output_size) {
    // First hidden layer
    matrix_multiply(d_input, d_weights_hidden1, d_hidden1, batch_size, input_size, hidden1_size, false, false);
    add_biases(d_hidden1, d_biases_hidden1, batch_size, hidden1_size);
    apply_relu(d_hidden1, batch_size * hidden1_size);
    
    // Second hidden layer
    matrix_multiply(d_hidden1, d_weights_hidden2, d_hidden2, batch_size, hidden1_size, hidden2_size, false, false);
    add_biases(d_hidden2, d_biases_hidden2, batch_size, hidden2_size);
    apply_relu(d_hidden2, batch_size * hidden2_size);
    
    // Output layer
    matrix_multiply(d_hidden2, d_weights_output, d_output, batch_size, hidden2_size, output_size, false, false);
    add_biases(d_output, d_biases_output, batch_size, output_size);
    apply_softmax(d_output, batch_size, output_size);
}

float compute_loss(float* d_output, int* d_labels, int batch_size, int output_size) {
    float *d_loss_per_sample, *d_total_loss;
    cudaMalloc(&d_loss_per_sample, batch_size * sizeof(float));
    cudaMalloc(&d_total_loss, sizeof(float));
    cudaMemset(d_total_loss, 0, sizeof(float));

    compute_loss_per_sample(d_output, d_labels, d_loss_per_sample, batch_size, output_size);
    sum_loss(d_loss_per_sample, d_total_loss, batch_size);

    float total_loss;
    cudaMemcpy(&total_loss, d_total_loss, sizeof(float), cudaMemcpyDeviceToHost);
    float loss = total_loss / batch_size;

    cudaFree(d_loss_per_sample);
    cudaFree(d_total_loss);
    return loss;
}

void backward_propagation(float* d_input, float* d_hidden1, float* d_hidden2, float* d_output, int* d_labels,
                        float* d_weights_hidden1, float* d_weights_hidden2, float* d_weights_output,
                        float* d_gradient_weights_hidden1, float* d_gradient_biases_hidden1,
                        float* d_gradient_weights_hidden2, float* d_gradient_biases_hidden2,
                        float* d_gradient_weights_output, float* d_gradient_biases_output,
                        int batch_size, int input_size, int hidden1_size, int hidden2_size, int output_size) {
    float *d_delta_output, *d_delta_hidden2, *d_delta_hidden1;
    float *d_temp_hidden2, *d_temp_hidden1;
    
    cudaMalloc(&d_delta_output, batch_size * output_size * sizeof(float));
    cudaMalloc(&d_delta_hidden2, batch_size * hidden2_size * sizeof(float));
    cudaMalloc(&d_delta_hidden1, batch_size * hidden1_size * sizeof(float));
    cudaMalloc(&d_temp_hidden2, batch_size * hidden2_size * sizeof(float));
    cudaMalloc(&d_temp_hidden1, batch_size * hidden1_size * sizeof(float));

    // Compute delta for output layer
    compute_delta_output(d_output, d_labels, d_delta_output, batch_size, output_size);

    // Gradients for output layer
    matrix_multiply(d_hidden2, d_delta_output, d_gradient_weights_output, hidden2_size, batch_size, output_size, true, false);
    compute_biases_gradient(d_delta_output, d_gradient_biases_output, batch_size, output_size);

    // Delta for second hidden layer
    matrix_multiply(d_delta_output, d_weights_output, d_temp_hidden2, batch_size, output_size, hidden2_size, false, true);
    apply_relu_derivative(d_hidden2, d_delta_hidden2, batch_size * hidden2_size);
    element_wise_multiply(d_temp_hidden2, d_delta_hidden2, d_delta_hidden2, batch_size * hidden2_size);

    // Gradients for second hidden layer
    matrix_multiply(d_hidden1, d_delta_hidden2, d_gradient_weights_hidden2, hidden1_size, batch_size, hidden2_size, true, false);
    compute_biases_gradient(d_delta_hidden2, d_gradient_biases_hidden2, batch_size, hidden2_size);

    // Delta for first hidden layer
    matrix_multiply(d_delta_hidden2, d_weights_hidden2, d_temp_hidden1, batch_size, hidden2_size, hidden1_size, false, true);
    apply_relu_derivative(d_hidden1, d_delta_hidden1, batch_size * hidden1_size);
    element_wise_multiply(d_temp_hidden1, d_delta_hidden1, d_delta_hidden1, batch_size * hidden1_size);

    // Gradients for first hidden layer
    matrix_multiply(d_input, d_delta_hidden1, d_gradient_weights_hidden1, input_size, batch_size, hidden1_size, true, false);
    compute_biases_gradient(d_delta_hidden1, d_gradient_biases_hidden1, batch_size, hidden1_size);

    cudaFree(d_delta_output);
    cudaFree(d_delta_hidden2);
    cudaFree(d_delta_hidden1);
    cudaFree(d_temp_hidden2);
    cudaFree(d_temp_hidden1);
}

void update_weights(float* d_weights, float* d_gradient_weights, int size, float learning_rate) {
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    update_weights_kernel<<<blocks, threadsPerBlock>>>(d_weights, d_gradient_weights, size, learning_rate);
}

void update_biases(float* d_biases, float* d_gradient_biases, int size, float learning_rate) {
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    update_biases_kernel<<<blocks, threadsPerBlock>>>(d_biases, d_gradient_biases, size, learning_rate);
}

void forward_propagation_single(float* d_input, float* d_weights_hidden, float* d_biases_hidden,
                             float* d_weights_output, float* d_biases_output, float* d_hidden, float* d_output,
                             int batch_size, int input_size, int hidden_size, int output_size) {
    // Hidden layer
    matrix_multiply(d_input, d_weights_hidden, d_hidden, batch_size, input_size, hidden_size, false, false);
    add_biases(d_hidden, d_biases_hidden, batch_size, hidden_size);
    apply_relu(d_hidden, batch_size * hidden_size);
    
    // Output layer
    matrix_multiply(d_hidden, d_weights_output, d_output, batch_size, hidden_size, output_size, false, false);
    add_biases(d_output, d_biases_output, batch_size, output_size);
    apply_softmax(d_output, batch_size, output_size);
}

void backward_propagation_single(float* d_input, float* d_hidden, float* d_output, int* d_labels,
                              float* d_weights_hidden, float* d_weights_output,
                              float* d_gradient_weights_hidden, float* d_gradient_biases_hidden,
                              float* d_gradient_weights_output, float* d_gradient_biases_output,
                              int batch_size, int input_size, int hidden_size, int output_size) {
    float *d_delta_output, *d_delta_hidden, *d_temp;
    cudaMalloc(&d_delta_output, batch_size * output_size * sizeof(float));
    cudaMalloc(&d_delta_hidden, batch_size * hidden_size * sizeof(float));
    cudaMalloc(&d_temp, batch_size * hidden_size * sizeof(float));

    // Compute delta for output layer
    compute_delta_output(d_output, d_labels, d_delta_output, batch_size, output_size);

    // Gradients for output layer
    matrix_multiply(d_hidden, d_delta_output, d_gradient_weights_output, hidden_size, batch_size, output_size, true, false);
    compute_biases_gradient(d_delta_output, d_gradient_biases_output, batch_size, output_size);

    // Delta for hidden layer
    matrix_multiply(d_delta_output, d_weights_output, d_temp, batch_size, output_size, hidden_size, false, true);
    apply_relu_derivative(d_hidden, d_delta_hidden, batch_size * hidden_size);
    element_wise_multiply(d_temp, d_delta_hidden, d_delta_hidden, batch_size * hidden_size);

    // Gradients for hidden layer
    matrix_multiply(d_input, d_delta_hidden, d_gradient_weights_hidden, input_size, batch_size, hidden_size, true, false);
    compute_biases_gradient(d_delta_hidden, d_gradient_biases_hidden, batch_size, hidden_size);

    cudaFree(d_delta_output);
    cudaFree(d_delta_hidden);
    cudaFree(d_temp);
}