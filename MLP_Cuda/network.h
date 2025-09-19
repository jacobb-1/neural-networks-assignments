#ifndef NETWORK_H
#define NETWORK_H

void init_weights(float** weights, int rows, int cols);
void init_biases(float** biases, int size);

// Two-layer network
void forward_propagation(float* d_input, 
                         float* d_weights_hidden1, float* d_biases_hidden1,
                         float* d_weights_hidden2, float* d_biases_hidden2,
                         float* d_weights_output, float* d_biases_output, 
                         float* d_hidden1, float* d_hidden2, float* d_output,
                         int batch_size, int input_size, int hidden1_size, int hidden2_size, int output_size);

void backward_propagation(float* d_input, float* d_hidden1, float* d_hidden2, float* d_output, int* d_labels,
                          float* d_weights_hidden1, float* d_weights_hidden2, float* d_weights_output,
                          float* d_gradient_weights_hidden1, float* d_gradient_biases_hidden1,
                          float* d_gradient_weights_hidden2, float* d_gradient_biases_hidden2,
                          float* d_gradient_weights_output, float* d_gradient_biases_output,
                          int batch_size, int input_size, int hidden1_size, int hidden2_size, int output_size);

// Single-layer network
void forward_propagation_single(float* d_input, float* d_weights_hidden, float* d_biases_hidden,
                               float* d_weights_output, float* d_biases_output, float* d_hidden, float* d_output,
                               int batch_size, int input_size, int hidden_size, int output_size);

void backward_propagation_single(float* d_input, float* d_hidden, float* d_output, int* d_labels,
                               float* d_weights_hidden, float* d_weights_output,
                               float* d_gradient_weights_hidden, float* d_gradient_biases_hidden,
                               float* d_gradient_weights_output, float* d_gradient_biases_output,
                               int batch_size, int input_size, int hidden_size, int output_size);

float compute_loss(float* d_output, int* d_labels, int batch_size, int output_size);
void update_weights(float* d_weights, float* d_gradient_weights, int size, float learning_rate);
void update_biases(float* d_biases, float* d_gradient_biases, int size, float learning_rate);

#endif