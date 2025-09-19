#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <cstddef>
#include <cuda_runtime.h>
#include "neural_net.cuh"
#include "mnist_data.h"

// Define constants that should come from neural_net.cuh but let's be explicit
#define INPUT_SIZE 784     // 28x28 pixels for MNIST
#define HIDDEN_SIZE 256    // Hidden layer size
#define OUTPUT_SIZE 10     // 10 digits
#define BATCH_SIZE 32      // Smaller batch size for faster iterations

// Function to load MNIST data - with smaller datasets
bool loadMNISTData(int train_count, std::vector<std::vector<float>>& train_images, 
                  std::vector<std::vector<float>>& train_labels,
                  int test_count, std::vector<std::vector<float>>& test_images, 
                  std::vector<std::vector<float>>& test_labels) {
    
    // Load training data - use smaller amount
    auto train_data = load_limited_mnist_data("mnist_train.csv", train_count);
    train_images = train_data.first;
    train_labels = train_data.second;
    
    // Load test data - use smaller amount
    auto test_data = load_limited_mnist_data("mnist_test.csv", test_count);
    test_images = test_data.first;
    test_labels = test_data.second;
    
    return true;
}

// Function to evaluate the model on test data
float evaluate(const std::vector<std::vector<float>>& test_images, 
               const std::vector<std::vector<float>>& test_labels,
               int num_samples = -1, int batch_size = 32) {
    
    if (num_samples == -1 || num_samples > test_images.size()) {
        num_samples = test_images.size();
    }
    
    int correct = 0;
    int processed = 0;
    
    // Process test data in batches
    for (int batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
        // Determine actual batch size (might be smaller for the last batch)
        int actual_batch_size = std::min(batch_size, num_samples - batch_start);
        
        // Create batch input and target vectors
        std::vector<std::vector<float>> batch_inputs;
        std::vector<std::vector<float>> batch_targets;
        batch_inputs.reserve(actual_batch_size);
        batch_targets.reserve(actual_batch_size);
        
        // Fill batch with data
        for (int i = 0; i < actual_batch_size; ++i) {
            batch_inputs.push_back(test_images[batch_start + i]);
            batch_targets.push_back(test_labels[batch_start + i]);
        }
        
        // Run forward pass on the batch
        forward_pass_batch(batch_inputs, batch_targets, actual_batch_size);
        
        // Get predictions for the whole batch
        float* h_output_data = new float[actual_batch_size * OUTPUT_SIZE];
        cudaMemcpy(h_output_data, d_output, actual_batch_size * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Process each sample in the batch
        for (int i = 0; i < actual_batch_size; ++i) {
            // Find the predicted class
            int predicted = 0;
            for (int j = 1; j < OUTPUT_SIZE; ++j) {
                if (h_output_data[i * OUTPUT_SIZE + j] > h_output_data[i * OUTPUT_SIZE + predicted]) {
                    predicted = j;
                }
            }
            
            // Find the actual class
            int actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; ++j) {
                if (batch_targets[i][j] > 0.5f) {
                    actual = j;
                    break;
                }
            }
            
            if (predicted == actual) {
                correct++;
            }
        }
        
        processed += actual_batch_size;
        delete[] h_output_data;
    }
    
    return static_cast<float>(correct) / processed;
}

int main(int argc, char **argv) {
    // Load MNIST training data with smaller sizes
    std::vector<std::vector<float>> train_images;
    std::vector<std::vector<float>> train_labels;
    std::vector<std::vector<float>> test_images;
    std::vector<std::vector<float>> test_labels;
    
    // Load a much smaller dataset for faster learning
    int train_samples = 5000;  // Use 5000 training samples
    int test_samples = 1000;   // Use 1000 test samples
    
    loadMNISTData(train_samples, train_images, train_labels, test_samples, test_images, test_labels);
    initialize_network();
    
    // Training parameters
    int num_epochs = 50; // Run for 25 epochs
    int num_samples = train_images.size();
    int num_batches = (num_samples + BATCH_SIZE - 1) / BATCH_SIZE;
    int report_interval = 1; // Report every epoch
    
    // Training loop
    for (int epoch = 1; epoch <= num_epochs; epoch++) {
        // Shuffle training data
        std::vector<int> indices(num_samples);
        for (int i = 0; i < num_samples; i++) {
            indices[i] = i;
        }
        std::random_shuffle(indices.begin(), indices.end());
        
        // Train on batches
        float epoch_loss = 0.0f;
        for (int batch = 0; batch < num_batches; batch++) {
            std::vector<std::vector<float>> batch_inputs;
            std::vector<std::vector<float>> batch_targets;
            
            int start_idx = batch * BATCH_SIZE;
            int end_idx = std::min(start_idx + BATCH_SIZE, num_samples);
            int actual_batch_size = end_idx - start_idx;
            
            for (int i = 0; i < actual_batch_size; i++) {
                int idx = indices[start_idx + i];
                batch_inputs.push_back(train_images[idx]);
                batch_targets.push_back(train_labels[idx]);
            }
            
            // Pad batch if needed to reach BATCH_SIZE
            if (actual_batch_size < BATCH_SIZE) {
                for (int i = actual_batch_size; i < BATCH_SIZE; i++) {
                    batch_inputs.push_back(std::vector<float>(INPUT_SIZE, 0.0f));
                    batch_targets.push_back(std::vector<float>(OUTPUT_SIZE, 0.0f));
                }
            }
            
            float batch_loss = train_batch(batch_inputs, batch_targets, BATCH_SIZE);
            epoch_loss += batch_loss * actual_batch_size;
        }
        epoch_loss /= num_samples;
        
        // Calculate test loss and accuracy every epoch
        float test_loss = 0.0f;
        float test_accuracy = evaluate(test_images, test_labels);
        
        // Calculate test loss manually
        int test_batches = (test_images.size() + BATCH_SIZE - 1) / BATCH_SIZE;
        for (int batch = 0; batch < test_batches; batch++) {
            std::vector<std::vector<float>> batch_inputs;
            std::vector<std::vector<float>> batch_targets;
            
            int start_idx = batch * BATCH_SIZE;
            int end_idx = std::min(start_idx + BATCH_SIZE, (int)test_images.size());
            int actual_batch_size = end_idx - start_idx;
            
            for (int i = 0; i < actual_batch_size; i++) {
                batch_inputs.push_back(test_images[start_idx + i]);
                batch_targets.push_back(test_labels[start_idx + i]);
            }
            
            // Pad batch if needed
            if (actual_batch_size < BATCH_SIZE) {
                for (int i = actual_batch_size; i < BATCH_SIZE; i++) {
                    batch_inputs.push_back(std::vector<float>(INPUT_SIZE, 0.0f));
                    batch_targets.push_back(std::vector<float>(OUTPUT_SIZE, 0.0f));
                }
            }
            
            // Forward pass only to get loss
            forward_pass_batch(batch_inputs, batch_targets, BATCH_SIZE);
            float batch_loss = calculate_batch_loss(BATCH_SIZE);
            test_loss += batch_loss * actual_batch_size;
        }
        test_loss /= test_images.size();
        
        // Show train loss, test loss, and test accuracy for each epoch
        std::cout << "Epoch " << epoch << " - Train Loss: " << epoch_loss 
                  << ", Test Loss: " << test_loss 
                  << ", Test Accuracy: " << (test_accuracy * 100.0f) << "%" << std::endl;
    }
    
    // Free resources without any messages
    free_network();
    
    return 0;
}
