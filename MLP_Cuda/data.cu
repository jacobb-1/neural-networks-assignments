#include "data.h"
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

void split_data(float* features, int* labels, int num_samples, int num_features,
                float** train_features, int** train_labels, int* num_train,
                float** test_features, int** test_labels, int* num_test, float split_ratio) {
    *num_train = static_cast<int>(num_samples * split_ratio);
    *num_test = num_samples - *num_train;

    *train_features = new float[*num_train * num_features];
    *train_labels = new int[*num_train];
    *test_features = new float[*num_test * num_features];
    *test_labels = new int[*num_test];

    for (int i = 0; i < *num_train; i++) {
        for (int j = 0; j < num_features; j++) {
            (*train_features)[i * num_features + j] = features[i * num_features + j];
        }
        (*train_labels)[i] = labels[i];
    }
    for (int i = 0; i < *num_test; i++) {
        for (int j = 0; j < num_features; j++) {
            (*test_features)[i * num_features + j] = features[(*num_train + i) * num_features + j];
        }
        (*test_labels)[i] = labels[*num_train + i];
    }
}

void normalize_features(float* features, int num_samples, int num_features) {
    for (int j = 0; j < num_features; j++) {
        float min_val = features[j];
        float max_val = features[j];
        for (int i = 0; i < num_samples; i++) {
            float val = features[i * num_features + j];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
        for (int i = 0; i < num_samples; i++) {
            features[i * num_features + j] = (features[i * num_features + j] - min_val) / (max_val - min_val);
        }
    }
}

void load_mnist_from_files(const char* images_file, const char* labels_file, 
                      float** features, int** labels, int* num_samples, int* num_features) {
    std::cout << "Loading MNIST data from files: " << images_file << " and " << labels_file << std::endl;
    
    // Open images file
    std::ifstream images_stream(images_file);
    if (!images_stream.is_open()) {
        std::cerr << "Error opening images file: " << images_file << std::endl;
        exit(1);
    }
    
    // Open labels file
    std::ifstream labels_stream(labels_file);
    if (!labels_stream.is_open()) {
        std::cerr << "Error opening labels file: " << labels_file << std::endl;
        exit(1);
    }
    
    // Count lines (samples) in the file
    std::string line;
    int count = 0;
    while (std::getline(images_stream, line)) count++;
    images_stream.clear();
    images_stream.seekg(0);
    
    std::cout << "Found " << count << " samples in file" << std::endl;

    // For MNIST: 28x28 = 784 features
    *num_samples = count;
    *num_features = 784;
    *features = new float[count * 784];
    *labels = new int[count];

    // Read labels
    int label_idx = 0;
    while (std::getline(labels_stream, line) && label_idx < count) {
        try {
            (*labels)[label_idx] = std::stoi(line);
        } catch (const std::exception& e) {
            std::cerr << "Error converting label at line " << label_idx << ": " << line << std::endl;
            (*labels)[label_idx] = 0; // Default value
        }
        label_idx++;
    }
    labels_stream.close();
    std::cout << "Loaded " << label_idx << " labels" << std::endl;
    
    // Read image data
    int idx = 0;
    while (std::getline(images_stream, line) && idx < count) {
        std::stringstream ss(line);
        std::string token;
        
        // Read all 784 pixel values
        for (int i = 0; i < 784; i++) {
            if (std::getline(ss, token, ',')) {
                try {
                    (*features)[idx * 784 + i] = std::stof(token) / 255.0f; // Normalize pixel values to 0-1
                } catch (const std::exception& e) {
                    std::cerr << "Error converting pixel at sample " << idx << ", pixel " << i << std::endl;
                    (*features)[idx * 784 + i] = 0.0f; // Default value
                }
            } else {
                (*features)[idx * 784 + i] = 0.0f; // Missing values
            }
        }
        
        idx++;
    }
    images_stream.close();
    
    std::cout << "Loaded " << idx << " image samples" << std::endl;
}