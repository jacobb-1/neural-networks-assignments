#ifndef DATA_H
#define DATA_H

void load_mnist_from_files(const char* images_file, const char* labels_file, 
                      float** features, int** labels, int* num_samples, int* num_features);
void split_data(float* features, int* labels, int num_samples, int num_features,
                float** train_features, int** train_labels, int* num_train,
                float** test_features, int** test_labels, int* num_test, float split_ratio);
void normalize_features(float* features, int num_samples, int num_features);

#endif