#ifndef NEURAL_NET_H
#define NEURAL_NET_H

void initialize_network();
float train_sample(const float* input, const float* target);
void free_network();

#endif
