//CNN.h
#ifndef _CNN_H
#define _CNN_H
#include <vector>
#include "cublas_v2.h"

#define TEST_K 1

class CNN{
    int layers;
    std::vector<int> nodes_per_layer;
    int batch_size;
    int *d_label; //label on device
    int *h_label; //label on host
    float **d_W, **d_b; //weights and biases on device
    float **h_W, **h_b; //weights and biases on host
    float **d_A; //per neuron activation value
    float *h_p, *d_p; //softmax probability
    int *h_pred_label; //predicted label
    float *d_Ones; //1s vector

    float **d_dW; //device weights derivative
    float **d_db; //device bias derivative
    float **d_delta; //input derivative
    
    float alpha = 1.0;
    float beta = 0.0;

    cublasHandle_t handle;

  public:
    CNN(int in_layers, std::vector<int> in_nodes, int in_batch_size);
    int get_layers();
    void set_batch(float *b_data, int *label, int in_batch_size, int in_dim);
    float compute_cost();
    float compute_accuracy();

    void feed_forward();
    void feed_forward_stream(cudaStream_t*);
    void back_prop();
    float train(int num_steps, float eta);

    float** get_weights();
    float** get_bias();

    void set_weights(float **in_weights);
    void set_bias(float **in_bias);
};
#endif