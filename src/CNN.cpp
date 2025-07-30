#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <iostream>
#include <random>
#include <math.h>
#include <assert.h>

#include "CNN.h"
#include "functions.h"
#include "cublas_v2.h"
#include "kernels.h"

CNN::CNN(int in_layers, std::vector<int> in_nodes, int in_batch_size){
    layers = in_layers;
    nodes_per_layer = in_nodes;
    batch_size = in_batch_size;

    cublasStatus_t stat = cublasCreate(&handle);
    float *h_Ones;
    h_Ones = new float[batch_size];
    for(int i=0; i<batch_size; i++){
        h_Ones[i] = 1.0;
    }

    cudaMalloc((void**)&d_Ones, sizeof(float)*batch_size);
    cudaMemcpy(d_Ones, h_Ones, sizeof(float)*batch_size, cudaMemcpyHostToDevice);

    h_W = new float* [layers-1];
    h_b = new float* [layers-1];
    d_W = new float* [layers-1];
    d_b = new float* [layers-1];
    d_dW = new float* [layers-1];
    d_db = new float* [layers-1];

    for(int i=0; i<layers; i++){
        h_W[i] = new float [nodes_per_layer[i]*nodes_per_layer[i+1]];
        h_b[i] = new float [nodes_per_layer[i+1]];

        cudaMalloc((void**)&d_W[i], sizeof(float)*nodes_per_layer[i]*nodes_per_layer[i+1]);
        cudaMalloc((void**)&d_b[i], sizeof(float)*nodes_per_layer[i+1]);
        cudaMalloc((void**)&d_dW[i], sizeof(float)*nodes_per_layer[i]*nodes_per_layer[i+1]);
        cudaMalloc((void**)&d_db[i], sizeof(float)*nodes_per_layer[i+1]);
    }

    std::string s = "./data/W_";
    std::uniform_real_distribution<float> uniform(-1.0, 1.0);
    for(int i=0; i<layers-1; i++){
        std::ofstream out_file(s + std::to_string(i) + ".txt");
        for(int j=0; j<nodes_per_layer[i]; j++){
            for(int k=0; k<nodes_per_layer[i+1]; k++){
                h_W[i][IDX(j, k, nodes_per_layer[i])] = uniform(myGenerator());
                out_file << h_W[i][IDX(j, k, nodes_per_layer[i])] << ' ';
            }
            out_file << std::endl;
        }
        out_file.close();
        for(int j=0; j<nodes_per_layer[i+1]; j++){
            h_b[i][j] = 0;
        }
    }

    for(int i=0; i<layers; i++){
        cudaMemcpy(d_W[i], h_W[i], sizeof(float)*nodes_per_layer[i]*nodes_per_layer[i+1], cudaMemcpyHostToDevice);
        cudaMemcpy(d_b[i], h_b[i], sizeof(float)*nodes_per_layer[i+1], cudaMemcpyHostToDevice);
    }

    d_A = new float* [layers];
    for(int i=0; i<layers; i++){
        cudaMalloc((void**)&d_A[i], sizeof(float)*batch_size*nodes_per_layer[i]);
    }

    h_p = new float [batch_size*nodes_per_layer[layers-1]];
    cudaMalloc((void**)&d_p, sizeof(float)*batch_size*nodes_per_layer[layers-1]);

    h_pred_label = new int[batch_size];
    h_label = new int[batch_size];
    cudaMalloc((void**)&d_label, sizeof(int)*batch_size);

    d_delta = new float* [layers-1];
    for(int i=0; i<layers-1; i++){
        cudaMalloc((void**)&d_delta[i], sizeof(float)*batch_size*nodes_per_layer[i+1]);
    }

    delete h_Ones;
};

int CNN::get_layers(){
    return layers;
};

void CNN::set_batch(float *b_data, int *label, int in_batch_size, int in_dim){
    assert(batch_size == in_batch_size);
    assert(nodes_per_layer[0] == in_dim);

    cudaMemcpy(d_A[0], b_data, sizeof(float)*batch_size*nodes_per_layer[0], cudaMemcpyHostToDevice);
    cudaMemcpy(d_label, label, sizeof(int)*batch_size, cudaMemcpyHostToDevice);

    for(int i=0; i<batch_size; i++){
        h_label[i] = label[i];
    }
};

void CNN::feed_forward(){
    alpha = 1.0;

    for(int i=1; i<layers; i++){
    if(!TEST_K){
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batch_size, nodes_per_layer[i],
            nodes_per_layer[i-1], &alpha, d_A[i-1], batch_size, d_W[i-1],
            nodes_per_layer[i-1], &beta, d_A[i], batch_size);
        cublasSger(handle, batch_size, nodes_per_layer[i], &alpha, d_Ones, 1, d_b[i-1],
            1, d_A[i], batch_size);
    }else{
        int h = batch_size, w = nodes_per_layer[i], k = nodes_per_layer[i-1];
        dim3 tileSz(TILE_SIZE, TILE_SIZE);
        dim3 blocks(w + tileSz.x - 1 / tileSz.x, h + tileSz.y - 1 / tileSz.y);
        tiled_matMul_Acc<<<blocks, tileSz>>>(d_A[i-1], d_W[i-1], d_A[i], d_b[i-1], h, w, k);
    }
        //sigmoid is not performed on the last layer
        if(i < layers)
            sigmoid<<<batch_size*nodes_per_layer[i]/512 + 1, 512>>>(d_A[i], batch_size*nodes_per_layer[i]);
    }
    softmax<<<batch_size/512 + 1, 512>>>(d_A[layers-1], d_p, batch_size, nodes_per_layer[layers-1]);
    cudaMemcpy(h_p, d_p, sizeof(float)*batch_size*nodes_per_layer[layers-1], cudaMemcpyDeviceToHost);
};

void CNN::feed_forward_stream(cudaStream_t* stream){
    alpha = 1.0;
    cublasSetStream(handle, *stream);
    for(int i=1; i<layers; i++){
        cublasSetStream(handle, *stream);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batch_size, nodes_per_layer[i],
            nodes_per_layer[i-1], &alpha, d_A[i-1], batch_size, d_W[i-1],
            nodes_per_layer[i-1], &beta, d_A[i], batch_size);
        cublasSetStream(handle, *stream);
        cublasSger(handle, batch_size, nodes_per_layer[i], &alpha, d_Ones, 1, d_b[i-1],
            1, d_A[i], batch_size);
        //sigmoid is not performed on the last layer
        if(i < layers)
            sigmoid<<<batch_size*nodes_per_layer[i]/512 + 1, 512, 0, *stream>>>(d_A[i], batch_size*nodes_per_layer[i]);
    }
    softmax<<<batch_size/512 + 1, 512, 0, *stream>>>(d_A[layers-1], d_p, batch_size, nodes_per_layer[layers-1]);
    cudaStreamSynchronize(0);
    cudaMemcpy(h_p, d_p, sizeof(float)*batch_size*nodes_per_layer[layers-1], cudaMemcpyDeviceToHost);
};

float CNN::compute_cost(){
    float cost=0;
    for(int i=0; i<batch_size; i++){
        cost += -log(h_p[IDX(i, h_label[i], batch_size)]);
    }
    return cost/batch_size;
};

float CNN::compute_accuracy(){
    int correct_labels=0;

    for(int i=0; i<batch_size; i++){
        float max=0;
        h_pred_label[i]=0;
        for(int j=0; j<nodes_per_layer[layers-1]; j++){
            float pred = h_p[IDX(i, j, batch_size)];
            if(pred > max){
                h_pred_label[i] = j;
                max = pred;
            }
        }
        if(h_pred_label[i] == h_label[i]){
            correct_labels++;
        }
    }

    return float(correct_labels)/batch_size;
};

void CNN::back_prop(){
    //softmax derivative
    calc_delta_softmax<<<batch_size/512 + 1, 512>>>(d_label, d_p, batch_size,
        nodes_per_layer[layers-1], d_delta[layers-2]);
    
    for(int i=layers-3; i>=0; i--){
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batch_size, nodes_per_layer[i+1],
            nodes_per_layer[i+2], &alpha, d_delta[i+1], batch_size, d_W[i+1],
            nodes_per_layer[i+1], &beta, d_delta[i], batch_size);
        calc_delta<<<batch_size*nodes_per_layer[i+1]/512, 512>>>(d_delta[i], d_A[i+1],
            batch_size, nodes_per_layer[i+1]);
    }

    //weights and biases derivatives
    alpha = 1.0 / batch_size;
    for(int i=0; i<layers-1; i++){
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, nodes_per_layer[i],
            nodes_per_layer[i+1], batch_size, &alpha, d_A[i], batch_size, d_delta[i],
            batch_size, &beta, d_dW[i], nodes_per_layer[i]);
        cublasSgemv(handle, CUBLAS_OP_T, batch_size, nodes_per_layer[i+1], &alpha,
            d_delta[i], batch_size, d_Ones, 1, &beta, d_db[i], 1);
    }
};

float CNN::train(int num_steps, float eta){
    for(int i=0; i<num_steps; i++){
        feed_forward();
        back_prop();
        for(int j=0; j<layers-1; j++){
            move_one_step<<<nodes_per_layer[j]*nodes_per_layer[j+1]/512 + 1, 512>>>(
                d_W[j], d_dW[j], nodes_per_layer[j]*nodes_per_layer[j+1], eta);
            move_one_step<<<nodes_per_layer[j+1]/512 + 1, 512>>>(d_b[j], d_db[j],
                nodes_per_layer[j+1], eta);
        }
    }
    return compute_cost();
};

float** CNN::get_weights(){
    for(int i=0; i<layers-1; i++){
        cudaMemcpy(h_W[i], d_W[i], sizeof(float)*nodes_per_layer[i]*nodes_per_layer[i+1], cudaMemcpyDeviceToHost);
    }
    return h_W;
};

float** CNN::get_bias(){
    for(int i=0; i<layers-1; i++){
        cudaMemcpy(h_b[i], d_b[i], sizeof(float)*nodes_per_layer[i+1], cudaMemcpyDeviceToHost);
    }
    return h_b;
};

void CNN::set_weights(float **in_weights){
    for(int i=0; i<layers-1; i++){
        cudaMemcpy(d_W[i], in_weights[i], sizeof(float)*nodes_per_layer[i]*nodes_per_layer[i+1], cudaMemcpyHostToDevice);
    }
};

void CNN::set_bias(float **in_bias){
    for(int i=0; i<layers-1; i++){
        cudaMemcpy(d_b[i], in_bias[i], sizeof(float)*nodes_per_layer[i+1], cudaMemcpyHostToDevice);
    }
};
