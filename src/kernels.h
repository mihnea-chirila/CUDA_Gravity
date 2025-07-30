/* @(#)kernels.h
 */

#ifndef _KERNELS_H
#define _KERNELS_H 1

#define TILE_SIZE 16

__global__ void sigmoid(float *A, int n);
__global__ void expM(float *A, int n);
__global__ void softmax(float *A, float* p, int m, int n);
__global__ void calc_delta_softmax(int *label, float* p, int m, int n, float *delta);
__global__ void calc_delta(float *delta, float* A, int m, int n);
__global__ void move_one_step(float *w, float *dw, int n, float eta);
__global__ void tiled_matMul_Acc(float *A, float *B, float *C, float *Bias, int h, int w, int k);
#endif /* _KERNELS_H */

