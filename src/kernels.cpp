#include "kernels.h"
#include "functions.h"

__global__ void sigmoid(float *A, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    A[idx] = 1 / (1 + expf(-A[idx]));
  }
};

__global__ void expM(float *A, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    A[idx] = expf(A[idx]);
  }
};

__global__ void softmax(float *A, float* p, int m, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m)
  {
    float max = A[IDX(idx,0,m)];
    for ( int j = 1; j < n; j++)
    {
      if (A[IDX(idx,j,m)] > max)
      {
	max = A[IDX(idx,j,m)];
      }
    }
    float sum = 0;
    for(int j = 0; j < n; j++)
    {
      p[IDX(idx,j,m)] = expf(A[IDX(idx,j,m)] - max);
      sum = sum + p[IDX(idx,j,m)];
    }

    for(int j = 0; j < n; j++)
    {
      p[IDX(idx,j,m)] = p[IDX(idx,j,m)] / sum;
    }
  }
}

__global__ void calc_delta_softmax(int *label, float* p, int m, int n, float* delta)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m)
  {
    for (int j = 0; j < n; j++)
    {
      if (label[idx] == j )
	delta[IDX(idx,j,m)] = 1 - p[IDX(idx,j,m)];
      else
	delta[IDX(idx,j,m)] = - p[IDX(idx,j,m)];
    }
  }
}

__global__ void calc_delta(float *delta, float* A, int m, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m*n)
  {
    delta[idx] = delta[idx] * A[idx] * (1-A[idx]);
  }
}

__global__ void move_one_step(float *w, float *dw, int n, float eta)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    w[idx] = w[idx] + eta * dw[idx];
  }
}

__global__ void tiled_matMul_Acc(float *A, float *B, float *C, float *Bias, int h, int w, int k) {
  __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
  __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;

  int idx_r = threadIdx.y;
  int idx_c = threadIdx.x;

  float res = 0.0f;
  for(int K_tileIdx = 0; K_tileIdx < (k + TILE_SIZE - 1)/TILE_SIZE; K_tileIdx++){
    A_tile[idx_c][idx_r] = ((r<h) && (K_tileIdx * TILE_SIZE + idx_c < k)) ? 
      A[(K_tileIdx * TILE_SIZE + idx_c) * h + r] : 0;
    B_tile[idx_c][idx_r] = ((c<w) && (K_tileIdx * TILE_SIZE + idx_r < k)) ? 
      B[c * k + (idx_r + K_tileIdx * TILE_SIZE)] : 0;
    __syncthreads();

    for(int i=0; i<TILE_SIZE; i++){
      res += A_tile[i][idx_r] * B_tile[idx_c][i];
    }
    __syncthreads();
  }

  if((r < h) && (c < w)){
    C[c * h + r] = res + Bias[c];
  }
}