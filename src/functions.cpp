#include "functions.h"
#include <random>
std::default_random_engine& myGenerator()
{
  static std::default_random_engine gene;
  return gene;
};

void *launch_kernel(void *in){
  CNN* nn=(CNN*)(in);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  nn->feed_forward_stream(&stream);
}

void launchStreams(uint8_t num_threads, CNN* net){
  pthread_t threads[num_threads];

  for (int i = 0; i < num_threads; i++) {
      if (pthread_create(&threads[i], NULL, launch_kernel, net)) {
          fprintf(stderr, "Error creating threadn");
          return;
      }
  }

  for (int i = 0; i < num_threads; i++) {
      if(pthread_join(threads[i], NULL)) {
          fprintf(stderr, "Error joining threadn");
          return;
      }
  }
};