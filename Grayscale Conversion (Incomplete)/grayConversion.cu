// THIS CODE DOES NOT WORK AND IS IN NEED OF HEAVY EDITING

#include <cuda.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#define CHANNELS 3

__global__ void colortoGrayscaleConvertion(unsigned char *Pout,
                                           unsigned char *Pin, int width,
                                           int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    int grayOffset = row * width + col;

    int rgbOffset = grayOffset * CHANNELS;
    unsigned char r = Pin[rgbOffset];
    unsigned char g = Pin[rgbOffset + 1];
    unsigned char b = Pin[rgbOffset + 2];

    Pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
  }
}

int main(int argc, char **argv) {

  std::string input_file;
  std::string output_file;

  // Check for the input file and output file names
  switch (argc) {
  case 3:
    input_file = std::string(argv[1]);
    output_file = std::string(argv[2]);
    break;
  default:
    std::cerr << "Usage: <executable> input_file output_file";
    exit(1);
  }

  int width = 1920;
  int height = 1080;
  int numPixels = width * height;

  unsigned char *in_h = new unsigned char[numPixels * CHANNELS];
  unsigned char *out_h = new unsigned char[numPixels];

  unsigned char *in_d, *out_d;
  cudaMalloc((void **)&in_d, numPixels * CHANNELS);
  cudaMalloc((void **)&out_d, numPixels);

  cudaMemcpy(in_d, in_h, numPixels * CHANNELS, cudaMemcpyHostToDevice);
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  colortoGrayscaleConvertion<<<grid, block>>>(out_d, in_d, width, height);
  cudaDeviceSynchronize();
  cudaMemcpy(out_h, out_d, numPixels, cudaMemcpyDeviceToHost);
}
