// Host orchestration only. Math is handwritten in naive_cuda.cu and built by NVRTC.
#include <cuda.h>
#include <nvrtc.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#define CU(call)                                                                                   \
  do {                                                                                             \
    CUresult rc = (call);                                                                          \
    if (rc != CUDA_SUCCESS) {                                                                      \
      const char *msg = NULL;                                                                      \
      cuGetErrorString(rc, &msg);                                                                  \
      fprintf(stderr, "%s: %s\n", #call, msg ? msg : "CUDA error");                                \
      exit(2);                                                                                     \
    }                                                                                              \
  } while (0)
#define NV(call)                                                                                   \
  do {                                                                                             \
    nvrtcResult rc = (call);                                                                       \
    if (rc != NVRTC_SUCCESS) {                                                                     \
      fprintf(stderr, "%s: %s\n", #call, nvrtcGetErrorString(rc));                                 \
      exit(2);                                                                                     \
    }                                                                                              \
  } while (0)
static CUmodule module;
static CUfunction packing_kernel, convolution_kernel, addition_kernel, pooling_kernel,
    upsampling_kernel, decoding_kernel;
static float *allocate_floats(size_t n) {
  CUdeviceptr p;
  CU(cuMemAlloc(&p, n * 4));
  return (float *)(uintptr_t)p;
}
static void device_free(float *p) {
  CU(cuMemFree((CUdeviceptr)(uintptr_t)p));
}
static float *load_floats(const char *path, size_t n) {
  float *p = allocate_floats(n);
  void *host = malloc(n * 4);
  FILE *f = fopen(path, "rb");
  if (!host || !f) {
    perror(path);
    exit(2);
  }
  if (fread(host, 4, n, f) != n || fgetc(f) != EOF)
    exit(2);
  fclose(f);
  CU(cuMemcpyHtoD((CUdeviceptr)(uintptr_t)p, host, n * 4));
  free(host);
  return p;
}
static void dump_tensor(const char *dir, const char *label, const float *p, size_t n) {
  void *host = malloc(n * 4);
  if (!host)
    exit(2);
  CU(cuMemcpyDtoH(host, (CUdeviceptr)(uintptr_t)p, n * 4));
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s.bin", dir, label);
  FILE *f = fopen(path, "wb");
  if (!f) {
    perror(path);
    exit(2);
  }
  if (fwrite(host, 4, n, f) != n)
    exit(2);
  fclose(f);
  free(host);
}
static void device_copy(float *a, const float *b, size_t n) {
  CU(cuMemcpyDtoDAsync((CUdeviceptr)(uintptr_t)a, (CUdeviceptr)(uintptr_t)b, n, NULL));
}
static void launch(CUfunction kernel, int n, void **arguments) {
  CU(cuLaunchKernel(kernel, (n + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, arguments, NULL));
}
static void convolve(const float *input, const float *weights, const float *bias, float *output,
                     float *patches, int input_channel_count, int output_channel_count,
                     int input_height, int input_width, int kernel_size, int stride, int padding,
                     int output_height, int output_width, int activation) {
  int output_positions = output_height * output_width,
      reduction_length = input_channel_count * kernel_size * kernel_size,
      element_count = output_positions * reduction_length;
  void *packing_arguments[] = {&input,  &patches, &input_height, &input_width,      &kernel_size,
                               &stride, &padding, &output_width, &reduction_length, &element_count};
  launch(packing_kernel, element_count, packing_arguments);
  element_count = output_channel_count * output_positions;
  void *kernel_arguments[] = {&patches,          &weights,          &bias,       &output,
                              &output_positions, &reduction_length, &activation, &element_count};
  launch(convolution_kernel, element_count, kernel_arguments);
}
static void add_arrays(const float *left, const float *right, float *output, int element_count) {
  void *arguments[] = {&left, &right, &output, &element_count};
  launch(addition_kernel, element_count, arguments);
}
static void max_pool(const float *input, float *output, int channels, int input_height,
                     int input_width) {
  int element_count = channels * input_height * input_width;
  void *arguments[] = {&input, &output, &input_height, &input_width, &element_count};
  launch(pooling_kernel, element_count, arguments);
}
static void upsample_nearest(const float *input, float *output, int channels, int input_height,
                             int input_width) {
  int element_count = channels * input_height * input_width * 4;
  void *arguments[] = {&input, &output, &input_height, &input_width, &element_count};
  launch(upsampling_kernel, element_count, arguments);
}
static void decode_predictions(const float *input, float *output, int input_height, int input_width,
                               float stride, const float *anchors) {
  int element_count = 3 * input_height * input_width * 85;
  void *arguments[] = {&input,
                       &output,
                       &input_height,
                       &input_width,
                       &stride,
                       (void *)&anchors[0],
                       (void *)&anchors[1],
                       (void *)&anchors[2],
                       (void *)&anchors[3],
                       (void *)&anchors[4],
                       (void *)&anchors[5],
                       &element_count};
  launch(decoding_kernel, element_count, arguments);
}
#define memcpy device_copy
#define free device_free
#ifndef GRAPH_HEADER
#define GRAPH_HEADER "bus_640.h"
#endif
#include GRAPH_HEADER
#undef memcpy
#undef free
static double monotonic_seconds(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
static void compile_kernels(void) {
  FILE *f = fopen("benchmarks/backends/naive_cuda.cu", "rb");
  if (!f)
    exit(2);
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  rewind(f);
  char *src = calloc(n + 1, 1);
  if (!src || fread(src, 1, n, f) != (size_t)n)
    exit(2);
  fclose(f);
  nvrtcProgram prog;
  NV(nvrtcCreateProgram(&prog, src, "naive_cuda.cu", 0, NULL, NULL));
  const char *opts[] = {"--gpu-architecture=compute_89", "--fmad=false"};
  nvrtcResult rc = nvrtcCompileProgram(prog, 2, opts);
  if (rc != NVRTC_SUCCESS) {
    size_t len;
    nvrtcGetProgramLogSize(prog, &len);
    char *log = malloc(len);
    nvrtcGetProgramLog(prog, log);
    fprintf(stderr, "%s\n", log);
    exit(2);
  }
  size_t len;
  NV(nvrtcGetPTXSize(prog, &len));
  char *ptx = malloc(len);
  NV(nvrtcGetPTX(prog, ptx));
  CU(cuModuleLoadData(&module, ptx));
  NV(nvrtcDestroyProgram(&prog));
  free(ptx);
  free(src);
  CU(cuModuleGetFunction(&packing_kernel, module, "col_kernel"));
  CU(cuModuleGetFunction(&convolution_kernel, module, "conv_kernel"));
  CU(cuModuleGetFunction(&addition_kernel, module, "add_kernel"));
  CU(cuModuleGetFunction(&pooling_kernel, module, "pool_kernel"));
  CU(cuModuleGetFunction(&upsampling_kernel, module, "up_kernel"));
  CU(cuModuleGetFunction(&decoding_kernel, module, "decode_kernel"));
}
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  CU(cuInit(0));
  CUdevice device;
  CUcontext ctx;
  CU(cuDeviceGet(&device, 0));
  CU(cuDevicePrimaryCtxRetain(&ctx, device));
  CU(cuCtxSetCurrent(ctx));
  compile_kernels();
  load_graph();
  alloc_graph();
  forward_graph();
  CU(cuCtxSynchronize());
  dump_graph(argv[1]);
  CUevent begin, end;
  CU(cuEventCreate(&begin, 0));
  CU(cuEventCreate(&end, 0));
  double wall[7], gpu[7];
  for (int i = -2; i < 7; i++) {
    CU(cuCtxSynchronize());
    double start = monotonic_seconds();
    CU(cuEventRecord(begin, NULL));
    forward_graph();
    CU(cuEventRecord(end, NULL));
    CU(cuEventSynchronize(end));
    double elapsed = monotonic_seconds() - start;
    float ms;
    CU(cuEventElapsedTime(&ms, begin, end));
    if (i >= 0) {
      wall[i] = elapsed;
      gpu[i] = ms / 1000.0;
    }
  }
  char path[4096];
  snprintf(path, sizeof(path), "%s/timing.json", argv[1]);
  FILE *f = fopen(path, "w");
  if (!f)
    return 2;
  fprintf(f, "{\"backend\":\"handwritten naive "
             "CUDA\",\"warmups\":2,\"preallocated_device_buffers\":true,\"seconds\":[");
  for (int i = 0; i < 7; i++)
    fprintf(f, "%s%.9f", i ? "," : "", wall[i]);
  fprintf(f, "],\"cuda_event_seconds\":[");
  for (int i = 0; i < 7; i++)
    fprintf(f, "%s%.9f", i ? "," : "", gpu[i]);
  fprintf(f, "]}\n");
  fclose(f);
  free_graph();
  return 0;
}
