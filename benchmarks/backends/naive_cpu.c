// Handwritten scalar kernels. No BLAS, OpenMP, fast-math, or Bend-generated math.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static float *allocate_floats(size_t n) {
  float *p = malloc(n * sizeof(float));
  if (!p) {
    perror("malloc");
    exit(2);
  }
  return p;
}
static float *load_floats(const char *path, size_t n) {
  float *p = allocate_floats(n);
  FILE *f = fopen(path, "rb");
  if (!f) {
    perror(path);
    exit(2);
  }
  if (fread(p, 4, n, f) != n || fgetc(f) != EOF)
    exit(2);
  fclose(f);
  return p;
}
static void dump_tensor(const char *dir, const char *label, const float *p, size_t n) {
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s.bin", dir, label);
  FILE *f = fopen(path, "wb");
  if (!f) {
    perror(path);
    exit(2);
  }
  if (fwrite(p, 4, n, f) != n)
    exit(2);
  fclose(f);
}
static void convolve(const float *input, const float *weights, const float *bias, float *output,
                     float *patches, int input_channel_count, int output_channel_count,
                     int input_height, int width, int kernel_size, int stride, int padding,
                     int output_height, int output_width, int activation) {
  int reduction_length = input_channel_count * kernel_size * kernel_size,
      output_positions = output_height * output_width;
  for (int output_position = 0; output_position < output_positions; output_position++)
    for (int reduction_index = 0; reduction_index < reduction_length; reduction_index++) {
      int input_row = output_position / output_width * stride +
                      (reduction_index / kernel_size) % kernel_size - padding,
          input_column =
              output_position % output_width * stride + reduction_index % kernel_size - padding;
      patches[output_position * reduction_length + reduction_index] =
          (input_row >= 0 && input_row < input_height && input_column >= 0 && input_column < width)
              ? input[reduction_index / (kernel_size * kernel_size) * input_height * width +
                      input_row * width + input_column]
              : 0.0f;
    }
  for (int output_channel_index = 0; output_channel_index < output_channel_count;
       output_channel_index++)
    for (int output_position = 0; output_position < output_positions; output_position++) {
      float accumulator = 0.0f;
      for (int reduction_index = 0; reduction_index < reduction_length; reduction_index++)
        accumulator =
            accumulator + patches[output_position * reduction_length + reduction_index] *
                              weights[output_channel_index * reduction_length + reduction_index];
      float value = accumulator + bias[output_channel_index];
      output[output_channel_index * output_positions + output_position] =
          activation ? value * (1.0f / (1.0f + expf(-value))) : value;
    }
}
static void add_arrays(const float *left, const float *right, float *output, int count) {
  for (int index = 0; index < count; index++)
    output[index] = left[index] + right[index];
}
static void max_pool(const float *input, float *output, int channels, int input_height,
                     int input_width) {
  for (int channel = 0; channel < channels; channel++)
    for (int input_row = 0; input_row < input_height; input_row++)
      for (int input_column = 0; input_column < input_width; input_column++) {
        float maximum = -INFINITY;
        for (int row_offset = -2; row_offset <= 2; row_offset++)
          for (int column_offset = -2; column_offset <= 2; column_offset++) {
            int sample_row = input_row + row_offset, sample_column = input_column + column_offset;
            if (sample_row >= 0 && sample_row < input_height && sample_column >= 0 &&
                sample_column < input_width)
              maximum = fmaxf(maximum, input[channel * input_height * input_width +
                                             sample_row * input_width + sample_column]);
          }
        output[channel * input_height * input_width + input_row * input_width + input_column] =
            maximum;
      }
}
static void upsample_nearest(const float *input, float *output, int channels, int input_height,
                             int input_width) {
  for (int index = 0; index < channels * input_height * input_width * 4; index++)
    output[index] = input[index / (input_height * input_width * 4) * input_height * input_width +
                          (index / (input_width * 2) % (input_height * 2)) / 2 * input_width +
                          (index % (input_width * 2)) / 2];
}
static void decode_predictions(const float *input, float *output, int input_height, int input_width,
                               float stride, const float *anchors) {
  int output_positions = input_height * input_width;
  for (int index = 0; index < 3 * output_positions * 85; index++) {
    int field = index % 85, output_position = index / 85,
        anchor_index = output_position / output_positions;
    float value = 1.0f / (1.0f + expf(-input[(anchor_index * 85 + field) * output_positions +
                                             output_position % output_positions]));
    if (field == 0)
      value = (value * 2.0f - 0.5f + (float)(output_position % input_width)) * stride;
    else if (field == 1)
      value =
          (value * 2.0f - 0.5f + (float)((output_position / input_width) % input_height)) * stride;
    else if (field < 4)
      value = value * 2.0f * (value * 2.0f) * anchors[anchor_index * 2 + field - 2];
    output[index] = value;
  }
}
#ifndef GRAPH_HEADER
#define GRAPH_HEADER "bus_640.h"
#endif
#include GRAPH_HEADER
static double monotonic_seconds(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: naive-c OUTPUT_DIRECTORY\n");
    return 2;
  }
  const char *dir = argv[1];
  load_graph();
  alloc_graph();
  forward_graph();
  dump_graph(dir);
  free_graph();
  double times[7];
  for (int i = -2; i < 7; i++) {
    double begin = monotonic_seconds();
    alloc_graph();
    forward_graph();
    double elapsed = monotonic_seconds() - begin;
    free_graph();
    if (i >= 0)
      times[i] = elapsed;
  }
  char path[4096];
  snprintf(path, sizeof(path), "%s/timing.json", dir);
  FILE *f = fopen(path, "w");
  if (!f)
    return 2;
  fprintf(
      f, "{\"backend\":\"handwritten naive C "
         "CPU\",\"threads\":1,\"warmups\":2,\"includes_activation_allocation\":true,\"seconds\":[");
  for (int i = 0; i < 7; i++)
    fprintf(f, "%s%.9f", i ? "," : "", times[i]);
  fprintf(f, "]}\n");
  fclose(f);
  return 0;
}
