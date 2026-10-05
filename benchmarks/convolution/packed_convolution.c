#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
static int input_channels, output_channels, height, width, kernel_size, stride, padding,
    output_height, output_width, output_positions, padded_positions, reduction_length, thread_count;
static float *input, *weight, *bias, *packed_input, *output;
static double now_milliseconds(void) {
  struct timespec timestamp;
  clock_gettime(CLOCK_MONOTONIC, &timestamp);
  return timestamp.tv_sec * 1e3 + timestamp.tv_nsec / 1e6;
}
static float *load_floats(const char *name, size_t count) {
  FILE *file = fopen(name, "rb");
  float *values = malloc(count * 4);
  if (!file || !values || fread(values, 4, count, file) != count || fgetc(file) != EOF)
    exit(2);
  fclose(file);
  return values;
}
static void pack_input(void) {
  for (int block_index = 0; block_index < padded_positions / 8; block_index++)
    for (int reduction_index = 0; reduction_index < reduction_length; reduction_index++)
      for (int lane_index = 0; lane_index < 8; lane_index++) {
        int output_position = block_index * 8 + lane_index,
            input_row = output_position / output_width * stride + (reduction_index / kernel_size) % kernel_size - padding,
            input_column = output_position % output_width * stride + reduction_index % kernel_size - padding;
        packed_input[(block_index * reduction_length + reduction_index) * 8 + lane_index] =
            (input_row >= 0 && input_row < height && input_column >= 0 && input_column < width)
                ? input[(reduction_index / (kernel_size * kernel_size) * height + input_row) * width + input_column]
                : 0;
      }
}
static void *compute_channels(void *arg) {
  int worker_index = (int)(intptr_t)arg;
  for (int output_channel = output_channels * worker_index / thread_count; output_channel < output_channels * (worker_index + 1) / thread_count;
       output_channel++)
    for (int block_index = 0; block_index < padded_positions / 8; block_index++) {
      float values[8] = {0};
      for (int reduction_index = 0; reduction_index < reduction_length; reduction_index++)
        for (int lane_index = 0; lane_index < 8; lane_index++)
          values[lane_index] = values[lane_index] + packed_input[(block_index * reduction_length + reduction_index) * 8 + lane_index] *
                            weight[output_channel * reduction_length + reduction_index];
      for (int lane_index = 0; lane_index < 8; lane_index++)
        if (block_index * 8 + lane_index < output_positions)
          output[output_channel * output_positions + block_index * 8 + lane_index] = values[lane_index] + bias[output_channel];
    }
  return NULL;
}
int main(int argc, char **argv) {
  uint32_t configuration[10];
  FILE *file = fopen("config.bin", "rb");
  if (!file || fread(configuration, 4, 10, file) != 10)
    exit(2);
  fclose(file);
  input_channels = configuration[0];
  output_channels = configuration[1];
  height = configuration[2];
  width = configuration[3];
  kernel_size = configuration[4];
  stride = configuration[5];
  padding = configuration[6];
  output_height = configuration[7];
  output_width = configuration[8];
  thread_count = argc > 1 ? atoi(argv[1]) : 1;
  if (thread_count < 1)
    exit(2);
  if (thread_count > output_channels) thread_count = output_channels;
  pthread_t *workers = calloc((size_t)thread_count, sizeof(*workers));
  if (!workers) exit(2);
  output_positions = output_height * output_width;
  padded_positions = (output_positions + 7) / 8 * 8;
  reduction_length = input_channels * kernel_size * kernel_size;
  int repetitions = getenv("CONV_REPS") ? atoi(getenv("CONV_REPS")) : 1;
  for (int repetition = 0; repetition < repetitions; repetition++) {
    input = load_floats("input.bin", input_channels * height * width);
    weight = load_floats("weight.bin", output_channels * reduction_length);
    bias = load_floats("bias.bin", output_channels);
    double packing_start = now_milliseconds();
    packed_input = malloc((size_t)padded_positions * reduction_length * 4);
    output = malloc((size_t)output_channels * output_positions * 4);
    if (!packed_input || !output)
      exit(2);
    pack_input();
    double kernel_start = now_milliseconds();
    if (thread_count == 1)
      compute_channels(0);
    else {
      for (int worker_index = 0; worker_index < thread_count; worker_index++)
        if (pthread_create(workers + worker_index, 0, compute_channels, (void *)(intptr_t)worker_index))
          exit(2);
      for (int worker_index = 0; worker_index < thread_count; worker_index++)
        pthread_join(workers[worker_index], 0);
    }
    double finished = now_milliseconds();
    file = fopen("c.bin", "wb");
    if (!file || fwrite(output, 4, output_channels * output_positions, file) !=
                  (size_t)(output_channels * output_positions))
      exit(2);
    fclose(file);
    file = fopen("c.json", "w");
    fprintf(file, "{\"packing_ms\":%.9f,\"kernel_ms\":%.9f,\"total_ms\":%.9f}\n", kernel_start - packing_start, finished - kernel_start,
            finished - packing_start);
    fclose(file);
    file = fopen("c_samples.jsonl", "a");
    fprintf(file, "{\"packing_ms\":%.9f,\"kernel_ms\":%.9f,\"total_ms\":%.9f}\n", kernel_start - packing_start, finished - kernel_start,
            finished - packing_start);
    fclose(file);
    free(input);
    free(weight);
    free(bias);
    free(packed_input);
    free(output);
  }
  free(workers);
  return 0;
}
