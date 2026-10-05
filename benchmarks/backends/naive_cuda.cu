// Handwritten naive CUDA: one thread per output, scalar sequential reduction.
// No shared-memory tiling, Tensor Cores, cuBLAS, cuDNN, or fast math.
extern "C" __global__ void col_kernel(const float *input, float *patches, int input_height,
                                      int input_width, int kernel_size, int stride, int padding,
                                      int output_width, int reduction_length, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  int reduction_index = index % reduction_length, output_position = index / reduction_length,
      input_row = output_position / output_width * stride +
                  (reduction_index / kernel_size) % kernel_size - padding,
      input_column =
          output_position % output_width * stride + reduction_index % kernel_size - padding;
  patches[index] =
      (input_row >= 0 && input_row < input_height && input_column >= 0 &&
       input_column < input_width)
          ? input[reduction_index / (kernel_size * kernel_size) * input_height * input_width +
                  input_row * input_width + input_column]
          : 0.0f;
}
extern "C" __global__ void conv_kernel(const float *patches, const float *weights,
                                       const float *bias, float *output, int output_positions,
                                       int reduction_length, int activation, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  int output_channel_index = index / output_positions, output_position = index % output_positions;
  float accumulator = 0.0f;
  for (int reduction_index = 0; reduction_index < reduction_length; reduction_index++)
    accumulator =
        accumulator + patches[output_position * reduction_length + reduction_index] *
                          weights[output_channel_index * reduction_length + reduction_index];
  float value = accumulator + bias[output_channel_index];
  output[index] = activation ? value * (1.0f / (1.0f + expf(-value))) : value;
}
extern "C" __global__ void add_kernel(const float *left, const float *right, float *output,
                                      int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count)
    output[index] = left[index] + right[index];
}
extern "C" __global__ void pool_kernel(const float *input, float *output, int input_height,
                                       int input_width, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  int channel = index / (input_height * input_width),
      input_row = index / input_width % input_height, input_column = index % input_width;
  float maximum = -1.0f / 0.0f;
  for (int row_offset = -2; row_offset <= 2; row_offset++)
    for (int column_offset = -2; column_offset <= 2; column_offset++) {
      int sample_row = input_row + row_offset, sample_column = input_column + column_offset;
      if (sample_row >= 0 && sample_row < input_height && sample_column >= 0 &&
          sample_column < input_width)
        maximum = fmaxf(
            maximum,
            input[channel * input_height * input_width + sample_row * input_width + sample_column]);
    }
  output[index] = maximum;
}
extern "C" __global__ void up_kernel(const float *input, float *output, int input_height,
                                     int input_width, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count)
    output[index] = input[index / (input_height * input_width * 4) * input_height * input_width +
                          (index / (input_width * 2) % (input_height * 2)) / 2 * input_width +
                          (index % (input_width * 2)) / 2];
}
extern "C" __global__ void decode_kernel(const float *input, float *output, int input_height,
                                         int input_width, float stride, float anchor_0_width,
                                         float anchor_0_height, float anchor_1_width,
                                         float anchor_1_height, float anchor_2_width,
                                         float anchor_2_height, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  int output_positions = input_height * input_width, field = index % 85,
      output_position = index / 85, anchor_index = output_position / output_positions;
  float value = 1.0f / (1.0f + expf(-input[(anchor_index * 85 + field) * output_positions +
                                           output_position % output_positions]));
  if (field == 0)
    value = (value * 2.0f - 0.5f + (float)(output_position % input_width)) * stride;
  else if (field == 1)
    value =
        (value * 2.0f - 0.5f + (float)((output_position / input_width) % input_height)) * stride;
  else if (field < 4) {
    float anchor = field == 2 ? (anchor_index == 0   ? anchor_0_width
                                 : anchor_index == 1 ? anchor_1_width
                                                     : anchor_2_width)
                              : (anchor_index == 0   ? anchor_0_height
                                 : anchor_index == 1 ? anchor_1_height
                                                     : anchor_2_height);
    value = value * 2.0f * (value * 2.0f) * anchor;
  }
  output[index] = value;
}
