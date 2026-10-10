// FP32 results of the elementwise mathematical functions against MPFR, over
// consecutive 32-bit input words.
//   math_accuracy input BASE COUNT FILE
//   math_accuracy check FUNCTION BASE COUNT INPUT OUTPUT
#include <float.h>
#include <math.h>
#include <mpfr.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PRECISION 128

static float from_word(uint32_t word) {
  float value;
  memcpy(&value, &word, 4);
  return value;
}

static uint32_t to_word(float value) {
  uint32_t word;
  memcpy(&word, &value, 4);
  return word;
}

// Position of a float among the representable values; the two zeros share one.
static int64_t position(float value) {
  uint32_t word = to_word(value);
  int64_t magnitude = word & 0x7fffffffu;
  return word >> 31 ? -magnitude : magnitude;
}

static mpfr_t scale, cubic, scratch;

// 1 / (1 + exp(-t))
static void logistic(mpfr_t result, mpfr_t t) {
  mpfr_neg(scratch, t, MPFR_RNDN);
  mpfr_exp(scratch, scratch, MPFR_RNDN);
  mpfr_add_ui(scratch, scratch, 1, MPFR_RNDN);
  mpfr_ui_div(result, 1, scratch, MPFR_RNDN);
}

// The real function each name stands for. gelu_tanh is PyTorch's
// 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))), written as x logistic(2 t).
static void reference(mpfr_t result, mpfr_t x, const char *name) {
  if (!strcmp(name, "exp")) mpfr_exp(result, x, MPFR_RNDN);
  else if (!strcmp(name, "log")) mpfr_log(result, x, MPFR_RNDN);
  else if (!strcmp(name, "tanh")) mpfr_tanh(result, x, MPFR_RNDN);
  else if (!strcmp(name, "sigmoid")) logistic(result, x);
  else if (!strcmp(name, "silu")) {
    logistic(result, x);
    mpfr_mul(result, result, x, MPFR_RNDN);
  } else if (!strcmp(name, "gelu_tanh")) {
    mpfr_pow_ui(result, x, 3, MPFR_RNDN);
    mpfr_mul(result, result, cubic, MPFR_RNDN);
    mpfr_add(result, result, x, MPFR_RNDN);
    mpfr_mul(result, result, scale, MPFR_RNDN);
    mpfr_mul_2ui(result, result, 1, MPFR_RNDN);
    logistic(result, result);
    mpfr_mul(result, result, x, MPFR_RNDN);
  } else {
    fprintf(stderr, "unknown function %s\n", name);
    exit(2);
  }
}

static float *read_floats(const char *path, size_t count) {
  float *values = malloc(count * sizeof(float));
  FILE *file = fopen(path, "rb");
  if (!values || !file || fread(values, sizeof(float), count, file) != count) {
    fprintf(stderr, "cannot read %zu values from %s\n", count, path);
    exit(2);
  }
  fclose(file);
  return values;
}

int main(int argc, char **argv) {
  if (argc == 5 && !strcmp(argv[1], "input")) {
    uint32_t base = (uint32_t)strtoull(argv[2], 0, 0);
    size_t count = strtoull(argv[3], 0, 0);
    FILE *file = fopen(argv[4], "wb");
    for (size_t index = 0; index < count; index++) {
      uint32_t word = base + (uint32_t)index;
      fwrite(&word, 4, 1, file);
    }
    return fclose(file) != 0;
  }
  if (argc != 7 || strcmp(argv[1], "check")) {
    fprintf(stderr, "usage: math_accuracy input BASE COUNT FILE | check FUNCTION BASE COUNT INPUT OUTPUT\n");
    return 2;
  }
  const char *name = argv[2];
  uint32_t base = (uint32_t)strtoull(argv[3], 0, 0);
  size_t count = strtoull(argv[4], 0, 0);
  float *inputs = read_floats(argv[5], count);
  float *outputs = read_floats(argv[6], count);
  // silu and gelu_tanh are 0 times infinity at an infinite input.
  int product = !strcmp(name, "silu") || !strcmp(name, "gelu_tanh");

  mpfr_set_default_prec(PRECISION);
  mpfr_t x, value, nudge, error;
  mpfr_inits(x, value, nudge, error, scale, cubic, scratch, (mpfr_ptr)0);
  mpfr_const_pi(scale, MPFR_RNDN);
  mpfr_ui_div(scale, 2, scale, MPFR_RNDN);
  mpfr_sqrt(scale, scale, MPFR_RNDN);
  mpfr_set_str(cubic, "0.044715", 10, MPFR_RNDN);

  uint64_t finite = 0, steps[5] = {0}, undecided = 0, nan_wrong = 0, infinity_wrong = 0, zero_sign_wrong = 0;
  double largest = 0;
  uint32_t largest_word = base, plus_infinity = 0, minus_infinity = 0;

  for (size_t index = 0; index < count; index++) {
    uint32_t word = base + (uint32_t)index;
    float input = inputs[index], output = outputs[index];
    if (word != to_word(input)) {
      fprintf(stderr, "input file does not hold word %u at %zu\n", word, index);
      return 2;
    }
    if (isnan(input)) {
      nan_wrong += !isnan(output);
      continue;
    }
    if (isinf(input)) {
      if (input > 0) plus_infinity = to_word(output); else minus_infinity = to_word(output);
      if (product) continue;
    }
    mpfr_set_flt(x, input, MPFR_RNDN);
    reference(value, x, name);
    if (mpfr_nan_p(value)) {
      nan_wrong += !isnan(output);
      continue;
    }
    if (isnan(output)) {
      nan_wrong++;
      continue;
    }
    float rounded = mpfr_get_flt(value, MPFR_RNDN);
    if (mpfr_regular_p(value)) {
      // The reference is decided when both ends of a far wider interval than
      // the working error round to the same float.
      mpfr_mul_2si(nudge, value, -100, MPFR_RNDN);
      mpfr_add(error, value, nudge, MPFR_RNDN);
      float above = mpfr_get_flt(error, MPFR_RNDN);
      mpfr_sub(error, value, nudge, MPFR_RNDN);
      float below = mpfr_get_flt(error, MPFR_RNDN);
      undecided += to_word(above) != to_word(rounded) || to_word(below) != to_word(rounded);
    }
    if (isinf(rounded) || isinf(output)) {
      infinity_wrong += to_word(rounded) != to_word(output);
      continue;
    }
    finite++;
    int64_t distance = llabs(position(output) - position(rounded));
    steps[distance < 4 ? distance : 4]++;
    zero_sign_wrong += rounded == 0 && output == 0 && to_word(rounded) != to_word(output);
    // One unit in the last place of the correctly rounded result.
    int exponent = -149;
    if (fabsf(rounded) >= FLT_MIN) {
      frexpf(rounded, &exponent);
      exponent -= 24;
    }
    mpfr_sub_d(error, value, (double)output, MPFR_RNDN);
    mpfr_abs(error, error, MPFR_RNDN);
    mpfr_mul_2si(error, error, -exponent, MPFR_RNDN);
    double units = mpfr_get_d(error, MPFR_RNDU);
    if (units > largest) {
      largest = units;
      largest_word = word;
    }
  }
  printf("{\"function\":\"%s\",\"base\":%u,\"count\":%zu,\"finite\":%llu,\"largest_ulp\":%.9f,\"largest_word\":%u,"
         "\"steps\":[%llu,%llu,%llu,%llu,%llu],\"undecided\":%llu,\"nan_wrong\":%llu,\"infinity_wrong\":%llu,"
         "\"zero_sign_wrong\":%llu,\"plus_infinity\":%u,\"minus_infinity\":%u}\n",
         name, base, count, (unsigned long long)finite, largest, largest_word,
         (unsigned long long)steps[0], (unsigned long long)steps[1], (unsigned long long)steps[2],
         (unsigned long long)steps[3], (unsigned long long)steps[4], (unsigned long long)undecided,
         (unsigned long long)nan_wrong, (unsigned long long)infinity_wrong, (unsigned long long)zero_sign_wrong,
         plus_infinity, minus_infinity);
  return 0;
}
