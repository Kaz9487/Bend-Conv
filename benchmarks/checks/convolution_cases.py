"""Existing small FP32 oracle cases for compiler/runtime integration."""

import numpy as np


def convolution_cases():
    cases = []
    rng = np.random.default_rng(918)

    def case(name, shape, x=None, w=None, b=None):
        input_channel_count, output_channel_count, h, wi, k, s, p = shape
        if x is None:
            x = (rng.integers(-8, 9, input_channel_count * h * wi) / 4).astype(np.float32).tolist()
        if w is None:
            w = (
                (rng.integers(-8, 9, output_channel_count * input_channel_count * k * k) / 4)
                .astype(np.float32)
                .tolist()
            )
        if b is None:
            b = (rng.integers(-4, 5, output_channel_count) / 4).astype(np.float32).tolist()
        cases.append(dict(name=name, shape=shape, x=x, w=w, b=b))

    case('pointwise', [1, 1, 1, 1, 1, 1, 0], [2.0], [3.0], [4.0])
    case(
        'ordered_cancellation_bias_last',
        [3, 1, 1, 1, 1, 1, 0],
        [16777216.0, 1.0, -16777216.0],
        [1.0] * 3,
        [1.0],
    )
    case('signed_zero', [1, 1, 1, 1, 1, 1, 0], [-0.0], [1.0], [-0.0])
    for shape in [
        [2, 3, 2, 3, 1, 1, 0],
        [2, 2, 3, 4, 3, 1, 1],
        [1, 2, 4, 5, 3, 2, 1],
        [1, 1, 3, 3, 3, 1, 0],
        [1, 1, 1, 2, 3, 1, 2],
        [1, 2, 4, 4, 2, 2, 0],
        [2, 1, 2, 3, 1, 2, 0],
    ]:
        case('shape_' + '_'.join(map(str, shape)), shape)
    # Rejection is proved for every problem; one case per kind checks the compiled guard.
    case('zero_dimension', [0, 1, 2, 2, 1, 1, 0])
    case('kernel_too_large', [1, 1, 1, 1, 3, 1, 0])
    case('short_input', [1, 1, 2, 2, 1, 1, 0], [1.0] * 3)
    rng = np.random.default_rng(20260919)
    for input_channel_count, output_channel_count, h, w, k, stride, pad in [
        (3, 5, 4, 7, 3, 1, 1),
        (5, 3, 5, 6, 3, 2, 1),
        (2, 5, 2, 9, 1, 2, 2),
        (3, 3, 3, 4, 2, 1, 0),
        (1, 2, 2, 2, 5, 1, 3),
    ]:
        shape = [input_channel_count, output_channel_count, h, w, k, stride, pad]
        case(
            'rounding_' + '_'.join(map(str, shape)),
            shape,
            rng.normal(size=input_channel_count * h * w).astype(np.float32).tolist(),
            rng.normal(size=output_channel_count * input_channel_count * k * k)
            .astype(np.float32)
            .tolist(),
            rng.normal(size=output_channel_count).astype(np.float32).tolist(),
        )
    return cases


def oracle(c):
    input_channel_count, output_channel_count, h, wi, k, s, p = c['shape']
    x, w, b = c['x'], c['w'], c['b']
    valid = (
        min(input_channel_count, output_channel_count, h, wi, k, s) > 0
        and h + 2 * p >= k
        and wi + 2 * p >= k
        and len(x) == input_channel_count * h * wi
        and len(w) == output_channel_count * input_channel_count * k * k
        and len(b) == output_channel_count
    )
    if not valid:
        return dict(accepted=False, height=0, width=0, bits=[])
    output_height = (h + 2 * p - k) // s + 1
    output_width = (wi + 2 * p - k) // s + 1
    values = []
    for output_channel_index in range(output_channel_count):
        for output_row_index in range(output_height):
            for output_column_index in range(output_width):
                acc = np.float32(0)
                for input_channel_index in range(input_channel_count):
                    for kernel_row_index in range(k):
                        for kernel_column_index in range(k):
                            iy = output_row_index * s + kernel_row_index - p
                            ix = output_column_index * s + kernel_column_index - p
                            sample = (
                                np.float32(x[(input_channel_index * h + iy) * wi + ix])
                                if 0 <= iy < h and 0 <= ix < wi
                                else np.float32(0)
                            )
                            product = np.float32(
                                sample
                                * np.float32(
                                    w[
                                        (
                                            (
                                                output_channel_index * input_channel_count
                                                + input_channel_index
                                            )
                                            * k
                                            + kernel_row_index
                                        )
                                        * k
                                        + kernel_column_index
                                    ]
                                )
                            )
                            acc = np.float32(acc + product)
                values.append(np.float32(acc + np.float32(b[output_channel_index])))
    bits = np.asarray(values, np.float32).view(np.uint32).tolist()
    return dict(accepted=True, height=output_height, width=output_width, bits=bits)


def flist(xs):
    def literal(x):
        return 'F32.neg(' + repr(abs(float(x))) + ')' if np.signbit(x) else repr(float(x))

    return ''.join(literal(x) + ' <> ' for x in xs) + 'Nil{}'
