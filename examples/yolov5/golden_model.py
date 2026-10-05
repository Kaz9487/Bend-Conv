import numpy as np


def numpy_conv2d(x, weight, bias, stride=1, padding=0):
    """FP32 NCHW convolution, im2col + NumPy matmul (no torch inference)."""
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    n, input_channel_count, hi, wi = x.shape
    output_channel_count, _, kh, kw = weight.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    windows = np.lib.stride_tricks.sliding_window_view(xp, (kh, kw), axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride]
    ho, wo = windows.shape[2:4]
    cols = np.ascontiguousarray(windows.transpose(0, 2, 3, 1, 4, 5)).reshape(
        -1, input_channel_count * kh * kw
    )
    out = cols @ weight.reshape(output_channel_count, -1).T
    out += np.asarray(bias, dtype=np.float32)
    return np.ascontiguousarray(out.reshape(n, ho, wo, output_channel_count).transpose(0, 3, 1, 2))


def numpy_concat(inputs, dimension=1):
    """Concatenate inputs along the selected axis."""
    return np.concatenate(inputs, axis=dimension)


def numpy_silu(x):
    x = np.asarray(x, dtype=np.float32)
    with np.errstate(over='ignore'):
        return x / (np.float32(1.0) + np.exp(-x))


def numpy_bottleneck(x, weights_biases, shortcut=True):
    """Apply the two-convolution bottleneck and optional residual."""
    cv1_out = numpy_silu(
        numpy_conv2d(x, weights_biases['cv1_w'], weights_biases['cv1_b'], stride=1, padding=0)
    )
    cv2_out = numpy_silu(
        numpy_conv2d(cv1_out, weights_biases['cv2_w'], weights_biases['cv2_b'], stride=1, padding=1)
    )

    if shortcut and x.shape == cv2_out.shape:
        return x + cv2_out
    else:
        return cv2_out


def numpy_c3(x, weights_biases, n=1, shortcut=True):
    """Apply the C3 block with n bottlenecks and the selected residual policy."""
    # cv1 path
    cv1_out = numpy_silu(
        numpy_conv2d(x, weights_biases['cv1_w'], weights_biases['cv1_b'], stride=1, padding=0)
    )

    # bottleneck path
    bottleneck_path_out = cv1_out
    for i in range(n):
        bottleneck_weights = {
            'cv1_w': weights_biases[f'm{i}_cv1_w'],
            'cv1_b': weights_biases[f'm{i}_cv1_b'],
            'cv2_w': weights_biases[f'm{i}_cv2_w'],
            'cv2_b': weights_biases[f'm{i}_cv2_b'],
        }

        bottleneck_path_out = numpy_bottleneck(
            bottleneck_path_out, bottleneck_weights, shortcut=shortcut
        )

    # cv2 path
    cv2_out = numpy_silu(
        numpy_conv2d(x, weights_biases['cv2_w'], weights_biases['cv2_b'], stride=1, padding=0)
    )

    # concat
    concat_out = numpy_concat([bottleneck_path_out, cv2_out], dimension=1)

    # cv3 path
    cv3_out = numpy_silu(
        numpy_conv2d(
            concat_out, weights_biases['cv3_w'], weights_biases['cv3_b'], stride=1, padding=0
        )
    )

    return cv3_out


def numpy_max_pool(x, kernel_size, stride, padding=0):
    """FP32 maxpool; padded cells are negative infinity, not zero."""
    xp = np.pad(
        x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), constant_values=-np.inf
    )
    windows = np.lib.stride_tricks.sliding_window_view(xp, (kernel_size, kernel_size), axis=(2, 3))
    return np.ascontiguousarray(
        windows[:, :, ::stride, ::stride].max(axis=(-1, -2)), dtype=np.float32
    )


def numpy_sppf(x, weights_biases, k=5):
    """Apply spatial pyramid pooling with three successive max pools."""
    cv1_out = numpy_silu(
        numpy_conv2d(x, weights_biases['cv1_w'], weights_biases['cv1_b'], stride=1, padding=0)
    )

    pool1 = numpy_max_pool(cv1_out, kernel_size=k, stride=1, padding=k // 2)
    pool2 = numpy_max_pool(pool1, kernel_size=k, stride=1, padding=k // 2)
    pool3 = numpy_max_pool(pool2, kernel_size=k, stride=1, padding=k // 2)

    concat_out = numpy_concat([cv1_out, pool1, pool2, pool3], dimension=1)

    cv2_out = numpy_silu(
        numpy_conv2d(
            concat_out, weights_biases['cv2_w'], weights_biases['cv2_b'], stride=1, padding=0
        )
    )

    return cv2_out


def numpy_upsample(x, scale_factor=2, mode='nearest'):
    """Upsample spatial axes using nearest-neighbor repetition."""
    if mode != 'nearest':
        raise NotImplementedError("Only 'nearest' neighbor upsampling is implemented.")

    return x.repeat(scale_factor, axis=2).repeat(scale_factor, axis=3)


def numpy_detect(x, anchors, strides):
    """Decode head tensors with anchors in grid units and strides in pixels."""
    z = []
    for i in range(len(x)):
        feature_map = x[i]
        bs, na, ny, nx, no = feature_map.shape

        stride = strides[i]
        anchor = anchors[i]

        yv, xv = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        grid = np.stack((xv, yv), 2).astype(np.float32)
        grid = grid.reshape(1, 1, ny, nx, 2)

        anchor_grid = (anchor).reshape(1, na, 1, 1, 2)

        # Sigmoid
        y = 1 / (1 + np.exp(-feature_map))

        pred_xy = (y[..., 0:2] * 2 - 0.5 + grid) * stride
        pred_wh = (y[..., 2:4] * 2) ** 2 * anchor_grid * stride

        y_out = np.concatenate((pred_xy, pred_wh, y[..., 4:]), axis=-1)

        z.append(y_out.reshape(bs, -1, no))

    return np.concatenate(z, axis=1)
