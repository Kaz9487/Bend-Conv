"""Adapted copy of the existing golden runner; architecture is exported metadata.
Only NumPy is used during inference. PyTorch is confined to export/reference.
"""

import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .golden_model import (
    numpy_conv2d,
    numpy_c3,
    numpy_sppf,
    numpy_concat,
    numpy_upsample,
    numpy_detect,
    numpy_silu,
)


def read_bin_file(file_path, shape=None):
    data = np.fromfile(file_path, dtype='<f4')
    return data.reshape(shape) if shape is not None else data


class NumpyYoloModel:
    def __init__(self, model_dir):
        self.weights_dir = Path(model_dir)
        meta = json.loads((self.weights_dir / 'model.json').read_text())
        self.nc = meta['nc']
        self.strides = np.array(meta['strides'], dtype=np.float32)
        self.anchors = (
            np.array(meta['anchors_pixel'], dtype=np.float32) / self.strides[:, None, None]
        )
        self.weights = {
            name: read_bin_file(self.weights_dir / item['file'], item['shape'])
            for name, item in meta['weights'].items()
        }
        self.model_arch = []
        for spec in meta['layers']:
            spec = dict(spec)
            if 'conv' in spec:
                spec['conv'] = SimpleNamespace(**spec['conv'])
            if spec['type'] == 'C3':
                spec['m'] = [SimpleNamespace(add=spec['shortcut']) for _ in range(spec['n'])]
            elif spec['type'] == 'Detect':
                spec['m'] = [None] * 3
            self.model_arch.append(SimpleNamespace(**spec))

    def _get_c3_weights(self, layer_idx, pt_c3_module):
        """Collect fused weights for a C3 block and its bottlenecks."""
        weights_biases = {}

        weight_map_c3 = {
            'cv1_w': f'model.{layer_idx}.cv1.conv.weight',
            'cv1_b': f'model.{layer_idx}.cv1.conv.bias',
            'cv2_w': f'model.{layer_idx}.cv2.conv.weight',
            'cv2_b': f'model.{layer_idx}.cv2.conv.bias',
            'cv3_w': f'model.{layer_idx}.cv3.conv.weight',
            'cv3_b': f'model.{layer_idx}.cv3.conv.bias',
        }
        for key, weight_name in weight_map_c3.items():
            if weight_name in self.weights:
                weights_biases[key] = self.weights[weight_name]

        for i, m_bottle in enumerate(pt_c3_module.m):
            bottleneck_map = {
                f'm{i}_cv1_w': f'model.{layer_idx}.m.{i}.cv1.conv.weight',
                f'm{i}_cv1_b': f'model.{layer_idx}.m.{i}.cv1.conv.bias',
                f'm{i}_cv2_w': f'model.{layer_idx}.m.{i}.cv2.conv.weight',
                f'm{i}_cv2_b': f'model.{layer_idx}.m.{i}.cv2.conv.bias',
            }
            for key, weight_name in bottleneck_map.items():
                if weight_name in self.weights:
                    weights_biases[key] = self.weights[weight_name]
        return weights_biases

    def _get_sppf_weights(self, layer_idx):
        """Collect fused weights for an SPPF block."""
        return {
            'cv1_w': self.weights[f'model.{layer_idx}.cv1.conv.weight'],
            'cv1_b': self.weights[f'model.{layer_idx}.cv1.conv.bias'],
            'cv2_w': self.weights[f'model.{layer_idx}.cv2.conv.weight'],
            'cv2_b': self.weights[f'model.{layer_idx}.cv2.conv.bias'],
        }

    def forward(
        self, image_np, debug=False, test_vectors_dir=None, return_all_outputs=False, atol=2e-5
    ):
        """Execute the NumPy graph and optionally compare saved intermediate tensors."""
        outputs = []
        x = image_np

        if debug:
            print('--- Running in Layer-wise Verification Mode ---')
            if not test_vectors_dir:
                raise ValueError('test_vectors_dir must be provided in debug mode.')
            test_vectors_dir = Path(test_vectors_dir)

        for i, m in enumerate(self.model_arch):
            module_name = m.type.split('.')[-1]

            if m.f != -1:
                if isinstance(m.f, int):
                    x = outputs[m.f]
                else:
                    x = [x if j == -1 else outputs[j] for j in m.f]

            # --- Module Computation ---
            if module_name == 'Conv':
                w = self.weights[f'model.{i}.conv.weight']
                b = self.weights[f'model.{i}.conv.bias']
                s = self.model_arch[i].conv.stride[0]
                p = self.model_arch[i].conv.padding[0]
                output = numpy_silu(numpy_conv2d(x, w, b, stride=s, padding=p))
            elif module_name == 'C3':
                c3_weights = self._get_c3_weights(i, m)
                n_bottlenecks = len(m.m)
                shortcut = m.m[0].add
                output = numpy_c3(x, c3_weights, n=n_bottlenecks, shortcut=shortcut)
            elif module_name == 'SPPF':
                sppf_weights = self._get_sppf_weights(i)
                output = numpy_sppf(x, sppf_weights)
            elif module_name == 'Upsample':
                output = numpy_upsample(x)
            elif module_name == 'Concat':
                output = numpy_concat(x, dimension=1)
            elif module_name == 'Detect':
                detect_inputs = x
                final_feature_maps = []

                for j, conv_layer in enumerate(m.m):
                    inp = detect_inputs[j]
                    w = self.weights[f'model.{i}.m.{j}.weight']
                    b = self.weights[f'model.{i}.m.{j}.bias']
                    conv_out = numpy_conv2d(inp, w, b, stride=1, padding=0)

                    bs, _, ny, nx = conv_out.shape
                    na = len(self.anchors[0])
                    no = self.nc + 5

                    # Reshape [bs, na*no, ny, nx] -> [bs, na, no, ny, nx] -> [bs, na, ny, nx, no]
                    reshaped_out = conv_out.reshape(bs, na, no, ny, nx).transpose(0, 1, 3, 4, 2)

                    if debug:
                        golden_file = test_vectors_dir / f'layer_{i:02d}_{module_name}_conv_{j}.bin'
                        if golden_file.exists():
                            # Read golden_output in its original shape (bs, na*no, ny, nx)
                            golden_raw_output = read_bin_file(golden_file, conv_out.shape)
                            # Then reshape and transpose golden_raw_output to match reshaped_out
                            golden_reshaped_output = golden_raw_output.reshape(
                                bs, na, no, ny, nx
                            ).transpose(0, 1, 3, 4, 2)

                            print(f'Verifying Layer {i}-{j} ({module_name} sub-layer)... ', end='')
                            if not np.allclose(reshaped_out, golden_reshaped_output, atol=atol):
                                abs_diff = np.abs(reshaped_out - golden_reshaped_output)
                                print(
                                    f'❌ Mismatch! Max diff: {np.max(abs_diff)}, Mean diff: {np.mean(abs_diff)}'
                                )
                                # Optionally, raise an error to stop at first mismatch
                                # raise ValueError(f"Mismatch in Layer {i}-{j} ({module_name} sub-layer)")
                            else:
                                print('✅ Match!')
                        else:
                            print(
                                f'Skipping verification for Layer {i}-{j} ({module_name} sub-layer) (no golden file).'
                            )

                    final_feature_maps.append(reshaped_out)

                # Anchors are in grid units; numpy_detect applies the stride.
                output = numpy_detect(final_feature_maps, self.anchors, self.strides)
            else:
                raise NotImplementedError(
                    f'NumPy implementation for module {module_name} not found.'
                )

            # --- Verification Logic (for intermediate layers) ---
            if debug and module_name != 'Detect':
                golden_file = test_vectors_dir / f'layer_{i:02d}_{module_name}_output.bin'
                if golden_file.exists():
                    golden_output = read_bin_file(golden_file, output.shape)

                    print(f'Verifying Layer {i:02d} ({module_name})... ', end='')
                    if not np.allclose(output, golden_output, atol=atol):
                        abs_diff = np.abs(output - golden_output)
                        print(
                            f'❌ Mismatch! Max diff: {np.max(abs_diff)}, Mean diff: {np.mean(abs_diff)}'
                        )
                        # Optionally, raise an error to stop at first mismatch
                        # raise ValueError(f"Mismatch in Layer {i:02d} ({module_name})")
                    else:
                        print('✅ Match!')
                else:
                    print(
                        f'Skipping verification for Layer {i:02d} ({module_name}) (no golden file).'
                    )
            x = output
            outputs.append(output)

        if return_all_outputs:
            return outputs
        return x
