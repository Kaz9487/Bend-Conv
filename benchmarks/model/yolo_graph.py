"""One YOLOv5n operation graph shared by the Bend and C/CUDA emitters."""

import math


def input_slots(operation):
    """Ordered reads, including repeated operands, for every supported operation."""
    kind = operation['kind']
    if kind == 'conv':
        return [operation[key] for key in ['input', 'weights', 'bias']]
    if kind == 'concat':
        return operation['inputs']
    if kind == 'add':
        return [operation['left'], operation['right']]
    if kind in ['pool', 'up', 'decode']:
        return [operation['input']]
    raise ValueError(f'Unknown graph operation: {kind}')


def build_graph(model, reference, case):
    assert model['sha256'] == reference['weights_sha256'], 'Model/input weights differ'
    assert model['nc'] == 80 and reference['input_shape'][0] == 1
    tensors, operations, snapshots, weights = [], [], {}, {}

    def add_tensor(shape, path=None):
        index = len(tensors)
        tensors.append(dict(shape=shape, count=math.prod(shape), path=path))
        return index

    for name, weight in model['weights'].items():
        weights[name] = add_tensor(weight['shape'], 'out/weights/' + weight['file'])
    input_id = add_tensor(reference['input_shape'], f'out/results/{case}/input.bin')

    def convolution(input_id, prefix, stride=1, padding=0, activation=True):
        weight_id, bias_id = weights[prefix + '.weight'], weights[prefix + '.bias']
        output_channels, input_channels, kernel, kernel_width = tensors[weight_id]['shape']
        batch, channels, height, width = tensors[input_id]['shape']
        assert batch == 1 and channels == input_channels and kernel == kernel_width
        output_height = (height + 2 * padding - kernel) // stride + 1
        output_width = (width + 2 * padding - kernel) // stride + 1
        output_id = add_tensor([1, output_channels, output_height, output_width])
        operations.append(
            dict(
                kind='conv',
                input=input_id,
                weights=weight_id,
                bias=bias_id,
                output=output_id,
                input_channels=input_channels,
                output_channels=output_channels,
                input_height=height,
                input_width=width,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                output_height=output_height,
                output_width=output_width,
                activation=int(activation),
                prefix=prefix,
            )
        )
        return output_id

    def concatenate(inputs, shape=None):
        if shape is None:
            batch, _, height, width = tensors[inputs[0]]['shape']
            assert all(
                tensors[index]['shape'][0] == batch
                and tensors[index]['shape'][2:] == [height, width]
                for index in inputs
            )
            shape = [batch, sum(tensors[index]['shape'][1] for index in inputs), height, width]
        assert math.prod(shape) == sum(tensors[index]['count'] for index in inputs)
        output_id = add_tensor(shape)
        operations.append(dict(kind='concat', inputs=inputs, output=output_id))
        return output_id

    def residual_add(left, right):
        assert tensors[left]['shape'] == tensors[right]['shape']
        output_id = add_tensor(tensors[left]['shape'])
        operations.append(dict(kind='add', left=left, right=right, output=output_id))
        return output_id

    def max_pool(input_id):
        output_id = add_tensor(tensors[input_id]['shape'])
        operations.append(dict(kind='pool', input=input_id, output=output_id))
        return output_id

    def upsample(input_id):
        batch, channels, height, width = tensors[input_id]['shape']
        output_id = add_tensor([batch, channels, height * 2, width * 2])
        operations.append(dict(kind='up', input=input_id, output=output_id))
        return output_id

    def c3_block(input_id, prefix, repetitions, shortcut):
        left = convolution(input_id, prefix + '.cv1.conv')
        right = convolution(input_id, prefix + '.cv2.conv')
        for index in range(repetitions):
            residual = left
            left = convolution(left, prefix + f'.m.{index}.cv1.conv')
            left = convolution(left, prefix + f'.m.{index}.cv2.conv', 1, 1)
            if shortcut:
                left = residual_add(left, residual)
        return convolution(concatenate([left, right]), prefix + '.cv3.conv')

    def sppf_block(input_id, prefix):
        first = convolution(input_id, prefix + '.cv1.conv')
        second = max_pool(first)
        third = max_pool(second)
        fourth = max_pool(third)
        return convolution(concatenate([first, second, third, fourth]), prefix + '.cv2.conv')

    def detection_head(inputs, prefix):
        decoded = []
        for index, input_id in enumerate(inputs):
            raw = convolution(input_id, prefix + f'.m.{index}', activation=False)
            snapshots[f'head_{index}'] = raw
            height, width = tensors[raw]['shape'][-2:]
            output_id = add_tensor([1, 3 * height * width, 85])
            operations.append(
                dict(
                    kind='decode',
                    input=raw,
                    output=output_id,
                    input_height=height,
                    input_width=width,
                    stride=model['strides'][index],
                    anchors=model['anchors_pixel'][index],
                )
            )
            decoded.append(output_id)
        return concatenate(decoded, [1, sum(tensors[index]['shape'][1] for index in decoded), 85])

    layer_outputs = []
    current = input_id
    for layer in model['layers']:
        incoming = layer['f']
        if isinstance(incoming, list):
            inputs = [current if index == -1 else layer_outputs[index] for index in incoming]
        else:
            inputs = current if incoming == -1 else layer_outputs[incoming]
        prefix, kind = 'model.' + str(layer['i']), layer['type']
        if kind == 'Conv':
            current = convolution(
                inputs, prefix + '.conv', layer['conv']['stride'][0], layer['conv']['padding'][0]
            )
        elif kind == 'C3':
            current = c3_block(inputs, prefix, layer['n'], layer['shortcut'])
        elif kind == 'SPPF':
            current = sppf_block(inputs, prefix)
        elif kind == 'Upsample':
            current = upsample(inputs)
        elif kind == 'Concat':
            current = concatenate(inputs)
        elif kind == 'Detect':
            current = detection_head(inputs, prefix)
        else:
            raise ValueError(f'Unsupported layer: {kind}')
        layer_outputs.append(current)
        snapshots['pred' if kind == 'Detect' else f'layer_{layer["i"]:02}'] = current

    assert {name: tensors[index]['shape'] for name, index in snapshots.items()} == reference[
        'outputs'
    ]
    macs = sum(
        op['output_channels']
        * op['output_height']
        * op['output_width']
        * op['input_channels']
        * op['kernel_size'] ** 2
        for op in operations
        if op['kind'] == 'conv'
    )
    return dict(
        case=case,
        tensors=tensors,
        ops=operations,
        snapshots=snapshots,
        input=input_id,
        pred=current,
        weights_sha256=model['sha256'],
        macs=macs,
    )
