__all__ = [
    "resblock_m",
    "resblock_plain",
    "resblock_bottleneck",
    "resblock_preact",
    "resblock_postact",
]

resblock_m = {
    "src": {
        "input": ["input"],
        "output": ["input"],
        "opType": "None",
    },
    "conv_1": {
        "input": ["input"],
        "output": ["bn_1"],
        "opType": "Conv",
    },
    "bn_1": {
        "input": ["bn_1"],
        "output": ["relu_1"],
        "opType": "BatchNormalization",
    },
    "relu_1": {
        "input": ["relu_1"],
        "output": ["conv_2"],
        "opType": "Relu",
    },
    "conv_2": {
        "input": ["conv_2"],
        "output": ["add_in_1"],
        "opType": "Conv",
    },
    "conv_3": {
        "input": ["input"],
        "output": ["add_in_2"],
        "opType": "Conv",
    },
    "add_1": {
        "input": ["add_in_1", "add_in_2"],
        "output": ["output"],
        "opType": "Add",
    },
}

resblock_plain = {
    "src": {
        "input": ["input"],
        "output": ["input"],
        "opType": "None",
    },
    "conv_1": {
        "input": ["input"],
        "output": ["relu_1"],
        "opType": "Conv",
    },
    "relu_1": {
        "input": ["relu_1"],
        "output": ["conv_2"],
        "opType": "Relu",
    },
    "conv_2": {
        "input": ["conv_2"],
        "output": ["add_in_1"],
        "opType": "Conv",
    },
    "add_1": {
        "input": ["add_in_1", "input"],
        "output": ["add_out"],
        "opType": "Add",
    },
    "relu_2": {
        "input": ["add_out"],
        "output": ["output"],
        "opType": "Relu",
    },
}

resblock_bottleneck = {
    "src": {
        "input": ["input"],
        "output": ["input"],
        "opType": "None",
    },
    "conv_1": {
        "input": ["input"],
        "output": ["relu_1"],
        "opType": "Conv",
    },
    "relu_1": {
        "input": ["relu_1"],
        "output": ["conv_2"],
        "opType": "Relu",
    },
    "conv_2": {
        "input": ["conv_2"],
        "output": ["relu_2"],
        "opType": "Conv",
    },
    "relu_2": {
        "input": ["relu_1"],
        "output": ["relu_2"],
        "opType": "Conv",
    },
    "conv_3": {
        "input": ["relu_2"],
        "output": ["add_in_1"],
        "opType": "Conv",
    },
    "add_1": {
        "input": ["add_in_1", "input"],
        "output": ["add_out"],
        "opType": "Add",
    },
    "relu_3": {
        "input": ["add_out"],
        "output": ["output"],
        "opType": "Relu",
    },
}

resblock_preact = {
    "src": {
        "input": ["input"],
        "output": ["input"],
        "opType": "None",
    },
    "conv_1": {
        "input": ["input"],
        "output": ["relu_1"],
        "opType": "Conv",
    },
    "relu_1": {
        "input": ["relu_1"],
        "output": ["bn_1"],
        "opType": "Relu",
    },
    "bn_1": {
        "input": ["bn_1"],
        "output": ["conv_2"],
        "opType": "BatchNormalization",
    },
    "conv_2": {
        "input": ["conv_2"],
        "output": ["bn_2"],
        "opType": "Conv",
    },
    "bn_2": {
        "input": ["bn_2"],
        "output": ["add_1"],
        "opType": "BatchNormalization",
    },
    "add_1": {
        "input": ["add_1", "input"],
        "output": ["add_out"],
        "opType": "Add",
    },
    "relu_2": {
        "input": ["add_out"],
        "output": ["output"],
        "opType": "Relu",
    },
}

resblock_postact = {
    "src": {
        "input": ["input"],
        "output": ["input"],
        "opType": "None",
    },
    "bn_1": {
        "input": ["input"],
        "output": ["relu_1"],
        "opType": "BatchNormalization",
    },
    "relu_1": {
        "input": ["relu_1"],
        "output": ["conv_1"],
        "opType": "Relu",
    },
    "conv_1": {
        "input": ["conv_1"],
        "output": ["bn_2"],
        "opType": "Conv",
    },
    "bn_2": {
        "input": ["bn_2"],
        "output": ["relu_2"],
        "opType": "BatchNormalization",
    },
    "relu_2": {
        "input": ["relu_2"],
        "output": ["conv_2"],
        "opType": "Relu",
    },
    "conv_2": {
        "input": ["conv_2"],
        "output": ["add_1"],
        "opType": "Conv",
    },
    "add_1": {
        "input": ["add_1", "input"],
        "output": ["output"],
        "opType": "Add",
    },
}
