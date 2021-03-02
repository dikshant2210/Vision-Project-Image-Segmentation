# Input-Output dimensions 

## Convoutional layer 
- Input: (Ci, W1, H1)
- Number of filters: Co 
- Kernel size: F 
- Stride: S
- Zero Padding: P 
- Output: (Co, W2, H2) where 
    - W2 = (W1 - F + 2P) / S + 1
    - H2 = (H2 - F + 2P) / S + 1

## MaxPooling layer 
- Input: (Ci, W1, H1)
- Kernel size: F 
- Stride: S
- Zero Padding: P 
- Output: (Ci, W2, H2) where 
    - W2 = (W1 - F + 2P) / S + 1
    - H2 = (H2 - F + 2P) / S + 1

# Encoders

1. Layers 

- Convolutional layer: nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)
- Batch Normalization: nn.BatchNorm2d(in_channels) 
- ReLu: nn.ReLU()
- Max pooling: nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

A encoder layer is formed by:
```python
encoder_layer = nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(out_channels),
    nn.ReLU()
)
 ```

A encoder block is formed by: 
```python 
nn.Sequential( 
    encoder_layer(in_channels, out_channels),
    encoder_layer(out_channels, out_channels),
    {encoder_layer(out_channels, out_channels)},
)
nn.functional.max_pool2d(input, indices, kernel_size=2, stride=2)
```

2. Dimensions 
Using the structure [in_channels_layer01, in_channels_layer02, ...], [out_channels_layer02, ...] to define the dimensions of each block:

- encoder_block01: [3, 64], [64, 64]
- encoder_block02: [64, 128], [128, 128]
- encoder_block03: [128, 256, 256], [256, 256, 256]
- encoder_block04: [256, 512, 512], [512, 512, 512]
- encoder_block05: [512, 512, 512], [512, 512, 512]


# Decorders 
1. Layers 
- Upsampling: nn.functional.max_unpool2d
- Deconvolutional layer: nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1)
- Batch Normalization: nn.BatchNorm2d(in_channels) 
- ReLu: nn.ReLU()

A decoder layer is formed by:
```python
decoder_layer = nn.Sequential(
    nn.ConvTranspose2d(in_channels, out_channels,kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(out_channels),
    nn.ReLU()
)
 ```

A decoder block is formed by: 
```python 
nn.Sequential( 
    decoder_layer(in_channels, out_channels),
    decoder_layer(out_channels, out_channels),
    {decoder_layer(out_channels, out_channels)}
)
nn.functional.max_unpool2d(input, indices, kernel_size=2, stride=2)
```

2. Dimensions 
Using the structure [in_channels_layer01, in_channels_layer02, ...], [out_channels_layer02, ...]: 

- decoder_block05 [512, 512, 512], [512, 512, 512]
- decoder_block04 [512, 512, 512], [512, 512, 256]
- decoder_block03 [256, 256, 256], [256, 256, 128]
- decoder_block02 [128, 128], [128, 64]
- decoder_block01 [64, 64], [64, 3]
