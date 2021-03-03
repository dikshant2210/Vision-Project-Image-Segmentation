import torch.nn as nn
import torch.nn.functional as F

def segnet_encoder_layer(in_channels, out_channels, kernel_size=3, 
                         stride=1, padding=1): 
  """Create a SegNet encoder layer: Convolution + Batch Normalization + ReLU.

    Args:
      in_channels (int): number of input channels
      out_channels (int): number of output channels
      kernel_size (int): kernel size of the  convolution
      stride (int): stride of the convolution
      padding (int): padding of the convolution

    Returns: 
      (torch.nn.Sequential): SegNet encoder layer
  """
  layer = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                stride=stride, padding=padding),
      nn.BatchNorm2d(out_channels),
      nn.ReLU()
  )
  return layer 

def segnet_encoder_block(in_channels, out_channels):
  """Generate a SegNet encoder block ommitting the final Max Pooling.
  """
  assert len(in_channels) == len(out_channels)
  layers = [] 
  for i in range(len(in_channels)): 
    layer = segnet_encoder_layer(in_channels[i], out_channels[i])
    layers.append(layer)
  block = nn.Sequential(*layers)
  return block

def segnet_decoder_layer(in_channels, out_channels, kernel_size=3, 
                         stride=1, padding=1):
  """Create a SegNet decoder layer: Deconvolution + Batch Normalization + ReLU.

    Args:
      in_channels (int): number of input channels
      out_channels (int): number of output channels
      kernel_size (int): kernel size of the transpose convolution
      stride (int): stride of the transpose convolution
      padding (int): padding of the transpose convolution

    Returns:
      (torch.nn.Sequential): SegNet decoder layer
  """
  layer = nn.Sequential(
      nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, 
                stride=stride, padding=padding),
      nn.BatchNorm2d(out_channels),
      nn.ReLU()
  )
  return layer 

def segnet_decoder_block(in_channels, out_channels):
  """Generate a SegNet decoder block ommitting the initial Upsampling.
  """
  assert len(in_channels) == len(out_channels)
  layers = [] 
  for i in range(len(in_channels)): 
    layer = segnet_decoder_layer(in_channels[i], out_channels[i])
    layers.append(layer)
  block = nn.Sequential(*layers)
  return block

class Segnet(nn.Module):
  """SegNet Class"""

  def __init__(self, input_channels, output_channels, vgg16_bn):
    """Initialize an instance of SegNet

      Args:
        input_channels (int): number of input channels 
        output_channels (int): number of output channels
        vgg16_bn (torch.model): pretrained VGG-16 (with Batch Normalization) model
    """
    super(Segnet, self).__init__()
    self.input_channels = input_channels 
    self.output_channels = output_channels 
    self.debug = False   

    # Encoder (VGG16 without Classifier)
    self.enc_block00 = segnet_encoder_block(self.encoder_dims('block00','in'), self.encoder_dims('block00','out'))
    self.enc_block01 = segnet_encoder_block(self.encoder_dims('block01','in'), self.encoder_dims('block01','out'))
    self.enc_block02 = segnet_encoder_block(self.encoder_dims('block02','in'), self.encoder_dims('block02','out'))
    self.enc_block03 = segnet_encoder_block(self.encoder_dims('block03','in'), self.encoder_dims('block03','out'))
    self.enc_block04 = segnet_encoder_block(self.encoder_dims('block04','in'), self.encoder_dims('block04','out'))
    
    set pretrained weights 
    self._load_encoder_weights(vgg16_bn)

    # Decoder 
    self.dec_block04 = segnet_decoder_block(self.decoder_dims('block04','in'), self.decoder_dims('block04','out'))
    self.dec_block03 = segnet_decoder_block(self.decoder_dims('block03','in'), self.decoder_dims('block03','out'))
    self.dec_block02 = segnet_decoder_block(self.decoder_dims('block02','in'), self.decoder_dims('block02','out'))
    self.dec_block01 = segnet_decoder_block(self.decoder_dims('block01','in'), self.decoder_dims('block01','out'))
    self.dec_block00 = segnet_decoder_block(self.decoder_dims('block00','in'), self.decoder_dims('block00','out'))

    softmax
    self.softmax = nn.Softmax(dim=1)

  def debug(self):
    """Activate debug mode"""
    self.debug = True 

  def no_debug(self):
    """Deactivate debug mode"""
    self.debug = False

  def debug(self, debug):
    """Set debug mode """
    self.debug = debug
     
  def forward(self, x):
    """Compute a forward pass 

      Args: 
        x (torch.Tensor): input image

      Returns: 
        (torch.Tensor, torch.Tensor): output of the network without and applying softmax
    """ 
    # -- Encoder --
    # Encoder Block 00
    dim00 = x.size()
    enc00 = self.enc_block00(x)
    mp00, indices00 = F.max_pool2d(enc00, kernel_size=2, stride=2, return_indices=True)
    # Encoder Block 01
    dim01 = mp00.size()
    enc01 = self.enc_block01(mp00)
    mp01, indices01 = F.max_pool2d(enc01, kernel_size=2, stride=2, return_indices=True)
    # Encoder Block 02
    dim02 = mp01.size()
    enc02 = self.enc_block02(mp01)
    mp02, indices02 = F.max_pool2d(enc02, kernel_size=2, stride=2, return_indices=True)
    # Encoder Block 03
    dim03 = mp02.size()
    enc03 = self.enc_block03(mp02)
    mp03, indices03 = F.max_pool2d(enc03, kernel_size=2, stride=2, return_indices=True)
    # Encoder Block 04
    dim04 = mp03.size()
    enc04 = self.enc_block04(mp03)
    mp04, indices04 = F.max_pool2d(enc04, kernel_size=2, stride=2, return_indices=True)

    print("Endoded")
    # -- Decoder --
    # Decoder Block 04 
    up04 = F.max_unpool2d(mp04, indices04, kernel_size=2, stride=2, output_size=dim04)
    dec04 = self.dec_block04(up04)
    # Decoder Block 03
    up03 = F.max_unpool2d(dec04, indices03, kernel_size=2, stride=2, output_size=dim03)
    dec03 = self.dec_block03(up03)
    # Decoder Block 02
    up02 = F.max_unpool2d(dec03, indices02, kernel_size=2, stride=2, output_size=dim02)
    dec02 = self.dec_block02(up02)
    # Decoder Block 01
    up01 = F.max_unpool2d(dec02, indices01, kernel_size=2, stride=2, output_size=dim01)
    dec01 = self.dec_block01(up01)
    # Decoder Block 00
    up00 = F.max_unpool2d(dec01, indices00, kernel_size=2, stride=2, output_size=dim00)
    dec00 = self.dec_block00(up00)
    softmax
    output = self.softmax(dec00)

    return dec00, output

  def encoder_dims(self, block, io):
    """Obtain the encoder dimensions based on the input dimensions

      Args: 
        block (string): encoder block 
        io (string): 'in' or 'out'

      Returns: 
        (array): dimensions of the corresponding encoder block
    """
    encoder_dimensions = {
      'block00': { 'in': [self.input_channels, 64],'out': [64, 64] },
      'block01': { 'in': [64, 128],       'out': [128, 128] },
      'block02': { 'in': [128, 256, 256], 'out': [256, 256, 256] },
      'block03': { 'in': [256, 512, 512], 'out': [512, 512, 512] },
      'block04': { 'in': [512, 512, 512], 'out': [512, 512, 512] }
    }
    return encoder_dimensions[block][io]

  def decoder_dims(self, block, io): 
    """Obtain the decoder's dimensions based on the output dimensions 
    """
    decoder_dimensions = {
      'block04': { 'in': [512, 512, 512], 'out': [512, 512, 512] },
      'block03': { 'in': [512, 512, 512], 'out': [512, 512, 256] },
      'block02': { 'in': [256, 256, 256], 'out': [256, 256, 128] },
      'block01': { 'in': [128, 128],       'out': [128, 64] },
      'block00': { 'in': [64, 64],         'out': [64, self.output_channels] }
    }
    return decoder_dimensions[block][io]
    
  def _load_encoder_weights(self, vgg16_bn):
    """
      Load the corresponding weights of the train VGG16 model into the encoder.
    """ 
    # Encoder block00
    assert self.enc_block00[0][0].weight.size() == vgg16_bn.features[0].weight.size() 
    assert self.enc_block00[0][0].bias.size() == vgg16_bn.features[0].bias.size() 
    assert self.enc_block00[0][1].weight.size() == vgg16_bn.features[1].weight.size() 
    assert self.enc_block00[0][1].bias.size() == vgg16_bn.features[1].bias.size() 
    assert self.enc_block00[1][0].weight.size() == vgg16_bn.features[3].weight.size() 
    assert self.enc_block00[1][0].bias.size() == vgg16_bn.features[3].bias.size() 
    assert self.enc_block00[1][1].weight.size() == vgg16_bn.features[4].weight.size() 
    assert self.enc_block00[1][1].bias.size() == vgg16_bn.features[4].bias.size() 

    self.enc_block00[0][0].weight.data = vgg16_bn.features[0].weight.data 
    self.enc_block00[0][0].bias.data = vgg16_bn.features[0].bias.data 
    self.enc_block00[0][1].weight.data = vgg16_bn.features[1].weight.data 
    self.enc_block00[0][1].bias.data = vgg16_bn.features[1].bias.data 
    self.enc_block00[1][0].weight.data = vgg16_bn.features[3].weight.data 
    self.enc_block00[1][0].bias.data = vgg16_bn.features[3].bias.data 
    self.enc_block00[1][1].weight.data = vgg16_bn.features[4].weight.data 
    self.enc_block00[1][1].bias.data = vgg16_bn.features[4].bias.data

    # Encoder block01
    assert self.enc_block01[0][0].weight.size() == vgg16_bn.features[7].weight.size() 
    assert self.enc_block01[0][0].bias.size() == vgg16_bn.features[7].bias.size() 
    assert self.enc_block01[0][1].weight.size() == vgg16_bn.features[8].weight.size() 
    assert self.enc_block01[0][1].bias.size() == vgg16_bn.features[8].bias.size() 
    assert self.enc_block01[1][0].weight.size() == vgg16_bn.features[10].weight.size() 
    assert self.enc_block01[1][0].bias.size() == vgg16_bn.features[10].bias.size() 
    assert self.enc_block01[1][1].weight.size() == vgg16_bn.features[11].weight.size() 
    assert self.enc_block01[1][1].bias.size() == vgg16_bn.features[11].bias.size() 
    
    self.enc_block01[0][0].weight.data = vgg16_bn.features[7].weight.data 
    self.enc_block01[0][0].bias.data = vgg16_bn.features[7].bias.data 
    self.enc_block01[0][1].weight.data = vgg16_bn.features[8].weight.data 
    self.enc_block01[0][1].bias.data = vgg16_bn.features[8].bias.data 
    self.enc_block01[1][0].weight.data = vgg16_bn.features[10].weight.data 
    self.enc_block01[1][0].bias.data = vgg16_bn.features[10].bias.data 
    self.enc_block01[1][1].weight.data = vgg16_bn.features[11].weight.data 
    self.enc_block01[1][1].bias.data = vgg16_bn.features[11].bias.data 

    # Encoder block02 
    assert self.enc_block02[0][0].weight.size() == vgg16_bn.features[14].weight.size() 
    assert self.enc_block02[0][0].bias.size() == vgg16_bn.features[14].bias.size() 
    assert self.enc_block02[0][1].weight.size() == vgg16_bn.features[15].weight.size() 
    assert self.enc_block02[0][1].bias.size() == vgg16_bn.features[15].bias.size() 
    assert self.enc_block02[1][0].weight.size() == vgg16_bn.features[17].weight.size() 
    assert self.enc_block02[1][0].bias.size() == vgg16_bn.features[17].bias.size() 
    assert self.enc_block02[1][1].weight.size() == vgg16_bn.features[18].weight.size() 
    assert self.enc_block02[1][1].bias.size() == vgg16_bn.features[18].bias.size() 
    assert self.enc_block02[2][0].weight.size() == vgg16_bn.features[20].weight.size() 
    assert self.enc_block02[2][0].bias.size() == vgg16_bn.features[20].bias.size() 
    assert self.enc_block02[2][1].weight.size() == vgg16_bn.features[21].weight.size() 
    assert self.enc_block02[2][1].bias.size() == vgg16_bn.features[21].bias.size() 

    self.enc_block02[0][0].weight.data = vgg16_bn.features[14].weight.data 
    self.enc_block02[0][0].bias.data = vgg16_bn.features[14].bias.data 
    self.enc_block02[0][1].weight.data = vgg16_bn.features[15].weight.data 
    self.enc_block02[0][1].bias.data = vgg16_bn.features[15].bias.data 
    self.enc_block02[1][0].weight.data = vgg16_bn.features[17].weight.data 
    self.enc_block02[1][0].bias.data = vgg16_bn.features[17].bias.data 
    self.enc_block02[1][1].weight.data = vgg16_bn.features[18].weight.data 
    self.enc_block02[1][1].bias.data = vgg16_bn.features[18].bias.data 
    self.enc_block02[2][0].weight.data = vgg16_bn.features[20].weight.data 
    self.enc_block02[2][0].bias.data = vgg16_bn.features[20].bias.data 
    self.enc_block02[2][1].weight.data = vgg16_bn.features[21].weight.data 
    self.enc_block02[2][1].bias.data = vgg16_bn.features[21].bias.data 

    # Encoder block03
    assert self.enc_block03[0][0].weight.size() == vgg16_bn.features[24].weight.size() 
    assert self.enc_block03[0][0].bias.size() == vgg16_bn.features[24].bias.size() 
    assert self.enc_block03[0][1].weight.size() == vgg16_bn.features[25].weight.size() 
    assert self.enc_block03[0][1].bias.size() == vgg16_bn.features[25].bias.size()
    assert self.enc_block03[1][0].weight.size() == vgg16_bn.features[27].weight.size() 
    assert self.enc_block03[1][0].bias.size() == vgg16_bn.features[27].bias.size() 
    assert self.enc_block03[1][1].weight.size() == vgg16_bn.features[28].weight.size() 
    assert self.enc_block03[1][1].bias.size() == vgg16_bn.features[28].bias.size() 
    assert self.enc_block03[2][0].weight.size() == vgg16_bn.features[30].weight.size() 
    assert self.enc_block03[2][0].bias.size() == vgg16_bn.features[30].bias.size() 
    assert self.enc_block03[2][1].weight.size() == vgg16_bn.features[31].weight.size() 
    assert self.enc_block03[2][1].bias.size() == vgg16_bn.features[31].bias.size() 

    self.enc_block03[0][0].weight.data = vgg16_bn.features[24].weight.data 
    self.enc_block03[0][0].bias.data = vgg16_bn.features[24].bias.data 
    self.enc_block03[0][1].weight.data = vgg16_bn.features[25].weight.data 
    self.enc_block03[0][1].bias.data = vgg16_bn.features[25].bias.data
    self.enc_block03[1][0].weight.data = vgg16_bn.features[27].weight.data 
    self.enc_block03[1][0].bias.data = vgg16_bn.features[27].bias.data 
    self.enc_block03[1][1].weight.data = vgg16_bn.features[28].weight.data 
    self.enc_block03[1][1].bias.data = vgg16_bn.features[28].bias.data 
    self.enc_block03[2][0].weight.data = vgg16_bn.features[30].weight.data 
    self.enc_block03[2][0].bias.data = vgg16_bn.features[30].bias.data 
    self.enc_block03[2][1].weight.data = vgg16_bn.features[31].weight.data 
    self.enc_block03[2][1].bias.data = vgg16_bn.features[31].bias.data 

    # Encoder block04
    assert self.enc_block04[0][0].weight.size() == vgg16_bn.features[34].weight.size() 
    assert self.enc_block04[0][0].bias.size() == vgg16_bn.features[34].bias.size() 
    assert self.enc_block04[0][1].weight.size() == vgg16_bn.features[35].weight.size() 
    assert self.enc_block04[0][1].bias.size() == vgg16_bn.features[35].bias.size() 
    assert self.enc_block04[1][0].weight.size() == vgg16_bn.features[37].weight.size() 
    assert self.enc_block04[1][0].bias.size() == vgg16_bn.features[37].bias.size() 
    assert self.enc_block04[1][1].weight.size() == vgg16_bn.features[38].weight.size() 
    assert self.enc_block04[1][1].bias.size() == vgg16_bn.features[38].bias.size() 
    assert self.enc_block04[2][0].weight.size() == vgg16_bn.features[40].weight.size() 
    assert self.enc_block04[2][0].bias.size() == vgg16_bn.features[40].bias.size() 
    assert self.enc_block04[2][1].weight.size() == vgg16_bn.features[41].weight.size() 
    assert self.enc_block04[2][1].bias.size() == vgg16_bn.features[41].bias.size()

    self.enc_block04[0][0].weight.data = vgg16_bn.features[34].weight.data 
    self.enc_block04[0][0].bias.data = vgg16_bn.features[34].bias.data 
    self.enc_block04[0][1].weight.data = vgg16_bn.features[35].weight.data 
    self.enc_block04[0][1].bias.data = vgg16_bn.features[35].bias.data 
    self.enc_block04[1][0].weight.data = vgg16_bn.features[37].weight.data 
    self.enc_block04[1][0].bias.data = vgg16_bn.features[37].bias.data 
    self.enc_block04[1][1].weight.data = vgg16_bn.features[38].weight.data 
    self.enc_block04[1][1].bias.data = vgg16_bn.features[38].bias.data 
    self.enc_block04[2][0].weight.data = vgg16_bn.features[40].weight.data 
    self.enc_block04[2][0].bias.data = vgg16_bn.features[40].bias.data 
    self.enc_block04[2][1].weight.data = vgg16_bn.features[41].weight.data 
    self.enc_block04[2][1].bias.data = vgg16_bn.features[41].bias.data
