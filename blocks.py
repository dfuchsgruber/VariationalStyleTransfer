import torch.nn as nn
import torch.nn.functional as F
import torch
import function

class AdaInBlock(torch.nn.Module):
    """ Layer that applies adaptive instance normalization. """

    def __init__(self, style_dim, num_channels):
        """ Initializes an AdaIn layer that takes a style encoding as input, applies an affine transformation to it and passes
        mean and standard deviation coefficients to Adaptive Instance Normalization. 
        
        Parameters:
        -----------
        style_dim : int
            Dimensionality of the style encoding.
        num_channels : int
            How many channels the input map to be transformed has.
        """
        super().__init__()
        self.style_dim = style_dim
        self.num_channels = num_channels
        self.fc = nn.Linear(self.style_dim, 2 * self.num_channels)
    
    def forward(self, x, style_encoding):
        """ Applies learned affine coefficients to x using the style encoding.
        
        Parameters:
        -----------
        x : torch.Tensor, shape [B, C, H, W]
            The feature map to be transformed.
        style_encoding : torch.Tensor, shape [B, style_encoding_dim]
            The style encoding tensor.

        Returns:
        --------
        x' : torch.Tensor, shape [B, C, H, W]
            The transformed version of x.
        """
        affine_params = self.fc(style_encoding)
        mean = affine_params[:, : self.num_channels]
        std = affine_params[:, self.num_channels : ]
        transformed = function.adain_with_coefficients(x, mean, std)
        return transformed

class UpsamplingConvolution(torch.nn.Module):
    """ Block that upsamples an image, applies transposed convolutions and then normalizes the output. """

    def __init__(self, in_channels, out_channels, instance_normalization=None, style_dim=None):
        """ Initialization.
        
        Parameters:
        -----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        instance_normalization : 'adain', 'in' or None
            Which kind of instance normalization should be applied.
        style_dim : int
            Dimensionality of the style encoding, if Adaptive Instance Normalization is to be applied.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.instance_normalization = instance_normalization
        self.style_dim = style_dim
        
        # Two convolutions
        #self.tconv1 = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=(3, 3), stride=1, padding=1, dilation=1, bias=False)
        #self.tconv2 = nn.ConvTranspose2d(self.out_channels, self.out_channels, kernel_size=(3, 3), stride=1, padding=1, dilation=1, bias=False)
        self.tconv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(3, 3), stride=1, padding=0)
        self.tconv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(3, 3), stride=1, padding=0)

        # Instance Normalization
        if self.instance_normalization == 'in':
            self.norm = nn.InstanceNorm2d(self.out_channels)
        elif self.instance_normalization == 'adain':
            self.norm = AdaInBlock(self.style_dim, self.out_channels)
        elif self.instance_normalization is not None:
            raise NotImplemented

    def forward(self, x, style_encoding=None):
        
        x = F.interpolate(x, mode='nearest', scale_factor=2)
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        x = F.relu(self.tconv1(x), inplace=True)
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        x = F.relu(self.tconv2(x), inplace=True)

        if self.instance_normalization == 'in':
            x = self.norm(x)
        elif self.instance_normalization == 'adain':
            x = self.norm(x, style_encoding)

        return x
        

