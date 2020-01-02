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

class AdaInConvolution(torch.nn.Module):
    """ Block that does not change image size but applies convolutions and AdaIn. """

    def __init__(self, channels, style_dim, kernel_size=(3, 3), residual=True):
        """ Initialization.
        
        Parameters:
        -----------
        channels : int
            The number of channels, same for input and output.
        style_dim : int
            Dimensionality of the style encoding.
        kernel_size : int or tuple of ints
            The kernel size of the convolution operations.
        residual : bool
            If True, the block is implemented as a residual block.
        """
        super().__init__()
        self.residual = residual

        self.pad = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=0)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=0)

        self.norm1 = AdaInBlock(style_dim, channels)
        self.norm2 = AdaInBlock(style_dim, channels)

    def forward(self, x, style_encoding):
        """ Forward pass. 
        
        Parameters:
        -----------
        x : torch.Tensor, shape [B, C, H, W]
            The images to be processed.
        style_encoding : torch.Tensor, shape [B, D] or None
            The style encoding that is passed to AdaIn layers.
        
        Returns:
        --------
        x' : torch.Tensor, shape [B, C, H, W]
            The output image, with applied normalizations.
        """
        out = x
        out = F.relu(out, inplace=True)
        out = self.conv1(self.pad(x))
        out = self.norm1(out, style_encoding)
        out = F.relu(out, inplace=True)
        out = self.conv2(self.pad(x))
        out = self.norm2(out, style_encoding)
        
        if self.residual:
            out = out + x

        return out

class UpsamplingConvolution(torch.nn.Module):
    """ Block that upsamples an image, applies transposed convolutions and then normalizes the output. """

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, 
         instance_normalization=None, style_dim=None, residual=True):
        """ Initialization.
        
        Parameters:
        -----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : int or tuple of ints
            The kernel size of the convolution operations.
        stride : int
            The stride of the convolution operations.
        instance_normalization : 'adain', 'in' or None
            Which kind of instance normalization should be applied.
        style_dim : int
            Dimensionality of the style encoding, if Adaptive Instance Normalization is to be applied.
        residual : bool
            If True, the block is implemented as a residual block.
        """
        super().__init__()
        self.instance_normalization = instance_normalization
        self.residual = residual

        self.pad = nn.ReflectionPad2d(1)
        
        # Two convolutions, no padding since we use reflection padding to avoid artifacts at border
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)

        if self.residual:
            self.conv_residual = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)

        # Instance Normalization
        if self.instance_normalization == 'in':
            self.norm1 = nn.InstanceNorm2d(in_channels)
            self.norm2 = nn.InstanceNorm2d(out_channels)
        elif self.instance_normalization == 'adain':
            self.norm1 = AdaInBlock(style_dim, in_channels)
            self.norm2 = AdaInBlock(style_dim, out_channels)
        elif self.instance_normalization is not None:
            raise NotImplemented

    def forward(self, x, style_encoding=None):
        """ Forward pass. 
        
        Parameters:
        -----------
        x : torch.Tensor, shape [B, C, H, W]
            The images to be upsampled.
        style_encoding : torch.Tensor, shape [B, D] or None
            The style encoding that is passed to AdaIn layers.
        
        Returns:
        --------
        x' : torch.Tensor, shape [B, C', H * 2, W * 2]
            The upsampled output image, with potential applied normalizations.
        """
        out = x

        if self.instance_normalization == 'in':
            out = F.relu(self.norm1(out), inplace=False)
        elif self.instance_normalization == 'adain':
            out = F.relu(self.norm1(out, style_encoding), inplace=False)
        out = F.interpolate(out, mode='nearest', scale_factor=2)
        out = self.conv1(self.pad(out))
        if self.instance_normalization == 'in':
            out = F.relu(self.norm2(out), inplace=True)
        elif self.instance_normalization == 'adain':
            out = F.relu(self.norm2(out, style_encoding), inplace=True)
        out = self.conv2(self.pad(out))

        if self.residual: # Residual
            residual = F.interpolate(x, mode='nearest', scale_factor=2)
            residual = self.conv_residual(self.pad(residual))
            out = out + residual
        
        return out

class DownsamplingConvolution(nn.Module):
    """ Residual block that applies downsampling and convolutions. """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, residual=True, instance_normalization=False):
        """ Initializes the downsampling convolution.
        
        Parameters:
        -----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : int
            The kernel size.
        stride : int
            The stride of convolution operation.
        residual : bool
            If True, the block is implemented as a residual block.
        instance_normalization : bool
            If True, the instance normalization is applied downsampling.
        """
        super().__init__()
        self.residual = residual
        self.instance_normalization = instance_normalization

        self.pool = nn.AvgPool2d(2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        if self.residual:
            self.conv_residual = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        if self.instance_normalization:
            self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
            self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        """ Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor, shape [B, C, H, W]
            Input images of size (W, H) with C input channels.
        
        Returns:
        --------
        x' : torch.Tensor, shape [B, C', H / 2, W / 2]
            Output image of size (W / 2, H / 2) with C' output channels.
        """
        out = x
        out = self.conv1(out)
        if self.instance_normalization:
            out = self.norm1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        if self.instance_normalization:
            out = self.norm2(out)
        out = F.relu(out, inplace=True)
        out = self.pool(out)

        if self.residual: # Residual
            residual = self.conv_residual(F.relu(x, inplace=False))
            residual = self.pool(residual)
            out = out + residual

        return out