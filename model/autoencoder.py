import torchvision.models
import torch
import function
import model.blocks as blocks
import torch.nn.functional as F
import torch.nn as nn

class Encoder(torch.nn.Module):
    """ Generic encoder that encodes an image into a fixed size vector using down convolutions. """

    def __init__(self, output_dim, in_channels=3, normalization=True, residual=True, num_down_convolutions=6):
        """ Initializes the encoder.
        
        Parameters:
        -----------
        output_dim : int or None
            The size of the embedding. If None, the output is a downsampled batch of images with spatial extent.
        in_channels : int
            The number of input channels.
        normalization : bool
            If True, instance normalization is applied after a down convolution.
        residual : bool
            If True, down convolutions use residual blocks.
        """
        super().__init__()
        self.normalization = normalization
        self.num_down_convolutions = num_down_convolutions
        self.output_dim = output_dim
        dims = [min(512, 64*2**i) for i in range(num_down_convolutions)]

        self.convs = nn.ModuleList(
            blocks.DownsamplingConvolution(c_in, c_out, residual=residual) for c_in, c_out in zip([in_channels] + dims[:-1], dims)
        )

        if self.normalization:
            self.norms = nn.ModuleList(
                nn.InstanceNorm2d(c, affine=True) for c in dims
            )

        if self.output_dim is not None:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = nn.Linear(dims[-1], output_dim, bias=True)

    def forward(self, x):
        """ Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor, shape [B, channels_in, H, W]
            Inputs to the encoder.
        
        Returns:
        --------
        z : torch.Tensor, shape [B, output_dim] or shape [B, channels_out, H'', W'']
            Output image embedding.
        """
        out = x
        for idx in range(self.num_down_convolutions):
            out = self.convs[idx](out)
            if self.normalization:
                out = self.norms[idx](out)

        if self.output_dim is not None:
            out = F.relu(self.pool(out), inplace=True)
            out = self.fc(out.view(out.size(0), -1))
        
        return out

class Decoder(torch.nn.Module):
    """ General purpose decoder that uses AdaIn layers to modify the embedding multiple times and then uses
    TransposeConvs to create the output image. """

    def __init__(self, content_dim, style_dim, resolution, out_channels=3, residual_adain=True, residual_upsampling=True, 
        normalization='adain', num_adain_convolutions=5, 
        num_up_convolutions=6, output_activation='sigmoid'):
        """ Initializes the generic decoder.
        
        Parameters:
        -----------
        content_dim : int
            The content embedding dimensionality.
        style_dim : int or None
            The style embedding dimensionality.
        resolution : int or tuple of ints (H, W)
            The output resultion, a power of 2.
        out_channels : int
            The number of output channels.
        residual_adain : bool
            If the upsampling convolutions are implemented by residual blocks in adain convolutions
        residual_upsampling : bool
            If the upsampling convolutions are implemented by residual blocks in upsampling convolutions.
        normalization : 'in', 'adain' or None
            Which kind of normalization to use. Can be a single value (applied to all layers) or a list with length num_up_convolutions.
        num_adain_convolutions: int
            The number of AdaIn Convolutional blocks to be applied before upsampling.
        num_up_convolutions : int
            The number of upsampling convolutions.
        output_activation : 'sigmoid' or None
            Element-wise activation for the output of the decoder.
        """
        super().__init__()
        self.content_dim = content_dim
        self.style_dim = style_dim
        self.residual_adain = residual_adain
        self.residual_upsampling = residual_upsampling
        self.normalization = normalization
        self.num_adain_convolutions = num_adain_convolutions
        self.num_up_convolutions = num_up_convolutions
        self.output_activation = output_activation

        dims = list(reversed([min(512, 64*2**i) for i in range(num_up_convolutions)]))

        # The content input has to be reshaped to a spatial dimension, calculate the spatial dimensions
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        self.in_height = resolution[0] // 2**self.num_up_convolutions
        self.in_width = resolution[1] // 2**self.num_up_convolutions
        self.in_channels = dims[0]

        if self.content_dim is not None:
            self.fc = nn.Linear(self.content_dim, self.in_channels * self.in_width * self.in_height) 

        self.adain_convs = nn.ModuleList(blocks.AdaInConvolution(self.in_channels, style_dim, residual=self.residual_adain) for _ in range(num_adain_convolutions))

        if isinstance(self.normalization, str):
            self.normalization = [self.normalization for _ in range(self.num_up_convolutions)]

        self.up_convs = nn.ModuleList(
            blocks.UpsamplingConvolution(c_in, c_out, style_dim=style_dim, instance_normalization=self.normalization[idx], residual=self.residual_upsampling)
            for idx, (c_in, c_out) in enumerate(zip(dims, dims[1:] + [out_channels]))
        )


    def forward(self, c, s=None):
        """ Forward pass.
        
        Parameters:
        -----------
        c : torch.Tensor, shape [batch_size, ...]
            Encoding of the content image.
        s : torch.Tensor, shape [batch_size, style_dim]
            Style encoding of the style image to apply to the content image.
        
        Returns:
        --------
        stylized : torch.Tensor, shape [batch_size, , H, W]
            The stylized output image, where H and W are specified by the resolution given to the decoder initialization.
        """
        if self.content_dim is not None:
            c = F.relu(self.fc(c), inplace=True)
            c = c.view(c.size(0), self.in_channels, self.in_height, self.in_width)
        
        for idx in range(self.num_adain_convolutions):
            c = self.adain_convs[idx](c, style_encoding=s)
        
        for idx in range(self.num_up_convolutions):
            c = self.up_convs[idx](c, style_encoding=s)

        if self.output_activation is 'sigmoid':
            c = torch.sigmoid(c)
        elif self.output_activation is not None:
            raise RuntimeError(f'Unknown output activation {self.output_activation}')
        return c

