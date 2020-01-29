import torchvision.models
import torch
import function
import model.blocks as blocks
import torch.nn.functional as F
import torch.nn as nn

class UNetAutoencoder(nn.Module):
    """ U-Net Autoencoder architecture for style transfer, that uses a bunch of AdaIn transformations
    as battleneck. """

    def __init__(self, channels, style_dim, residual_downsampling=True, residual_adain=True, residual_upsampling=True, 
        down_normalization='in', up_normalization='adain', num_adain_convolutions=5, 
        num_downup_convolutions=5, num_downup_without_connections=0, output_activation='sigmoid'):
        """ Initializes the generic decoder.
        
        Parameters:
        -----------
        channels : int
            The number of channels the input image has.
        style_dim : int or None
            The style embedding dimensionality.
        residual_downsampling : bool
            If the downsampling convolutions are implemented by residual blocks.
        residual_adain : bool
            If the adain convolutions are implemented by residual blocks.
        residual_upsampling : bool
            If the upsampling convolutions are implemented by residual blocks.
        down_normalization : bool or list of bools
            Which kind of normalization to use for downsampling. Can be a single value (applied to all layers) or a list with length num_downup_convolutions.
        up_normalization : 'in', 'adain' or None
            Which kind of normalization to use. Can be a single value (applied to all layers) or a list with length num_downup_convolutions.
        num_adain_convolutions: int
            The number of AdaIn Convolutional blocks to be applied before upsampling.
        num_downup_convolutions : int
            The number of downsampling / upsampling convolutions.
        num_downup_without_connections : int
            The number of conv blocks without unet connections.
        output_activation : 'sigmoid' or None
            Element-wise activation for the output of the decoder.
        """
        super().__init__()
        self.style_dim = style_dim
        self.output_activation = output_activation
        self.num_downup_convolutions = num_downup_convolutions
        self.num_adain_convolutions = num_adain_convolutions
        self.num_downup_without_connections = num_downup_without_connections
        self.remaining_upconvs = self.num_downup_convolutions - self.num_downup_without_connections

        dims = [min(512, 64*2**i) for i in range(self.num_downup_convolutions)]

        # Downsampling
        if not isinstance(down_normalization, list):
            down_normalization = self.num_downup_convolutions * [down_normalization]
        self.use_down_normalization = down_normalization
        self.down_convs = nn.ModuleList(
            blocks.DownsamplingConvolution(c_in, c_out, residual=residual_downsampling) 
            for c_in, c_out, use_norm in zip([channels] + dims[:-1], dims, self.use_down_normalization)
        )

        # AdaIn Convs
        self.adain_convs = nn.ModuleList(
            blocks.AdaInConvolution(dims[-1], self.style_dim, residual=residual_adain) for _ in range(num_adain_convolutions)
        )

        # Upsampling
        dims = list(reversed(dims))
        if not isinstance(up_normalization, list):
            up_normalization = self.num_downup_convolutions * [up_normalization]
        self.use_up_normalizations = up_normalization

        if self.num_downup_without_connections > 0:
            self.up_convs = nn.ModuleList(
                blocks.UpsamplingConvolution(2 * c_in, c_out, style_dim=style_dim, instance_normalization=norm, residual=residual_upsampling)
                for c_in, c_out, norm in zip(dims[:self.remaining_upconvs], dims[1:(1 + self.remaining_upconvs)], up_normalization)
            )

            self.up_convs_no_connections = nn.ModuleList(
                blocks.UpsamplingConvolution(c_in, c_out, style_dim=style_dim, instance_normalization=norm, residual=residual_upsampling)
                for c_in, c_out, norm in zip(dims[self.remaining_upconvs:], dims[(1 + self.remaining_upconvs):] + [channels], up_normalization)
            )
        else:
            self.up_convs = nn.ModuleList(
                blocks.UpsamplingConvolution(2 * c_in, c_out, style_dim=style_dim, instance_normalization=norm, residual=residual_upsampling)
                for c_in, c_out, norm in zip(dims[:], dims[1:] + [channels], up_normalization)
            )

    def forward(self, x, s):
        """ Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor, shape [batch_size, channels, H, W]
            The content images to stylize.
        s : torch.Tensor, shape [batch_size, style_dim]
            The style encodings.
        
        Returns:
        --------
        y : torch.Tensor, shape [batch_size, channels, H, W]
            The stylized images.
        """
        # Downsampling, store all intermediate values
        xs = []
        for idx in range(self.num_downup_without_connections):
            x = self.down_convs[idx](x)
        
        for idx in range(self.num_downup_without_connections, self.num_downup_convolutions):
            x = self.down_convs[idx](x)
            xs.append(x) # Connected to decoder layers using U-Net
            #print(f'Appended to xs {x.size()}')
        
        # AdaIn convs
        for idx in range(self.num_adain_convolutions):
            x = self.adain_convs[idx](x, style_encoding=s)

        xs = list(reversed(xs))

        # Upsampling, use U-Net connections
        for idx in range(self.remaining_upconvs):
            # U-Net connection
            #print(f'Cat x of size {x.size()} to xs {xs[idx].size()}')
            x = torch.cat([x, xs[idx]], dim=1) # Concatenate along the channels dimension
            x = self.up_convs[idx](x, style_encoding=s)
        
        # Upsampling layers without U-Net connections
        for idx in range(self.num_downup_without_connections):
            x = self.up_convs_no_connections[idx](x, style_encoding=s)

        if self.output_activation is 'sigmoid':
            x = torch.sigmoid(x)
        elif self.output_activation is not None:
            raise RuntimeError(f'Unknown output activation {self.output_activation}')
        return x

