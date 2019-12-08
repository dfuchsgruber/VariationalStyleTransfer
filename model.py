import torchvision.models
import torch
import function

class Encoder(torch.nn.Module):
    """ Encoder network that contains of the first few layers of the vgg19 [1] network. 
    
    References:
    -----------
    [1] : https://arxiv.org/pdf/1409.1556.pdf
    """
    
    
    def __init__(self, input_dim, n_layers=19, architecture=torchvision.models.vgg19, pretrained=True, flattened_output_dim=None, mean_std_projection=False):
        """ Initializes an encoder model based on some (pretrained) architecture. 
        
        Parameters:
        -----------
        input_dim : int, int, int
            Number of channels, image height and image width of the inputs.
        n_layers : int
            How many layers of the architecture are used for the encoder.
        architecture : torch.nn.module
            A torchvision architecture that allows the first n layers to be extracted.
        pretrained : bool
            If True, pretrained weights of the architecture are used.
        flattened_output_dim : int or None
            If given, the output is flattened and transformed to this dimension. Used for encoding styles into a fixed-size
            latent space.
        """
        super().__init__()
        self.layers = architecture(pretrained=pretrained, progress=True).features[:n_layers]
        self.flattened_output_dim = flattened_output_dim
        self.mean_std_projection = mean_std_projection
        if self.flattened_output_dim:
            C_out, W_out, H_out = self.output_dim(input_dim)
            self.projection = torch.nn.modules.Linear(H_out * W_out * C_out, self.flattened_output_dim)
        if self.mean_std_projection:
            C_out, W_out, H_out = self.output_dim(input_dim)
            self.projection_mean = torch.nn.modules.Linear(H_out * W_out * C_out, 1)
            self.projection_std = torch.nn.modules.Linear(H_out * W_out * C_out, 1)

        


    def forward(self, input):
        """ Forward pass through the encoder network. 
        
        Parameters:
        -----------
        input : torch.Tensor, shape [batch_size, 3, width, height]
            A batch of images to encode.

        Returns:
        --------
        output : torch.Tensor, shape [batch_size, out_features, width / scale, height / scale]
            Feature maps for the images.
        """
        output = self.layers(input)
        if self.flattened_output_dim:
            B = output.size()[0]
            output = self.projection(output.view(B, -1))
        if self.mean_std_projection:
            B = output.size()[0]
            mean = self.projection_mean(output.view(B, -1))
            var = self.projection_std(output.view(B, -1))
            output = (mean, var)
        return output

    def output_dim(self, input_dim):
        """ Calculates the output dimensionality of the encoder structure. 
        
        Parameters:
        -----------
        input_dim : int, int, int
            Number of channels, input height and input width.
        
        Returns:
        --------
        output_dim : int, int, int
            Number of channels, output height and output width.
        """
        C, H, W = input_dim
        num_poolings = sum(map(lambda layer: isinstance(layer, torch.nn.modules.MaxPool2d), self.layers))
        C_out = list(filter(lambda layer: isinstance(layer, torch.nn.modules.Conv2d), self.layers))[-1].out_channels
        return C_out, H // 2**num_poolings, W // 2**num_poolings
        

class AdaInLayer(torch.nn.Module):
    """ Layer that applies adaptive instance normalization. """

    def __init__(self, input_dim, num_channels, idx):
        """ Initializes an AdaIn layer that takes a style encoding as input, applies an affine transformation to it and passes
        mean and standard deviation coefficients to Adaptive Instance Normalization. 
        
        Parameters:
        -----------
        input_dim : int
            Dimensionality of the style encoding.
        num_channels : int
            How many channels the input map to be transformed has.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_channels = num_channels
        self.idx = idx
        #self.transformation_mean = torch.nn.modules.Linear(input_dim, num_channels)
        #self.transformation_std = torch.nn.modules.Linear(input_dim, num_channels)
    
    def forward(self, x, style_encoding):
        """ Applies the affine coefficients of y to x.
        
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
        B, C, H, W = x.size()
        #mean = self.transformation_mean(style_encoding)
        #std = self.transformation_std(style_encoding)
        mean = style_encoding[:, self.idx : self.idx + self.num_channels]
        std = style_encoding[:, self.idx + self.num_channels : self.idx + (2 * self.num_channels)]

        transformed = function.adain_with_coefficients(x, mean, std)
        return transformed


class Decoder(torch.nn.Module):
    """ Decoder network that mirrors the structure of an encoder architecture. """

    def __init__(self, n_layers=19, architecture=torchvision.models.vgg19):
        """ Initializes a decoder model that tries to mirror the encoder architecture.
        
        Parameters:
        -----------
        style_dim : int
            Style embedding dimensionality.
        n_layers : int
            How many layers of the architecture are used for the decoder.
        architecture : torchvision.model
            A torchvision architecture that allows the first n layers to be extracted for the decoder.
        """
        super().__init__()
        self.layers = torch.nn.modules.ModuleList()
        #slice_idx = 0
        conv_layer_added = False # The first layer should be a convolution, ignore the first layers that are non convolutional
        for idx, layer in enumerate(reversed(architecture(pretrained=False, progress=True).features[:n_layers])):
            if isinstance(layer, torch.nn.modules.Conv2d):
                self.layers.append(torch.nn.modules.ReflectionPad2d(1))
                conv = torch.nn.modules.Conv2d(layer.out_channels, layer.in_channels, kernel_size=layer.kernel_size, 
                    stride=layer.stride, dilation=layer.dilation)
                self.layers.append(conv)
                #self.layers.append(AdaInLayer(style_dim, conv.out_channels, slice_idx))
                #slice_idx += conv.out_channels * 2
                #print(slice_idx)
                self.layers.append(torch.nn.InstanceNorm2d(conv.out_channels, affine=True))
                conv_layer_added = True
            elif isinstance(layer, torch.nn.modules.MaxPool2d):
                self.layers.append(torch.nn.modules.UpsamplingNearest2d(scale_factor=layer.stride))
            elif isinstance(layer, torch.nn.modules.ReLU):
                if not conv_layer_added: continue
                self.layers.append(torch.nn.modules.ReLU(inplace=False))
            else:
                raise NotImplementedError('Decoder implementation can not mirror {type(layer)} layer.')

    def forward(self, content, style=None):
        """ Forward pass through the decoder network. 
        
        Parameters:
        -----------
        content : torch.Tensor, shape [batch_size, out_features, width, height]
            A batch of images to decode.
        style : torch.Tensor, shape [batch_size, style_dim] or None
            Optional: A style encoding that is used during AdaIn layers.


        Returns:
        --------
        output : torch.Tensor, shape [batch_size, 3, width * scale, height * scale]
            Decoded images.
        """
        for layer in self.layers:
            if isinstance(layer, AdaInLayer):
                if style is not None:
                    content = layer(content, style)
            else:
                content = layer(content)

        return content
    
