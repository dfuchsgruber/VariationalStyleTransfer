import torchvision.models
import torch
import function
import torch.nn.functional as F


class ResNetEncoder(torch.nn.Module):
    """ Encoder that uses a pretrained ResNet architecture. """

    def __init__(self, embedding_dim, architecture=torchvision.models.resnet50, pretrained=True):
        """ Initializes the ResNet encoder.
        
        Parameters:
        -----------
        embedding_dim : int
            Embedding dimensionality.
        architecture : torch.nn.Module
            A torchvision.models.resnetXX architecture to use.
        pretrained : bool
            If true, uses pre-trained weights of the torchvision model.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.resnet = architecture(pretrained=pretrained)
        # Replace the fully connected layer that tries to predict 1000 class labels with a layer to the embedding dimensionality
        if architecture is torchvision.models.resnet18 or architecture is torchvision.models.resnet34 or architecture is torchvision.models.resnet152:
            fc_input_dim = 512
        elif architecture is torchvision.models.resnet50 or architecture is torchvision.models.resnet101:
            fc_input_dim = 2048
        else:
            raise NotImplementedError

        self.resnet.fc = torch.nn.Linear(fc_input_dim, self.embedding_dim)

    def forward(self, x):
        return self.resnet(x)


class Decoder(torch.nn.Module):
    """ General purpose decoder that uses AdaIn layers to modify the embedding multiple times and then uses
    TransposeConvs to create the output image. """

    def __init__(self, content_dim, style_dim, resolution):
        """ Initializes the generic decoder.
        
        Parameters:
        -----------
        content_dim : int
            The content embedding dimensionality.
        style_dim : int
            The style embedding dimensionality.
        resolution : int or tuple of ints (H, W)
            The output resolution, a multiple of 32.
        """
        super().__init__()
        # Each TransposeConv2D upsamples the image by a factor of 2, so does the Upsampling layer, i.e. the image is upscaled by a factor of 32
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        self.resolution = resolution
        self.content_dim = content_dim
        self.style_dim = style_dim
        self.fc = torch.nn.Linear(self.content_dim, (resolution[0] // 32) * (resolution[1] // 32) * 512)

        self.adain4 = AdaInLayer(self.style_dim, 512)
        self.adain3 = AdaInLayer(self.style_dim, 512)
        self.adain2 = AdaInLayer(self.style_dim, 256)
        self.adain1 = AdaInLayer(self.style_dim, 128)

        """    
        self.upsample1 = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.pad = torch.nn.ReflectionPad2d(1)
        self.conv4 = torch.nn.Conv2d(512, 512, (3, 3), stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(512, 256, (3, 3), stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(256, 128, (3, 3), stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(128, 64, (3, 3), stride=1, padding=0)
        # Two convolutions after last adaptive instance normalization layer
        self.conv_0_1 = torch.nn.Conv2d(64, 64, (3, 3), stride=1, padding=0)
        self.conv_0_2 = torch.nn.Conv2d(64, 3, (3, 3), stride=1, padding=0)
        """

        self.tconv4 = torch.nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, dilation=1)
        self.tconv3 = torch.nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, dilation=1)
        self.tconv2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, dilation=1)
        self.tconv1 = torch.nn.ConvTranspose2d(64, 3, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, dilation=1)


    def forward(self, content_encoding, style_encoding):
        """ Forward pass.
        
        Parameters:
        -----------
        content_encoding : torch.Tensor, shape [batch_size, content_dim]
            Encoding of the content image.
        style_encoding : torch.Tensor, shape [batch_size, style_dim]
            Style encoding of the style image to apply to the content image.
        
        Returns:
        --------
        stylized : torch.Tensor, shape [batch_size, 3, H, W]
            The stylized output image, where H and W are specified by the resolution given to the decoder initialization.
        """

        x = self.fc(content_encoding).view(-1, 512, self.resolution[0] // 32, self.resolution[1] // 32)
        x = F.relu(x, inplace=True)

        x = self.upsample1(x)
        x = F.relu(self.tconv4(x), inplace=True)
        x = F.relu(self.tconv3(x), inplace=True)
        x = F.relu(self.tconv2(x), inplace=True)
        x = F.relu(self.tconv1(x), inplace=True)

        """
        if style_encoding is not None: x = self.adain4(x, style_encoding)
        x = self.upsample1(x)
        x = F.relu(self.conv4(self.pad(x)), inplace=True)
        
        if style_encoding is not None: x = self.adain3(x, style_encoding)
        x = self.upsample1(x)
        x = F.relu(self.conv3(self.pad(x)), inplace=True)

        if style_encoding is not None: x = self.adain2(x, style_encoding)
        x = self.upsample1(x)
        x = F.relu(self.conv2(self.pad(x)), inplace=True)

        if style_encoding is not None: x = self.adain1(x, style_encoding)
        x = self.upsample1(x)
        x = F.relu(self.conv1(self.pad(x)), inplace=True)

        x = self.upsample1(x)
        x = F.relu(self.conv_0_1(self.pad(x)), inplace=True)
        x = F.relu(self.conv_0_2(self.pad(x)), inplace=True)
        """

        return x


class VGGEncoder(torch.nn.Module):
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
        self.fc = torch.nn.Linear(self.style_dim, 2 * self.num_channels)
    
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
        B, C, H, W = x.size()
<<<<<<< HEAD
        affine_params = self.fc(style_encoding)
        mean = affine_params[:, : self.num_channels]
        std = affine_params[:, self.num_channels : ]
=======
        #mean = self.transformation_mean(style_encoding)
        #std = self.transformation_std(style_encoding)
        mean = style_encoding[:, self.idx : self.idx + self.num_channels]
        std = style_encoding[:, self.idx + self.num_channels : self.idx + (2 * self.num_channels)]

>>>>>>> e997552c9500011202c2d52aa116336fb0391f40
        transformed = function.adain_with_coefficients(x, mean, std)
        return transformed


class VGGDecoder(torch.nn.Module):
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
    
