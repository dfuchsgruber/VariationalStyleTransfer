import torchvision.models
import torch

class Encoder(torch.nn.Module):
    """ Encoder network that contains of the first few layers of the vgg19 [1] network. 
    
    References:
    -----------
    [1] : https://arxiv.org/pdf/1409.1556.pdf
    """
    
    
    def __init__(self, n_layers=12, architecture=torchvision.models.vgg19, pretrained=True):
        """ Initializes an encoder model based on some (pretrained) architecture. 
        
        Parameters:
        -----------
        n_layers : int
            How many layers of the architecture are used for the encoder.
        architecture : torch.nn.module
            A torchvision architecture that allows the first n layers to be extracted.
        pretrained : bool
            If True, pretrained weights of the architecture are used.
        """
        super().__init__()
        self.layers = architecture(pretrained=pretrained, progress=True).features[:n_layers]

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
        return self.layers(input)
        

class Decoder(torch.nn.Module):
    """ Decoder network that mirrors the structure of an encoder architecture. """

    def __init__(self, n_layers=12, architecture=torchvision.models.vgg19):
        """ Initializes a decoder model that tries to mirror the encoder architecture.
        
        Parameters:
        -----------
        n_layers : int
            How many layers of the architecture are used for the decoder.
        architecture : torchvision.model
            A torchvision architecture that allows the first n layers to be extracted for the decoder.
        """
        super().__init__()
        self.layers = torch.nn.modules.Sequential()
        conv_layer_added = False # The first layer should be a convolution, ignore the first layers that are non convolutional
        for idx, layer in enumerate(reversed(architecture(pretrained=False, progress=True).features[:n_layers])):
            if isinstance(layer, torch.nn.modules.Conv2d):
                self.layers.add_module(f'{idx}_ReflectionPadding', torch.nn.modules.ReflectionPad2d(1))
                conv = torch.nn.modules.Conv2d(layer.out_channels, layer.in_channels, kernel_size=layer.kernel_size, 
                    stride=layer.stride, dilation=layer.dilation)
                self.layers.add_module(f'{idx}_Conv2d', conv)
                #self.layers.add_module(f'{idx}_IN', torch.nn.modules.InstanceNorm2d(layer.in_channels, affine=True))
                conv_layer_added = True
            elif isinstance(layer, torch.nn.modules.MaxPool2d): 
                if not conv_layer_added: continue
                self.layers.add_module(f'{idx}_UpsamplingNearest2d', torch.nn.modules.UpsamplingNearest2d(scale_factor=layer.stride))
            elif isinstance(layer, torch.nn.modules.ReLU):
                self.layers.add_module(f'{idx}_ReLU' ,torch.nn.modules.ReLU(inplace=False))
            else:
                raise NotImplementedError('Decoder implementation can not mirror {type(layer)} layer.')

    def forward(self, input):
        """ Forward pass through the decoder network. 
        
        Parameters:
        -----------
        input : torch.Tensor, shape [batch_size, out_features, width, height]
            A batch of images to decode.

        Returns:
        --------
        output : torch.Tensor, shape [batch_size, 3, width * scale, height * scale]
            Decoded images.
        """
        return self.layers(input)
    

