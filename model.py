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
        architecture : torchvision.model
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
        input : torch.Tensor, shape [batch_size, in_features, width, height]
            A batch of images to encode.

        Returns:
        --------
        output : torch.Tensor, shape [batch_size, in_features, width / 4, height / 4]
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
        for layer in reversed(architecture(pretrained=False, progress=True).features[:n_layers]):
            if isinstance(layer, torch.nn.modules.Conv2d): print('Conv2d', layer)
            elif isinstance(layer, torch.nn.modules.MaxPool2d): print('Pooling', layer)
            elif isinstance(layer, torch.nn.modules.ReLU): print('ReLU', layer)
            else: print('Unknown layer', layer)




