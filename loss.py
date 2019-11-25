import torch
import torchvision

# Layer idxs of the vgg19 net before pooling operations
vgg19_activation_maps = {
    'relu1' : '3',
    'relu2' : '8',
    'relu3' : '17',
    'relu4' : '26',
    'relu5' : '35',
}

class LossNet(torch.nn.Module):
    """ Loss network that consists of the first few layers of a pre-trained vgg19 and outputs the feature activation maps. """

    def __init__(self, activation_maps=['relu1', 'relu2', 'relu3', 'relu4', 'relu5'], architecture=torchvision.models.vgg19):
        """ Initializes the loss network.
        
        Parameters:
        -----------
        activation_maps : iterable
            Which activation maps the loss network outputs.
        architecture : torch.nn.Module
            The model that is used for the perceptual loss. Defaults to the vgg19.
        """
        super().__init__()
        self.layers = architecture(pretrained=True, progress=True).features
        self.activation_maps = [vgg19_activation_maps[key] for key in activation_maps]
    
    def forward(self, x):
        """ Forward pass through the loss network.
        
        Parameters:
        -----------
        x : torch.Tensor, shape [B, 3, H, W]
            A batch of input images.
        
        Returns:
        --------
        activation_maps : dict
            A dict of activation map outputs, where each map is of shape [B, C, H', W']
        """
        activation_maps = {}
        for name, layer in self.layers._modules.items(): # Appearently there is no workarround for accessing a private member
            x = layer(x)
            # print(name, name in self.activation_maps, self.activation_maps, type(name), type(self.activation_maps[0]))
            if name in self.activation_maps:
                activation_maps[name] = x
        return activation_maps


mse_loss = torch.nn.MSELoss(reduction='mean')

def perceptual_loss(feature_activations_x, feature_activations_y):
    """ Computes the perceptual loss between feature activations of two instances x and y.
    
    Parameters:
    -----------
    feature_activations_x : dict
        A dict of feature activation maps of shape [B, C, H', W']
    feature_activations_y : dict
        A dict of feature activation maps of shape [B, C, H', W']

    Returns:
    --------
    losses : dict
        A dict of L2 distances between feature activation maps.
    """
    losses = {}
    for key in feature_activations_x:
        losses[key] = mse_loss(feature_activations_x[key], feature_activations_y[key])
    return losses