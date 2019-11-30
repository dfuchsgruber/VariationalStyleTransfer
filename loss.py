import torch
import torchvision
import function

# Layer idxs of the vgg19 net before pooling operations
vgg19_activation_layer_names = {
    '3' : 'relu1',
    '8' : 'relu2',
    '17' : 'relu3',
    '26' : 'relu4',
    '35' : 'relu5',
}

class LossNet(torch.nn.Module):
    """ Loss network that consists of the first few layers of a pre-trained vgg19 and outputs the feature activation maps. """

    def __init__(self, architecture=torchvision.models.vgg19):
        """ Initializes the loss network.
        
        Parameters:
        -----------
        architecture : torch.nn.Module
            The model that is used for the perceptual loss. Defaults to the vgg19.
        """
        super().__init__()
        self.layers = architecture(pretrained=True, progress=True).features
        for parameter in self.parameters():
            parameter.requires_grad = False
    
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
        activation_maps = {'input' : x}
        for name, layer in self.layers._modules.items(): # Appearently there is no workarround for accessing a private member
            x = layer(x)
            if name in vgg19_activation_layer_names:
                activation_maps[vgg19_activation_layer_names[name]] = x
        return activation_maps
    


def perceptual_loss(feature_activations_x, feature_activations_y, weights : dict):
    """ Computes the perceptual loss between feature activations of two instances x and y. ONly activations with
    weight are considered.
    
    Parameters:
    -----------
    feature_activations_x : dict
        A dict of feature activation maps of shape [B, C, H', W']
    feature_activations_y : dict
        A dict of feature activation maps of shape [B, C, H', W']
    weights : dict
        A dict for weights of each feature map distance.

    Returns:
    --------
    loss : float
        Weighted sum of L2 feature map distances.
    """
    loss = 0.0
    for key, weight in weights.items():
        loss += weight * torch.nn.functional.mse_loss(feature_activations_x[key], feature_activations_y[key])
    return loss

def style_loss(feature_activations_x, feature_activations_y, weights : dict):
    """ Computes the style loss between feature activations of two instances x and y. That is, for each pair of feature maps,
    the gram matrices of the flattened maps are computed and the L2 distance is calculated.

    Parameters:
    -----------
    feature_activations_x : dict
        A dict of feature activation maps of shape [B, C, H', W']
    feature_activations_y : dict
        A dict of feature activation maps of shape [B, C, H', W']
    weights : dict
        A dict for weights of each feature map distance.

    Returns:
    --------
    loss : float
        Weighted sum of L2 gram matrix distances.
    """
    loss = 0.0
    for key, weight in weights.items():
        Gx = function.gram_matrix(feature_activations_x[key])
        Gy = function.gram_matrix(feature_activations_y[key])
        loss += weight * torch.nn.functional.mse_loss(Gx, Gy)
    return loss
    


"""
def adain_style_loss(feature_activations_x, feature_activations_y):
    losses = {}
    for key in feature_activations_x:
        mean_x, std_x = model.instance_mean_and_std(feature_activations_x[key])
        mean_y, std_y = model.instance_mean_and_std(feature_activations_y[key])
        losses[key] = mse_loss(mean_x, mean_y) + mse_loss(std_x, std_y)
    return losses
"""