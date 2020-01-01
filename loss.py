import torch
import torchvision
import function
from data import vgg_normalization_mean, vgg_normalization_std

# Layer idxs of the vgg19 net before pooling operations
vgg19_activation_layer_names = {
    '1': 'relu_1_1',
    '3': 'relu_1_2',
    '4': 'maxpool_1',
    '6': 'relu_2_1',
    '8': 'relu_2_2',
    '9': 'maxpool_2',
    '11': 'relu_3_1',
    '13': 'relu_3_2',
    '15': 'relu_3_3',
    '17': 'relu_3_4',
    '18': 'maxpool_3',
    '20': 'relu_4_1',
    '22': 'relu_4_2',
    '24': 'relu_4_3',
    '26': 'relu_4_4',
    '27': 'maxpool_4',
    '29': 'relu_5_1',
    '31': 'relu_5_2',
    '33': 'relu_5_3',
    '35': 'relu_5_4',
    '36': 'maxpool_5',
}

class LossNet(torch.nn.Module):
    """ Loss network that consists of the first few layers of a pre-trained vgg19 and outputs the feature activation maps. """

    def __init__(self, architecture=torchvision.models.vgg19, normalize=True):
        """ Initializes the loss network.
        
        Parameters:
        -----------
        architecture : torch.nn.Module
            The model that is used for the perceptual loss. Defaults to the vgg19.
        normalize : bool
            If True, the images are normalized using the vgg normalization.
        """
        super().__init__()
        self.layers = architecture(pretrained=True, progress=True).features
        self.normalize = normalize
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
        if self.normalize:
            mu = torch.Tensor(vgg_normalization_mean).to(x.device).reshape(1, 3, 1, 1)
            sigma = torch.Tensor(vgg_normalization_std).to(x.device).reshape(1, 3, 1, 1)
            x = (x - mu) / sigma

        activation_maps = {'input' : x}
        for name, layer in self.layers._modules.items(): # Appearently there is no workarround for accessing a private member
            x = layer(x)
            if name in vgg19_activation_layer_names:
                activation_maps[vgg19_activation_layer_names[name]] = x
        return activation_maps
    


def perceptual_loss(feature_activations_x, feature_activations_y, weights : dict, loss='l2'):
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
    loss : 'l1' or 'l2'
        Which kind of loss to apply to the perceptual features.

    Returns:
    --------
    loss : float
        Weighted sum of L2 feature map distances.
    """
    if loss in ('l1', 'L1'):
        loss_fn = torch.nn.functional.l1_loss
    elif loss in ('l2', 'L2', 'mse'):
        loss_fn = torch.nn.functional.mse_loss
    else:
        raise RuntimeError(f'Unknown loss reduction {loss}')

    loss = 0.0
    for key, weight in weights.items():
        loss += weight * loss_fn(feature_activations_x[key], feature_activations_y[key])
    return loss

def style_loss(feature_activations_x, feature_activations_y, weights : dict, loss):
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
    loss : 'l1' or 'l2'
        The metric to measure distances in gram matrix space.

    Returns:
    --------
    loss : float
        Weighted sum of L2 gram matrix distances.
    """
    if loss in ('l1', 'L1'):
        loss_fn = torch.nn.functional.l1_loss
    elif loss in ('l2', 'L2', 'mse'):
        loss_fn = torch.nn.functional.mse_loss
    else:
        raise RuntimeError(f'Unknown loss reduction {loss}')

    loss = 0.0
    for key, weight in weights.items():
        Gx = function.gram_matrix(feature_activations_x[key])
        Gy = function.gram_matrix(feature_activations_y[key])
        loss += weight * loss_fn(Gx, Gy)
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