import torch 

def gram_matrix(activations):
    """ Computes the gram matrix of feature activations. 
    
    Parameters:
    -----------
    activations : torch.Tensor, shape [B, C, H, W]
        The feature activations.
    
    Returns:
    --------
    gram_matrix : torch.Tensor, shape [C, C]
        The gram matrix of the activations.
    """
    B, C, H, W = activations.size()
    activations = activations.view(B, C, H * W)
    gram_matrix = torch.bmm(activations, activations.transpose(1, 2)) # Batchwise inner products of flattened activations
    return gram_matrix / (C * H * W + 1e-20)



def instance_mean_and_std(x):
    """ Calculates the mean and standard deviation of a batch of images over the image dimensions.
    Dimensions of the statistics are expanded to fit the input size.
    
    Parameters:
    -----------
    x : torch.Tensor, shape [B, C, H, W]
        The tensor to get the mean and standard deviation of.
    
    Returns:
    --------
    x_mean : torch.Tensor, shape [B, C, H, W]
        The instance means of x.
    x_std : torch.Tensor, shape [B, C, H, W]
        The instance standard deviations of x.
    """
    B, C, H, W = x.size()
    if H == 1 and W == 1:
        raise ValueError("height and width in decoder are 1, this means that the variance in the AdaIn layer becomes NaN. Use fewer downconvolutions or a higher resolution!")
    x_var, x_mean = torch.var_mean(x.view(B, C, -1), -1, keepdim=True)
    return x_mean.unsqueeze(-1), x_var.sqrt().unsqueeze(-1)

"""
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)
"""

def adain(x, y):
    """ Applies Adaptive instance normalization to x with the affine parameters of y. Both mean and variance 
    of per channel and instance in a batch (i.e. the summation is done over the image dimensions only) are 
    calculated for x and y. Afterwards, x is zero-centered and normalized to have a scale of 1.0. 
    Lastly, the centered \bar{x} is shifted by the mean and scaled by the standard deviation obtained by
    the instance normalization of y.

    Parameters:
    -----------
    x : torch.Tensor, shape [B, C, H, W]
        The tensor which is to be normalized and transformed.
    y : torch.Tensor, shape [B, C, H', W']
        The tensor from which the affine transformation is gained.

    Returns:
    --------
    z : torch.Tensor, shape [B, C, H, W]
        The transformed version of x.
    """
    x_mean, x_std = instance_mean_and_std(x)
    y_mean, y_std = instance_mean_and_std(y)
    x = x - x_mean # Centering
    x /= x_std + 1e-12 # Normalizing
    x *= y_std # Scaling
    x += y_mean # Offseting
    #print(x.mean(), y.mean(), x_mean.mean(), x_std.mean(), y_mean.mean(), y_std.mean())
    #exit(0)
    return x

def adain_mean_std(x, mean, std):
    """ Applies Adaptive instance normalization to x with the affine parameters of y. Both mean and variance 
    of per channel and instance in a batch (i.e. the summation is done over the image dimensions only) are 
    calculated for x and y. Afterwards, x is zero-centered and normalized to have a scale of 1.0. 
    Lastly, the centered \bar{x} is shifted by the mean and scaled by the standard deviation obtained by
    the instance normalization of y.

    Parameters:
    -----------
    x : torch.Tensor, shape [B, C, H, W]
        The tensor which is to be normalized and transformed.
    y : torch.Tensor, shape [B, C, H', W']
        The tensor from which the affine transformation is gained.

    Returns:
    --------
    z : torch.Tensor, shape [B, C, H, W]
        The transformed version of x.
    """
    x_mean, x_std = instance_mean_and_std(x)
    x = x - x_mean # Centering
    x /= x_std + 1e-12 # Normalizing
    x *= std # Scaling
    x += mean # Offseting
    #print(x.mean(), y.mean(), x_mean.mean(), x_std.mean(), y_mean.mean(), y_std.mean())
    #exit(0)
    return x

def adain_with_coefficients(x, mean, std):
    """ Applies Adaptive Instance Normalization given a offset and scaling parameter. 
    
    Parameters:
    -----------
    x : torch.Tensor, shape [B, C, H, W]
        The tensor to be transformed.
    mean : torch.Tensor, shape [B, C]
        Offset parameter to be applied to x.
    std : torch.Tensor, shape [B, C]
        Scaling parameter to be applied to x.
    
    Returns:
    --------
    z : torch.Tensor, shape [B, C, H, W]
        Transformed version of x.
    """
    B, C, H, W = x.size()
    x_mean, x_std = instance_mean_and_std(x)
    x = (x - x_mean) / (x_std + 1e-12) # Normalize x
    x *= std.view(B, C, 1, 1)
    x += mean.view(B, C, 1, 1)
    return x

