import data, model, loss, function
from torch.utils.data import DataLoader
import torch

# Load training and validation dataset for style and content

VAL_PORTION = 0.2
ITERATIONS = 5000
VAL_ITERATIONS = 10
RESOLUTION = 64
STYLE_DIM = 1024
BATCH_SIZE = 1

CONTENT_LOSS_WEIGHTS = {
    'relu_4_2' : 1.0,
}

STYLE_LOSS_WEIGHTS = {
    'relu_1_1' : 1e3,
    'relu_2_1' : 1e3,
    'relu_3_1' : 1e3,
    'relu_4_1' : 1e3,
    'relu_5_1' : 1e3,
}

data_style = data.load_debug_dataset('../dataset/debug/style', resolution=RESOLUTION)
data_style_train, data_style_val = torch.utils.data.random_split(data_style, [len(data_style) - int(VAL_PORTION * len(data_style)), int(VAL_PORTION * len(data_style))])
data_loader_style_train = DataLoader(data_style_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
data_loader_style_val = DataLoader(data_style_val, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

data_content = data.load_debug_dataset('../dataset/debug/content', resolution=RESOLUTION)
data_content_train, data_content_val = torch.utils.data.random_split(data_content, [len(data_content) - int(VAL_PORTION * len(data_content)), int(VAL_PORTION * len(data_content))])
data_loader_content_train = DataLoader(data_content_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
data_loader_content_val = DataLoader(data_content_val, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

data_loader_train = data.DatasetPairIterator(data_loader_content_train, data_loader_style_train)
data_loader_val = data.DatasetPairIterator(data_loader_content_val, data_loader_style_val)


content_encoder = model.Encoder((3, RESOLUTION, RESOLUTION), pretrained=True)
style_encoder = model.Encoder((3, RESOLUTION, RESOLUTION), pretrained=True, flattened_output_dim=STYLE_DIM)
decoder = model.Decoder(STYLE_DIM)
loss_net = loss.LossNet()
loss_net.eval()

def forward(content, style, content_encoder, style_encoder, decoder):
    """ Forwards a batch through the pipeline.
    
    Parameters:
    -----------
    content : torch.Tensor, shape [B, C, H, W]
        The content image.
    style : torch.Tensor, shape [B, C, H', W']
        The style image, usually H' = H and W' = W.
    content_encoder : torch.nn.modules.Module
        Encoder for content images.
    style_encoder : torch.nn.modules.Module
        Encoder for style images.
    decoder : torch.nn.modules.Module
        Decoder that uses AdaIn to decode the content and apply the style.

    Returns:
    --------
    reco : torch.Tensor, shape [B, C, H, W]
        A reconstruction of the content after application of the style.
    style_representation : torch.Tensor, shape [B, D]
        The latent style representation.
    """
    content_representation = content_encoder(content)
    style_representation = style_encoder(style)
    reco = decoder(content_representation, style_representation)
    return reco, style_representation



# Networks to CUDA device
if torch.cuda.is_available(): 
    content_encoder = content_encoder.cuda()
    style_encoder = style_encoder.cuda()
    decoder = decoder.cuda()
    loss_net = loss_net.cuda()

trainable_parameters = []
for parameter in content_encoder.parameters():
    trainable_parameters.append(parameter)
for parameter in style_encoder.parameters():
    trainable_parameters.append(parameter)
for parameter in decoder.parameters():
    trainable_parameters.append(parameter)


optimizer = torch.optim.Adam(trainable_parameters, lr=1e-3)


iteration = 0
running_perceptual_loss, running_style_loss, running_count = 0.0, 0.0, 0

for (content_image, content_path), (style_image, style_path) in data_loader_train:
    if iteration >= ITERATIONS: break
    iteration += 1
    
    content_encoder.train()
    style_encoder.train()
    decoder.train()


    optimizer.zero_grad()
    if torch.cuda.is_available():
        content_image = content_image.to('cuda')
        style_image = style_image.to('cuda')

   
    transformed, style_representation = forward(content_image, style_image, content_encoder, style_encoder, decoder)

    features_content = loss_net(content_image)
    features_style = loss_net(style_image)
    features_transformed = loss_net(transformed)

    perceptual_loss = loss.perceptual_loss(features_content, features_transformed, CONTENT_LOSS_WEIGHTS)
    style_loss = loss.style_loss(features_style, features_transformed, STYLE_LOSS_WEIGHTS)

    total_loss = perceptual_loss + style_loss

    total_loss.backward()
    optimizer.step()

    running_perceptual_loss += perceptual_loss.item()
    running_style_loss += style_loss.item()

    running_count += 1

    print(f'\r{iteration:06d} : avg perceptual_loss : {running_perceptual_loss / running_count:.4f}\tavg style loss : {running_style_loss / running_count:.4f}', end='\r')

    if iteration % 500 == 1:
        torch.save(content_encoder.state_dict(), f'output/content_encoder_{iteration}')
        torch.save(decoder.state_dict(), f'output/decoder_{iteration}')

    if iteration % 100 == 1:

        for key, weight in STYLE_LOSS_WEIGHTS.items():
            Gx = function.gram_matrix(features_style[key])
            Gy = function.gram_matrix(features_transformed[key])
            value = torch.nn.functional.mse_loss(Gx, Gy)
            print(f'Style loss {key} with weight {weight:.4f}: {value:.4f}')

        running_perceptual_loss, running_style_loss, running_count = 0.0, 0.0, 0 # After each validation, reset running training losses
        print(f'\nValidating...')

        content_encoder.eval()
        style_encoder.eval()
        decoder.eval()
        perceptual_loss = 0.0
        style_loss = 0.0
        val_iteration = 0

        with torch.no_grad():

            torch.save(content_image.cpu(), f'output/{iteration}_0_content.pt')
            torch.save(style_image.cpu(), f'output/{iteration}_0_style.pt')
            torch.save(transformed.cpu(), f'output/{iteration}_0_transformed.pt')

            """
            for (content_image, content_path), (style_image, style_path) in data_loader_val:
                val_iteration += 1
                if val_iteration >= VAL_ITERATIONS: break

                if torch.cuda.is_available():
                    content_image = content_image.to('cuda')
                    style_image = style_image.to('cuda')

                content_representation = content_encoder(content_image)
                style_representation = style_encoder(style_image)

                transformed = model.adaptive_instance_normalization(content_representation, style_representation)
                reconstruction = decoder(transformed)

                for key, value in loss.perceptual_loss(loss_net(content_image), loss_net(reconstruction)).items():
                    perceptual_loss += value
                for key, value in loss.style_loss(loss_net(style_image), loss_net(reconstruction)).items():
                    style_loss += value
                torch.save(content_image.cpu(), f'output/{iteration}_{val_iteration}_content.pt')
                torch.save(style_image.cpu(), f'output/{iteration}_{val_iteration}_style.pt')
                torch.save(reconstruction.cpu(), f'output/{iteration}_{val_iteration}_reconstruction.pt')
                torch.save(decoder(style_representation).cpu(), f'output/{iteration}_{val_iteration}_style_reconstruction.pt')

                print(f'\rValidation {val_iteration:02d} : Perceptual loss {perceptual_loss / val_iteration:.4f}\tStyle loss {style_loss / val_iteration:.4f}', end='\r')
            """
            print('\nValidation done.')
    



    


