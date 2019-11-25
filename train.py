import data, model, loss
from torch.utils.data import DataLoader
import torch

data_style = DataLoader(data.load_debug_style_dataset(), batch_size=16, shuffle=True)
data_content = DataLoader(data.load_debug_content_datset(), batch_size=16, shuffle=True)
content_encoder = model.Encoder(pretrained=True)
decoder = model.Decoder()
loss_net = loss.LossNet()
loss_net.eval()


# Networks to CUDA device
if torch.cuda.is_available(): 
    content_encoder = content_encoder.cuda()
    decoder = decoder.cuda()
    loss_net = loss_net.cuda()

trainable_parameters = []
for parameter in content_encoder.parameters():
    trainable_parameters.append(parameter)
for parameter in decoder.parameters():
    trainable_parameters.append(parameter)


optimizer = torch.optim.Adamax(trainable_parameters, lr=1e-3)


content_encoder.train()
decoder.train()


for epoch in range(5):
    for idx, batch in enumerate(data_content):
        optimizer.zero_grad()
        content, paths = batch
        if torch.cuda.is_available():
            content = content.to('cuda')
        content_representation = content_encoder(content)
        reconstruction = decoder(content_representation)

        perceptual_loss = 0.0
        for key, value in loss.perceptual_loss(loss_net(content), loss_net(reconstruction)).items():
            perceptual_loss += value
        perceptual_loss.backward()
        optimizer.step()
        print(f'\r{idx:06d} : perceptual_loss : {perceptual_loss:.4f}', end='\r')
    print('\n\n')


with torch.no_grad():
    content_encoder.eval()
    decoder.eval()
    for idx, batch in enumerate(data_content):
        content, paths = batch
        if torch.cuda.is_available():
            content = content.to('cuda')
        content_representation = content_encoder(content)
        reconstruction = decoder(content_representation)
        torch.save(content, f'content_{idx}.pt')
        torch.save(reconstruction, f'reconst_{idx}.pt')
        break
    



    


