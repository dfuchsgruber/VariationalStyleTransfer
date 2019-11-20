import data, model
from torch.utils.data import DataLoader

data_style = DataLoader(data.load_debug_style_dataset(), batch_size=32, shuffle=True)
data_content = DataLoader(data.load_debug_content_datset(), batch_size=32, shuffle=True)
encoder = model.Encoder()
decoder = model.Decoder()

exit(0)

for idx, batch in enumerate(data_content):
    images, paths = batch
    h_content = encoder(images)
    print(h_content.size())
    break
    


