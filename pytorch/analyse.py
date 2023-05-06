import torch
from pytorch.transformer import Transformer

model_path = 'transformer.pt'

# load model from file and visualize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(5000, 5000, 512, 8, 6, 2048, 100, 0.1)
model.load_state_dict(torch.load(model_path, map_location=device))

# define some sample data
src_data = torch.randint(1, 5000, (64, 100))

output = model(src_data, src_data[:, :-1])
