from pytorch.transformer import Transformer
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model,
                          num_heads, num_layers, d_ff, max_seq_length, dropout)

# Generate random sample data
# (batch_size, seq_length)
src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))
# (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(),
                       lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

for epoch in tqdm(range(100), desc='Epoch', bar_format='{l_bar}{bar:10}{r_bar}'):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size),
                     tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    tqdm.write(f"Loss: {loss.item()}")

# Save model
torch.save(transformer.state_dict(), 'transformer.pt')
