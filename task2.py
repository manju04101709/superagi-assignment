import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, embed_size):
        super(RotaryEmbedding, self).__init__()
        self.embed_size = embed_size
        self.freqs = torch.pow(10000.0, torch.arange(0, embed_size, 2.0) / embed_size)

    def forward(self, positions):
        angles = positions.unsqueeze(1) / self.freqs.unsqueeze(0)
        sine_vals = torch.sin(angles)
        cosine_vals = torch.cos(angles)
        pos_encodings = torch.cat([sine_vals, cosine_vals], dim=-1)
        return pos_encodings

class MyGPTModel(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_dim, num_layers, vocab_size, dropout):
        super(MyGPTModel, self).__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rotary_embedding = RotaryEmbedding(embed_size)
        self.layers = nn.ModuleList(
            [MyGPTLayer(embed_size, heads, ff_hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(device)
        rotary_pos_encodings = self.rotary_embedding(positions)
        x = self.embedding(x) + rotary_pos_encodings

        for layer in self.layers:
            x = layer(x, mask)

        x = self.fc_out(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_size = 256
heads = 8
ff_hidden_dim = 512
num_layers = 6
vocab_size = 10000
dropout = 0.1
my_gpt_model = MyGPTModel(embed_size, heads, ff_hidden_dim, num_layers, vocab_size, dropout).to(device)

sample_input = torch.randint(0, vocab_size, (32, 20)).to(device)
mask = torch.ones((32, 20)).to(device)
output = my_gpt_model(sample_input, mask)
print("Output shape with rotary positional embeddings:", output.shape)
