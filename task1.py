import torch
import torch.nn as nn

class CustomAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(CustomAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size should be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class SimpleFeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_dim):
        super(SimpleFeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_dim)
        self.fc2 = nn.Linear(ff_hidden_dim, embed_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MyGPTLayer(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_dim, dropout):
        super(MyGPTLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.custom_attention = CustomAttention(embed_size, heads)
        self.simple_ff = SimpleFeedForward(embed_size, ff_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.custom_attention(x, x, x, mask)
        x = x + self.dropout(x2)
        x = self.norm1(x)

        x2 = self.simple_ff(x)
        x = x + self.dropout(x2)
        x = self.norm2(x)

        return x


class MyGPTModel(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_dim, num_layers, vocab_size, dropout):
        super(MyGPTModel, self).__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.layers = nn.ModuleList(
            [MyGPTLayer(embed_size, heads, ff_hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(device)
        x = self.embedding(x, positions)

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
print("Output shape:", output.shape)
