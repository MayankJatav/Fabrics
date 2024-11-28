import torch
import torch.nn as nn
from patchify import patchify

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model % num_heads should be zero"
        self.d_k = self.d_v = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def attention(self, query, key, value):
        score = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k))
        attention_weights = torch.softmax(score, dim=-1)
        context = torch.matmul(attention_weights, value)
        return context

    def forward(self, Q, K, V):
        batch_size, seq_len, _ = Q.size()

        query = self.W_Q(Q).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        key = self.W_K(K).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        value = self.W_V(V).view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)

        x = self.attention(query, key, value)

        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        x = self.fc(x)
        return x

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_in_out, d_model, num_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.linear_in = nn.Linear(d_in_out, d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.linear_out = nn.Linear(d_model, d_in_out)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.mha(x, x, x)
        x = self.linear_out(x)
        return x

class MyModel(nn.Module):
    def __init__(self, patch_shape, d_model=96, num_heads=3):
        super(MyModel, self).__init__()

        self.patch_shape = patch_shape

        self.conv1 = nn.Conv2d(3, 8, 1)
        self.conv2 = nn.Conv2d(8, 8, 1)

        self.mhab1 = MultiHeadAttentionBlock(2048, d_model=d_model, num_heads=num_heads)
        self.mhab2 = MultiHeadAttentionBlock(2048, d_model=d_model, num_heads=num_heads)
        self.mhab3 = MultiHeadAttentionBlock(2048, d_model=d_model, num_heads=num_heads)
        self.mhab4 = MultiHeadAttentionBlock(2048, d_model=d_model, num_heads=num_heads)

        self.conv3 = nn.Conv2d(8, 8, 1)
        self.maxpool1 = nn.MaxPool2d(3)
        self.conv4 = nn.Conv2d(8, 1, 1)
        self.maxpool2 = nn.MaxPool2d(3)

        self.fc = nn.Linear(1764, 3)
        self.softmax = nn.Softmax()

    def create_patches(self, tensor, patch_size):
        patches = tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(tensor.size(0), tensor.size(1), -1, patch_size, patch_size).permute(0, 2, 1, 3, 4)
        return patches

    def forward(self, x):

        input_shape = x.shape

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.create_patches(x, self.patch_shape)
        
        x = x.flatten(2)

        x = self.mhab1(x)
        x = self.mhab2(x)
        x = self.mhab3(x)
        x = self.mhab4(x)

        x = x.reshape((x.shape[0], 8, 384, 384))

        x = self.conv3(x)
        x = self.maxpool1(x)
        x = self.conv4(x)
        x = self.maxpool2(x)

        x = x.flatten(1)
        x = self.fc(x)
        x = self.softmax(x)

        return x
