import torch
import torch.nn as nn

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

class ConvolutionalBlockIn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvolutionalBlockIn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvolutionalBlockOut(nn.Module):
    def __init__(self):
        super(ConvolutionalBlockOut, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.max_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool(x)
        return x

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, patch_shape, in_channels, d_model, num_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.linear_in = nn.Linear(patch_shape[0]*patch_shape[1]*in_channels, d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.linear_out = nn.Linear(d_model, patch_shape[0]*patch_shape[1]*in_channels)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.mha(x, x, x)
        x = self.linear_out(x)
        return x

class Model(nn.Module):
    def __init__(self, patch_shape, in_channels, out_channels, d_model, num_heads):
        super(Model, self).__init__()

        self.patch_shape = patch_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvolutionalBlockIn(in_channels, out_channels, 1)
        self.conv2 = ConvolutionalBlockIn(in_channels, out_channels, 3)
        self.conv3 = ConvolutionalBlockIn(in_channels, out_channels, 5)

        self.mha1 = MultiHeadAttentionBlock(self.patch_shape, out_channels, d_model, num_heads)
        self.mha2 = MultiHeadAttentionBlock(self.patch_shape, out_channels, d_model, num_heads)
        self.mha3 = MultiHeadAttentionBlock(self.patch_shape, out_channels, d_model, num_heads)

        self.conv_out = ConvolutionalBlockOut()
        self.fc = nn.Linear(128, 3)

        self.num_heads = num_heads

    def forward(self, x):
        input_shape = x.shape
        
        x1 = self.conv1(x)
        x1 = torch.tensor(create_patches(x1.permute(0, 2, 3, 1).detach().numpy(), patch_shape)).flatten(2)
        x1 = self.mha1(x1)

        x2 = self.conv2(x)
        x2 = torch.tensor(create_patches(x2.permute(0, 2, 3, 1).detach().numpy(), patch_shape)).flatten(2)
        x2 = self.mha2(x2)
        
        x3 = self.conv3(x)
        x3 = torch.tensor(create_patches(x3.permute(0, 2, 3, 1).detach().numpy(), patch_shape)).flatten(2)
        x3 = self.mha3(x3)

        x = x1 + x2 + x3
        x = x.reshape((input_shape[0],) + (self.out_channels,) + input_shape[2:])

        x = self.conv_out(x).flatten()
        x = self.fc(x)

        return x