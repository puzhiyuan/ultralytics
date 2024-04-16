import torch
import torch.nn as nn
from torch.nn import functional as F


# --------------------------------#
# CoTAttention的定义
# 输入与输出的张量shape保持一致
# --------------------------------#
class CoT(nn.Module):
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.factor = 4

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=self.kernel_size, padding=self.kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // self.factor, 1, bias=False),
            nn.BatchNorm2d(2 * dim // self.factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // self.factor, self.kernel_size * self.kernel_size * dim, 1)
        )

    def forward(self, x):
        bs, c, h, w = x.shape

        k1 = self.key_embed(x)  # bs,c,h,w
        v = self.value_embed(x).view(bs, c, -1)  # bs,c,h*w
        y = torch.cat([k1, x], dim=1)  # bs,2c,h,w

        att = self.attention_embed(y)  # bs,c*k*k,h,w
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # bs,c,h*w

        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w)

        return k1 + k2


if __name__ == '__main__':
    input_tensor = torch.randn(8, 512, 4, 4)
    attention_net = CoT(dim=512)
    output_tensor = attention_net(input_tensor)
    print(output_tensor.shape)  # torch.Size([8, 512, 4, 4])
