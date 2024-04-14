import torch
from torch import nn


class ECA(nn.Module):
    def __init__(self, c1, gamma=2, b=1):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 根据 gamma 和 b 计算卷积核大小
        kernel_size = int(abs((torch.log2(torch.tensor(c1).float()) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.conv = nn.Conv1d(c1, c1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = self.sigmoid(self.conv(y)).unsqueeze(-1)
        return x * y


if __name__ == '__main__':
    x = torch.ones((2, 256, 8, 8))
    print(x.shape)
    print(x)
    mod = ECA(256)
    out = mod(x)
    print(out.shape)
    print(out)
