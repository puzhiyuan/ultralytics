# import torch.nn as nn  
# import torch  

# class GAM_Attention(nn.Module):  
#     def __init__(self, in_channels, out_channels, rate=4):  
#         super(GAM_Attention, self).__init__()  

#         self.channel_attention = nn.Sequential(  
#             nn.Linear(in_channels, int(in_channels / rate)),  
#             nn.ReLU(inplace=True),  
#             nn.Linear(int(in_channels / rate), in_channels)  
#         )  
      
#         self.spatial_attention = nn.Sequential(  
#             nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),  
#             nn.BatchNorm2d(int(in_channels / rate)),  
#             nn.ReLU(inplace=True),  
#             nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),  
#             nn.BatchNorm2d(out_channels)  
#         )  
      
#     def forward(self, x):  
#         b, c, h, w = x.shape  
#         x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)  
#         x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)  
#         x_channel_att = x_att_permute.permute(0, 3, 1, 2)  
      
#         x = x * x_channel_att  
      
#         x_spatial_att = self.spatial_attention(x).sigmoid()  
#         out = x * x_spatial_att  
      
#         return out  

  

# if __name__ == '__main__':  
#     x = torch.randn(1, 64, 32, 48)  
#     b, c, h, w = x.shape  
#     net = GAM_Attention(in_channels=c, out_channels=c)  
#     y = net(x)
#     print(y.shape)




import numpy as np
import torch
from torch import nn
from torch.nn import init

class GAMAttention(nn.Module):
       #https://paperswithcode.com/paper/global-attention-mechanism-retain-information
    def __init__(self, c1, c2, group=True,rate=4):
        super(GAMAttention, self).__init__()
        
        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / rate), c1)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c1, c1//rate, kernel_size=7, padding=3,groups=rate)if group else nn.Conv2d(c1, int(c1 / rate), kernel_size=7, padding=3), 
            nn.BatchNorm2d(int(c1 /rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1//rate, c2, kernel_size=7, padding=3,groups=rate) if group else nn.Conv2d(int(c1 / rate), c2, kernel_size=7, padding=3), 
            nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att
 
        x_spatial_att = self.spatial_attention(x).sigmoid()
        x_spatial_att=channel_shuffle(x_spatial_att,4) #last shuffle 
        out = x * x_spatial_att
        return out  

def channel_shuffle(x, groups=2):
        B, C, H, W = x.size()
        out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
        out=out.view(B, C, H, W) 
        return out

