import torch
import torch.nn as nn
from src import config


class PhysNet(nn.Module):

    def __init__(self):
        super().__init__()
        C = config.PHYSNET_BASE_CHANNELS   # 32 by default

        self.stem = nn.Sequential(
            nn.Conv1d(3, C, kernel_size=1, bias=False),
            nn.BatchNorm1d(C),
            nn.ELU(inplace=True),
        )

        # encoder 
        self.enc1 = PhysBlock(C,     C * 2, stride=2)
        self.enc2 = PhysBlock(C * 2, C * 4, stride=2)
        self.enc3 = PhysBlock(C * 4, C * 4, stride=2)

        # decoder 
        self.dec3 = PhysUpBlock(C * 4, C * 4)
        self.dec2 = PhysUpBlock(C * 4 + C * 4, C * 2)
        self.dec1 = PhysUpBlock(C * 2 + C * 2, C)

        # output head 
        self.head = nn.Conv1d(C + C, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        s = self.stem(x)      
        e1 = self.enc1(s)      
        e2 = self.enc2(e1)    
        e3 = self.enc3(e2)     
        d3 = self.dec3(e3)                             
        d2 = self.dec2(torch.cat([d3, e2], dim=1))    
        d1 = self.dec1(torch.cat([d2, e1], dim=1))     
        out = self.head(torch.cat([d1, s], dim=1))     
        return out[:, 0, :]                           


class PhysBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ELU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ELU(inplace=True),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class PhysUpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ELU(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ELU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.up(x)))
        return self.conv(x)
