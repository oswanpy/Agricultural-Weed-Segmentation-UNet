import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, base_c=64):
        super().__init__()
        self.d1 = DoubleConv(in_c, base_c); self.p1 = nn.MaxPool2d(2, 2)
        self.d2 = DoubleConv(base_c, base_c*2); self.p2 = nn.MaxPool2d(2, 2)
        self.d3 = DoubleConv(base_c*2, base_c*4); self.p3 = nn.MaxPool2d(2, 2)
        self.d4 = DoubleConv(base_c*4, base_c*8); self.p4 = nn.MaxPool2d(2, 2)
        self.bn = DoubleConv(base_c*8, base_c*16)
        self.u4 = nn.ConvTranspose2d(base_c*16, base_c*8, 2, stride=2); self.up4 = DoubleConv(base_c*16, base_c*8)
        self.u3 = nn.ConvTranspose2d(base_c*8, base_c*4, 2, stride=2); self.up3 = DoubleConv(base_c*8, base_c*4)
        self.u2 = nn.ConvTranspose2d(base_c*4, base_c*2, 2, stride=2); self.up2 = DoubleConv(base_c*4, base_c*2)
        self.u1 = nn.ConvTranspose2d(base_c*2, base_c, 2, stride=2); self.up1 = DoubleConv(base_c*2, base_c)
        self.final = nn.Conv2d(base_c, out_c, 1)

    def forward(self, x):
        d1 = self.d1(x); p1 = self.p1(d1)
        d2 = self.d2(p1); p2 = self.p2(d2)
        d3 = self.d3(p2); p3 = self.p3(d3)
        d4 = self.d4(p3); p4 = self.p4(d4)
        bn = self.bn(p4)
        u4 = self.up4(torch.cat([self.u4(bn), d4], dim=1))
        u3 = self.up3(torch.cat([self.u3(u4), d3], dim=1))
        u2 = self.up2(torch.cat([self.u2(u3), d2], dim=1))
        u1 = self.up1(torch.cat([self.u1(u2), d1], dim=1))
        return self.final(u1)