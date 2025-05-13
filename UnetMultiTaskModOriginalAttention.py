# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 02:02:38 2025
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

# Definir CBAM (Modulo de atención por bloques)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out



def conv_stride(C_in, C_out):
    return torch.nn.Sequential(
        torch.nn.Conv2d(C_in, C_out, kernel_size=2, stride=2)
    )

def conv_3x3(C_in, C_out):
    return torch.nn.Sequential(
        torch.nn.Conv2d(C_in, C_out, 3, padding=1),
        torch.nn.InstanceNorm2d(C_out),
        torch.nn.LeakyReLU(negative_slope=0.01, inplace=True)
    )

def EncoderDoubleConv(C_in, C_out):
    return nn.Sequential(
        conv_3x3(C_in, C_out),
        conv_3x3(C_out, C_out)
    )

class Decoder_UpSampling(torch.nn.Module):
    def __init__(self, C_in, C_out):
        super(Decoder_UpSampling, self).__init__()
        self.upsample = torch.nn.ConvTranspose2d(C_in, C_out, 2, stride=2)
        self.conv1 = conv_3x3(C_in, C_out)
        self.conv2 = conv_3x3(C_out, C_out)
    
    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX, 0, diffY, 0))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MultiTaskUNET(torch.nn.Module):
    def __init__(self, num_classes_seg=1, num_classes_skeleton=1, in_ch=1):
        super(MultiTaskUNET, self).__init__()
        
        #flts= [64, 128, 256, 512]
        #flts= [32, 64, 128, 256]
        flts= [16, 32, 64, 128]
        
        # Encoder
        self.enc1 = EncoderDoubleConv(in_ch, flts[0])
        self.cbam1 = CBAM(flts[0])
        self.enc2 = EncoderDoubleConv(flts[0], flts[1])
        self.cbam2 = CBAM(flts[1])
        self.enc3 = EncoderDoubleConv(flts[1], flts[2])
        self.cbam3 = CBAM(flts[2])
        self.enc4 = EncoderDoubleConv(flts[2], flts[3])
        self.cbam4 = CBAM(flts[3])

        # Bottleneck
        self.bottleneck = EncoderDoubleConv(flts[3], 2*flts[3])
        
        # DecoderSeg
        self.dec4 = Decoder_UpSampling(2*flts[3], flts[3])
        self.dec3 = Decoder_UpSampling(flts[3], flts[2])
        self.dec2 = Decoder_UpSampling(flts[2], flts[1])
        self.dec1 = Decoder_UpSampling(flts[1], flts[0])
        
        # DecoderSkel
        self.dec4S = Decoder_UpSampling(2*flts[3], flts[3])
        self.dec3S = Decoder_UpSampling(flts[3], flts[2])
        self.dec2S = Decoder_UpSampling(flts[2], flts[1])
        self.dec1S = Decoder_UpSampling(flts[1], flts[0])
        
        # última capa conv que nos da la máscara de segmentacinón
        self.out1 = torch.nn.Conv2d(flts[0], num_classes_seg, 1, padding=0)
        # última capa conv que nos da la máscara del esqueleto
        self.out2 = torch.nn.Conv2d(flts[0], num_classes_skeleton, 1, padding=0)
        
        # MaxPooling
        #self.pool = nn.MaxPool2d(2)
        # Convoluciones con stride en lugar de MaxPooling
        self.pool1 = conv_stride(flts[0], flts[0])
        self.pool2 = conv_stride(flts[1], flts[1])
        self.pool3 = conv_stride(flts[2], flts[2])
        self.pool4 = conv_stride(flts[3], flts[3])
        
    def forward(self, x):
        # Encoder
        x1E = self.enc1(x)
        x1E =self.cbam1(x1E)
        x2E = self.pool1(x1E)
        
        x2E = self.enc2(x2E)
        x2E =self.cbam2(x2E)
        x3E = self.pool2(x2E)
        
        x3E = self.enc3(x3E)
        x3E =self.cbam3(x3E)
        x4E = self.pool3(x3E)
        
        x4E = self.enc4(x4E)
        x4E =self.cbam4(x4E)

        # Bottleneck
        x5E = self.pool4(x4E)
        x5E = self.bottleneck(x5E)
        
        # DecoderSeg
        x4 = self.dec4(x5E,x4E)
        x3 = self.dec3(x4,x3E)
        x2 = self.dec2(x3,x2E)
        x1 = self.dec1(x2,x1E)
        
        # DecoderSkel
        x4S = self.dec4S(x5E,x4E)
        x3S = self.dec3S(x4S,x3E)
        x2S = self.dec2S(x3S,x2E)
        x1S = self.dec1S(x2S,x1E)
        
        # Salidas de la segmentación y esqueleto
        xa = self.out1(x1)
        xb = self.out2(x1S)
        return xa, xb


# Ejemplo de uso
if __name__ == "__main__":
    inputs = torch.randn((4, 1, 512, 512))  # Batch de imágenes de ejemplo
    model = MultiTaskUNET()  # InstanC_inar el modelo
    y, z = model(inputs)  # Forward pass
    print(y.shape)  # Salida de la máscara de segmentación
    print(z.shape)  # Salida de la máscara de esqueleto
    
    summary(model, input_size = (1, 512, 512))
