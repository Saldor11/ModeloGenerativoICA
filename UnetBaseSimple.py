# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 01:29:39 2025

@author: JSALVADORRC
"""

import torch
from torch import nn
from torch.nn import functional as F

from torchsummary import summary

def conv_3x3(C_in, C_out):
    return torch.nn.Sequential(
        torch.nn.Conv2d(C_in, C_out, 3, padding=1),
        torch.nn.BatchNorm2d( C_out),
        torch.nn.ReLU(inplace=True)
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
        Xnew = x2.size()[2] - x1.size()[2]
        Ynew = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (Xnew, 0, Ynew, 0))
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
        self.enc2 = EncoderDoubleConv(flts[0], flts[1])
        self.enc3 = EncoderDoubleConv(flts[1], flts[2])
        self.enc4 = EncoderDoubleConv(flts[2], flts[3])

        # Bottleneck
        self.bottleneck = EncoderDoubleConv(flts[3], 2*flts[3])
        
        # DecoderSeg
        self.dec4 = Decoder_UpSampling(2*flts[3], flts[3])
        self.dec3 = Decoder_UpSampling(flts[3], flts[2])
        self.dec2 = Decoder_UpSampling(flts[2], flts[1])
        self.dec1 = Decoder_UpSampling(flts[1], flts[0])
        
        
        # última capa conv que nos da la máscara de segmentacinón
        self.out1 = torch.nn.Conv2d(flts[0], num_classes_seg, 1, padding=0)
        
        # MaxPooling
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # Encoder
        x1E = self.enc1(x)
        x2E = self.pool(x1E)
        x2E = self.enc2(x2E)
        x3E = self.pool(x2E)
        x3E = self.enc3(x3E)
        x4E = self.pool(x3E)
        x4E = self.enc4(x4E)

        # Bottleneck
        x5E = self.pool(x4E)
        x5E = self.bottleneck(x5E)
        
        # DecoderSeg
        x4 = self.dec4(x5E,x4E)
        x3 = self.dec3(x4,x3E)
        x2 = self.dec2(x3,x2E)
        x1 = self.dec1(x2,x1E)
        
        
        # Salidas de la segmentación y esqueleto
        xa = self.out1(x1)
        return xa

# Ejemplo de uso
if __name__ == "__main__":
    inputs = torch.randn((4, 1, 512, 512))  # Batch de imágenes de ejemplo
    model = MultiTaskUNET()  # InstanC_inar el modelo
    y = model(inputs)  # Forward pass
    print(y.shape)  # Salida de la máscara de segmentacinón
    
    summary(model, input_size = (1, 512, 512))
   
        
        
        
        
        
        
        
