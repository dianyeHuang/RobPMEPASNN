#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Author: Yuan Bi
Date: Wed Jan 25 16:51:09 2023
LastEditors: Yuan Bi
LastEditTime: 2023-07-04 17:36:51
Description: 
    This script implements the proposed PAS-NN architecture.
    
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)


class Unflatten(nn.Module):
	def __init__(self, channel, height, width):
		super(Unflatten, self).__init__()
		self.channel = channel
		self.height = height
		self.width = width

	def forward(self, input):
		return input.view(input.size(0), self.channel, self.height, self.width)
     
class DoubleConv2d(nn.Module):
    def __init__(self,in_channels,features):
        super(DoubleConv2d, self).__init__()
        self.block = nn.Sequential(
                nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
                # nn.GroupNorm(num_groups=int(features/2), num_channels=features),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
                nn.Conv2d(features, features, kernel_size=3, padding=1),
                # nn.GroupNorm(num_groups=int(features/2), num_channels=features),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
                )

    def forward(self,x):
        return self.block(x)
    
class UpConv2d_s2(nn.Module):
    def __init__(self,in_channels,features):
        super(UpConv2d_s2, self).__init__()
        self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, features, kernel_size=2, stride=2),
                # nn.GroupNorm(num_groups=int(features/2), num_channels=features),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
                )

    def forward(self,x):
        return self.block(x)
    
class Res_block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class Res_layer(nn.Module):
    def __init__(self, inplanes, planes, blocks, stride=1):
        super(Res_layer, self).__init__()
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(Res_block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(Res_block(planes, planes))
        
        self.res_layer = nn.Sequential(*layers)
    
    def forward(self,x):
        return self.res_layer(x)
    
    
class Att_Gate(nn.Module):
    
    def __init__(self, in_channels_x, in_channels_g):
        super(Att_Gate, self).__init__()
        # self.in_channels = in_channels
        self.upsample_mode = 'bilinear'
        self.inter_channels = in_channels_g//2
        self.Wg = nn.Conv2d(in_channels=in_channels_g, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.Wx = nn.Conv2d(in_channels=in_channels_x, out_channels=self.inter_channels,
                             kernel_size=2, stride=2, padding=0)
        self.psi = nn.Conv2d(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=in_channels_x, out_channels=in_channels_x, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels_x),
        )
    
    def forward(self, x, g):
        x_size = x.size()
        g_size = g.size()

        theta_x = F.interpolate(self.Wx(x), size=g_size[2:], mode=self.upsample_mode)

        phi_g = self.Wg(g)
        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = torch.sigmoid(self.psi(f))

        sigm_psi_f = F.interpolate(sigm_psi_f, size=x_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f
        
    
class DNL_block(nn.Module):

    def __init__(self, in_dim):
        super(DNL_block, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.mask_conv = nn.Conv2d(in_dim, 1, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B * C * H * W)
            returns :
                out : self attention value + input feature
                attention: B * N * N (N is H * W)
        """
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # B * N * C
        proj_query -= proj_query.mean(1).unsqueeze(1)
        proj_key = self.key_conv(x).view(B, -1, H * W)  # B * C * N
        proj_key -= proj_key.mean(2).unsqueeze(2)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)  # B * N * N
        attention = attention.permute(0, 2, 1)
        proj_value = self.value_conv(x).view(B, -1, H * W)  # B * C * N
        proj_value = self.relu(proj_value)
        
        proj_mask = self.mask_conv(x).view(B, -1, H * W)  # B * 1 * N
        mask = self.softmax(proj_mask)
        mask = mask.permute(0, 2, 1)
        
        attention = attention+mask
        
        tissue = torch.bmm(proj_value, attention)
        tissue = tissue.view(B, C, H, W)

        out = x
        return out, tissue
    
    
class MagNet(nn.Module):
    def __init__(self,in_channels=1,init_features=32):
        super(MagNet, self).__init__()
        
        features = init_features
        
        self.res_layer_M_1 = Res_layer(in_channels,features,blocks=2,stride=1) # 1->64
        self.pool_M_1 = nn.AvgPool2d(2, 2)
        self.res_layer_M_2 = Res_layer(features,2*features,blocks=2,stride=1) # 64->128
        self.pool_M_2 = nn.AvgPool2d(2, 2)
        self.res_layer_M_3 = Res_layer(2*features,4*features,blocks=2,stride=1) # 128->256
        self.pool_M_3 = nn.AvgPool2d(2, 2)
        self.res_layer_M_4 = Res_layer(4*features,8*features,blocks=2,stride=1) # 256->512
        self.pool_M_4 = nn.AvgPool2d(2, 2)
        self.bottleneck_M = Res_layer(8*features,8*features,blocks=2,stride=1) # 512->1024
        
        self.res_layer_I_1 = Res_layer(in_channels,features,blocks=2,stride=1) # 1->64
        self.pool_I_1 = nn.AvgPool2d(2, 2)
        self.res_layer_I_2 = Res_layer(features,2*features,blocks=2,stride=1) # 64->128
        self.pool_I_2 = nn.AvgPool2d(2, 2)
        self.res_layer_I_3 = Res_layer(2*features,4*features,blocks=2,stride=1) # 128->256
        self.pool_I_3 = nn.AvgPool2d(2, 2)
        self.res_layer_I_4 = Res_layer(4*features,8*features,blocks=2,stride=1) # 256->512
        self.pool_I_4 = nn.AvgPool2d(2, 2)
        self.bottleneck_I = Res_layer(8*features,8*features,blocks=2,stride=1) # 512->1024
        
        # self.encoder_I = nn.Sequential(
        #     Res_layer(in_channels,features,blocks=2,stride=2), # 1->64
        #     Res_layer(features,2*features,blocks=2,stride=2), # 64->128
        #     Res_layer(2*features,4*features,blocks=2,stride=2), # 128->256
        #     Res_layer(4*features,8*features,blocks=2,stride=2), # 256->512
        #     Res_layer(8*features,8*features,blocks=2,stride=1) # 512->1024
        # )
        
        self.att_gate_4 = Att_Gate(8*features,8*features)
        self.upconv_4 = UpConv2d_s2(16*features,8*features)
        self.decoder_4 = DoubleConv2d(16*features,8*features)
        
        self.att_gate_3 = Att_Gate(4*features,4*features)
        self.upconv_3 = UpConv2d_s2(8*features,4*features)
        self.decoder_3 = DoubleConv2d(8*features,4*features)
        
        self.att_gate_2 = Att_Gate(2*features,2*features)
        self.upconv_2 = UpConv2d_s2(4*features,2*features)
        self.decoder_2 = DoubleConv2d(4*features,2*features)
        
        self.att_gate_1 = Att_Gate(features,features)
        self.upconv_1 = UpConv2d_s2(2*features,features)
        self.decoder_1 = DoubleConv2d(2*features,features)
        
        self.decoder_0 = nn.Sequential(
            nn.Conv2d(features, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        
    def forward(self,x,m):
        I_1 = self.res_layer_I_1(x)
        I_2 = self.res_layer_I_2(self.pool_I_1(I_1))
        I_3 = self.res_layer_I_3(self.pool_I_2(I_2))
        I_4 = self.res_layer_I_4(self.pool_I_3(I_3))
        bottleneck_I = self.bottleneck_I(self.pool_I_4(I_4))
        
        M_1 = self.res_layer_M_1(m)
        M_2 = self.res_layer_M_2(self.pool_M_1(M_1))
        M_3 = self.res_layer_M_3(self.pool_M_2(M_2))
        M_4 = self.res_layer_M_4(self.pool_M_3(M_3))
        bottleneck_M = self.bottleneck_M(self.pool_M_4(M_4))
                
        dec_4,_ = self.att_gate_4(I_4,M_4)
        # dec_4 = M_4*I_4
        temp = self.upconv_4(torch.cat([bottleneck_I,bottleneck_M],dim=1))
        dec_4 = torch.cat([dec_4,temp],dim=1)
        dec_4 = self.decoder_4(dec_4)
        
        dec_3,_ = self.att_gate_3(I_3,M_3)
        # dec_3 = M_3*I_3
        dec_3 = torch.cat([dec_3,self.upconv_3(dec_4)],dim=1)
        dec_3 = self.decoder_3(dec_3)
        
        dec_2,_ = self.att_gate_2(I_2,M_2)
        # dec_2 = M_2*I_2
        dec_2 = torch.cat([dec_2,self.upconv_2(dec_3)],dim=1)
        dec_2 = self.decoder_2(dec_2)
        
        dec_1,_ = self.att_gate_1(I_1,M_1)
        # dec_1 = M_1*I_1
        dec_1 = torch.cat([dec_1,self.upconv_1(dec_2)],dim=1)
        dec_1 = self.decoder_1(dec_1)
        
        output = self.decoder_0(dec_1)
        
        return output
    
    
class MagNet_v2(nn.Module):
    def __init__(self,in_channels=1,init_features=32):
        super(MagNet_v2, self).__init__()
        
        features = init_features
        
        self.res_layer_M_1 = Res_layer(in_channels,features,blocks=2,stride=1) # 1->64
        self.pool_M_1 = nn.AvgPool2d(2, 2)
        self.res_layer_M_2 = Res_layer(features,2*features,blocks=2,stride=1) # 64->128
        self.pool_M_2 = nn.AvgPool2d(2, 2)
        self.res_layer_M_3 = Res_layer(2*features,4*features,blocks=2,stride=1) # 128->256
        self.pool_M_3 = nn.AvgPool2d(2, 2)
        self.res_layer_M_4 = Res_layer(4*features,8*features,blocks=2,stride=1) # 256->512
        self.pool_M_4 = nn.AvgPool2d(2, 2)
        self.bottleneck_M = Res_layer(8*features,8*features,blocks=2,stride=1) # 512->1024
        
        self.res_layer_I_1 = Res_layer(in_channels,features,blocks=2,stride=1) # 1->64
        self.pool_I_1 = nn.AvgPool2d(2, 2)
        self.res_layer_I_2 = Res_layer(features,2*features,blocks=2,stride=1) # 64->128
        self.pool_I_2 = nn.AvgPool2d(2, 2)
        self.res_layer_I_3 = Res_layer(2*features,4*features,blocks=2,stride=1) # 128->256
        self.pool_I_3 = nn.AvgPool2d(2, 2)
        self.res_layer_I_4 = Res_layer(4*features,8*features,blocks=2,stride=1) # 256->512
        self.pool_I_4 = nn.AvgPool2d(2, 2)
        self.bottleneck_I = Res_layer(8*features,8*features,blocks=2,stride=1) # 512->1024
        
        # self.encoder_I = nn.Sequential(
        #     Res_layer(in_channels,features,blocks=2,stride=2), # 1->64
        #     Res_layer(features,2*features,blocks=2,stride=2), # 64->128
        #     Res_layer(2*features,4*features,blocks=2,stride=2), # 128->256
        #     Res_layer(4*features,8*features,blocks=2,stride=2), # 256->512
        #     Res_layer(8*features,8*features,blocks=2,stride=1) # 512->1024
        # )
        
        self.att_gate_4 = Att_Gate(16*features,8*features)
        self.upconv_4 = UpConv2d_s2(16*features,8*features)
        self.decoder_4 = DoubleConv2d(16*features,8*features)
        
        self.att_gate_3 = Att_Gate(8*features,4*features)
        self.upconv_3 = UpConv2d_s2(8*features,4*features)
        self.decoder_3 = DoubleConv2d(8*features,4*features)
        
        self.att_gate_2 = Att_Gate(4*features,2*features)
        self.upconv_2 = UpConv2d_s2(4*features,2*features)
        self.decoder_2 = DoubleConv2d(4*features,2*features)
        
        self.att_gate_1 = Att_Gate(2*features,features)
        self.upconv_1 = UpConv2d_s2(2*features,features)
        self.decoder_1 = DoubleConv2d(2*features,features)
        
        self.decoder_0 = nn.Sequential(
            nn.Conv2d(features, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        
    def forward(self,x,m):
        I_1 = self.res_layer_I_1(x)
        I_2 = self.res_layer_I_2(self.pool_I_1(I_1))
        I_3 = self.res_layer_I_3(self.pool_I_2(I_2))
        I_4 = self.res_layer_I_4(self.pool_I_3(I_3))
        bottleneck_I = self.bottleneck_I(self.pool_I_4(I_4))
        
        M_1 = self.res_layer_M_1(m)
        M_2 = self.res_layer_M_2(self.pool_M_1(M_1))
        M_3 = self.res_layer_M_3(self.pool_M_2(M_2))
        M_4 = self.res_layer_M_4(self.pool_M_3(M_3))
        bottleneck_M = self.bottleneck_M(self.pool_M_4(M_4))
                
        temp = self.upconv_4(torch.cat([bottleneck_I,bottleneck_M],dim=1))
        dec_4 = torch.cat([I_4,temp],dim=1)
        dec_4,_ = self.att_gate_4(dec_4,M_4)
        dec_4 = self.decoder_4(dec_4)
        
        dec_3 = torch.cat([I_3,self.upconv_3(dec_4)],dim=1)
        dec_3,_ = self.att_gate_3(dec_3,M_3)
        dec_3 = self.decoder_3(dec_3)
        
        dec_2 = torch.cat([I_2,self.upconv_2(dec_3)],dim=1)
        dec_2,_ = self.att_gate_2(dec_2,M_2)
        dec_2 = self.decoder_2(dec_2)
        
        dec_1 = torch.cat([I_1,self.upconv_1(dec_2)],dim=1)
        dec_1,_ = self.att_gate_1(dec_1,M_1)
        dec_1 = self.decoder_1(dec_1)
        
        output = self.decoder_0(dec_1)
        
        return output
        
    
class AttUNet(nn.Module):
    def __init__(self,in_channels=1,init_features=32):
        super(AttUNet, self).__init__()
        
        features = init_features
        
        self.encoder_1 = DoubleConv2d(in_channels,features) # 1->64
        self.pool_1 = nn.AvgPool2d(2, 2)
        self.encoder_2 = DoubleConv2d(features,2*features) # 64->128
        self.pool_2 = nn.AvgPool2d(2, 2)
        self.encoder_3 = DoubleConv2d(2*features,4*features) # 128->256
        self.pool_3 = nn.AvgPool2d(2, 2)
        self.encoder_4 = DoubleConv2d(4*features,8*features) # 256->512
        self.pool_4 = nn.AvgPool2d(2, 2)
        self.bottleneck = DoubleConv2d(8*features,16*features) # 512->1024
        
        self.att_gate_4 = Att_Gate(8*features,8*features)
        self.upconv_4 = UpConv2d_s2(16*features,8*features)
        self.decoder_4 = DoubleConv2d(16*features,8*features)
        
        self.att_gate_3 = Att_Gate(4*features,4*features)
        self.upconv_3 = UpConv2d_s2(8*features,4*features)
        self.decoder_3 = DoubleConv2d(8*features,4*features)
        
        self.att_gate_2 = Att_Gate(2*features,2*features)
        self.upconv_2 = UpConv2d_s2(4*features,2*features)
        self.decoder_2 = DoubleConv2d(4*features,2*features)
        
        self.att_gate_1 = Att_Gate(features,features)
        self.upconv_1 = UpConv2d_s2(2*features,features)
        self.decoder_1 = DoubleConv2d(2*features,features)
        
        self.decoder_0 = nn.Sequential(
            nn.Conv2d(features, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        
    def forward(self,x):
        I_1 = self.encoder_1(x)
        I_2 = self.encoder_2(self.pool_1(I_1))
        I_3 = self.encoder_3(self.pool_2(I_2))
        I_4 = self.encoder_4(self.pool_3(I_3))
        bottleneck_I = self.bottleneck(self.pool_4(I_4))

        dec_4,_ = self.att_gate_4(I_4,self.upconv_4(bottleneck_I))
        dec_4 = torch.cat([dec_4,I_4],dim=1)
        dec_4 = self.decoder_4(dec_4)
        
        dec_3,_ = self.att_gate_3(I_3,self.upconv_3(dec_4))
        dec_3 = torch.cat([dec_3,I_3],dim=1)
        dec_3 = self.decoder_3(dec_3)
        
        dec_2,_ = self.att_gate_2(I_2,self.upconv_2(dec_3))
        dec_2 = torch.cat([dec_2,I_2],dim=1)
        dec_2 = self.decoder_2(dec_2)
        
        dec_1,_ = self.att_gate_1(I_1,self.upconv_1(dec_2))
        dec_1 = torch.cat([dec_1,I_1],dim=1)
        dec_1 = self.decoder_1(dec_1)
        
        output = self.decoder_0(dec_1)
        
        return output
    
    