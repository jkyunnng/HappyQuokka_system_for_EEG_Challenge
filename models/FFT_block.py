import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict
import pdb
from models.SubLayers import MultiHeadAttention, PositionwiseFeedForward

class PreLNFFTBlock(torch.nn.Module):

    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,
                 dropout,
                 **kwargs):

        super(FFTBlock, self).__init__()

        d_k = d_model // n_head
        d_v = d_model // n_head

        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, fft_conv1d_kernel,fft_conv1d_padding, dropout=dropout)

    def forward(self, fft_input):

        # dec_input size: [B,T,C]
        fft_output, _= self.slf_attn(
            fft_input, fft_input, fft_input)

        fft_output = self.pos_ffn(fft_output)

        return fft_output


class Decoder(nn.Module):

    def __init__(self,
                 in_channel,
                 d_model,
                 d_inner,
                 n_head,
                 n_layers,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,             
                 dropout,
                 g_con,
                 within_sub_num = 71,
                 **kwargs):

        super(Decoder, self).__init__()
        d_k = d_model // n_head
        d_v = d_model // n_head
        self.g_con = g_con
        self.within_sub_num = within_sub_num
        self.slf_attn = MultiHeadAttention
        self.fc = nn.Linear(d_model, 1)   
        self.conv = nn.Conv1d(in_channel, d_model, kernel_size = 7, padding=3)
        self.layer_stack = nn.ModuleList([PreLNFFTBlock(
            d_model, d_inner, n_head, fft_conv1d_kernel,fft_conv1d_padding, dropout) for _ in range(n_layers)])
        self.sub_proj = nn.Linear(self.within_sub_num, d_model)

    def forward(self, dec_input, sub_id):

        dec_output = self.conv(dec_input.transpose(1,2))
        dec_output = dec_output.transpose(1,2)

        # Global conditioner.
        if self.g_con == True:
            sub_emb    = F.one_hot(sub_id, self.within_sub_num)
            sub_emb    = self.sub_proj(sub_emb.float())
            output = dec_output + sub_emb.unsqueeze(1)

        else:
            output = dec_output
        
        for dec_layer in self.layer_stack:
            output = dec_layer(
                output)

        output = self.fc(output)

        return output
