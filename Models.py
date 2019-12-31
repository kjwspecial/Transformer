#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import import_ipynb
from SubLayers import Multi_Head_Attention, Scaled_Dot_Product_Attention, Position_Wise_FFNN


# In[23]:


''' word / pad mask '''
def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

''' 패딩 & 디코더 마스크 : and 연산 '''
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(trg, pad):
    trg_mask = get_pad_mask(trg,pad)
    trg_mask = trg_mask & Variable(subsequent_mask(trg.size(-1)).type_as(trg_mask.data))
    return trg_mask


# In[24]:


class Positional_Encoding(nn.Module):
    def __init__(self,d_model, seq_len, dropout = 0.1):
        super(Positional_Encoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        '''
            pos : row
            div_term : 10000^(2i/d_model)
        '''
        PE=torch.zeros(seq_len, d_model)
        pos = torch.arange(0,seq_len).unsqueeze(1)# [seq_len , 1]
        div_term = torch.pow(10000, torch.arange(0, 1, 2/d_model)) 
        PE[:,0::2] = torch.sin(pos/div_term)
        PE[:,1::2] = torch.cos(pos/div_term)
        PE = PE.unsqueeze(0)# [1, seq_len, embed_dim]
        self.register_buffer('PE',PE)
    def forward(self, x):
        # x.size(1) : 입력 pos length [batch, seq_len, embedding]
        # 굳이 variable로 할 필요가 있을까?
        x = x + Variable(self.PE[:,:x.size(1)],requires_grad=False)# x.size(1)??
        return self.dropout(x)#.detach()


# In[25]:


class Encoder(nn.Module):
    def __init__(self, d_model, head_num, fc_dim):
        super(Encoder,self).__init__()
        self.multi_head_attention = Multi_Head_Attention(d_model,head_num)
        self.position_wise_ffnn = Position_Wise_FFNN(d_model,fc_dim)
        
    def forward(self,x,mask=None):
        ''' Q == x : [batch, seq_len, d_model] '''
        x = self.multi_head_attention(x,x,x,mask=mask)
        output = self.position_wise_ffnn(x)
        return output


# In[26]:


class Decoder(nn.Module):
    def __init__(self,d_model, head_num, fc_dim ):
        super(Decoder,self).__init__()
        self.masked_multi_head_attention = Multi_Head_Attention(d_model, head_num)
        self.multi_head_attention = Multi_Head_Attention(d_model, head_num)
        self.position_wise_ffnn = Position_Wise_FFNN(d_model,fc_dim)
        
    def forward(self,x, encoder_output, self_attention_mask=None, decoder_mask=None):
        x = self.masked_multi_head_attention(x,x,x,decoder_mask)
        x = self.multi_head_attention(x,encoder_output,encoder_output,self_attention_mask)   #여기서 왜 에러가 나는것이야
        output = self.position_wise_ffnn(x)
        
        return output


# In[27]:


class Stacked_Encoder(nn.Module):
    def __init__(self, n_layers, d_model, head_num, fc_dim , num_src_vocab, pad_idx, seq_len ,dropout=0.1):
        super(Stacked_Encoder,self).__init__()# init class 호출한것임.
        
        ''' 
            nn.Embedding(총 단어 개수, embed_dim, pad_idx)
            call 할 때 seq넣어주면 되는듯.
        '''
        self.src_word_embedding = nn.Embedding(num_src_vocab, d_model, padding_idx=pad_idx)
        self.positional_encoding = Positional_Encoding(d_model, seq_len)
        self.layer_stack = nn.ModuleList([
            Encoder(d_model, head_num, fc_dim) for _ in range(n_layers)])
        
        self.dropout= nn.Dropout()
        
#        self.layer_norm = nn.LayerNorm(d_model)
    def forward(self,SRC_seq, mask=None):
        '''
        ToDo. 
            attention map를 위해 attention 반환 추가 
        '''
        src_word_embed = self.dropout(self.positional_encoding(self.src_word_embedding(SRC_seq)))
        for each_layer in self.layer_stack:
            output = each_layer(src_word_embed, mask)
            
        #output = self.layer_norm(output)
        return output


# In[28]:


class Stacked_Decoder(nn.Module):
    def __init__(self,n_layers,d_model, head_num, fc_dim, num_trg_vocab, pad_idx, seq_len ,dropout=0.1 ):
        super(Stacked_Decoder,self).__init__()
        
        self.trg_word_embedding = nn.Embedding(num_trg_vocab, d_model, padding_idx= pad_idx)
        
        self.positional_encoding = Positional_Encoding(d_model, seq_len)
        self.layer_stack = nn.ModuleList([Decoder(d_model, head_num, fc_dim) for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    def forward(self, TRG_seq, encoder_output, self_attention_mask=None, decoder_mask=None):
        trg_word_embed = self.dropout(self.positional_encoding(self.trg_word_embedding(TRG_seq)))
        for each_layer in self.layer_stack:
            output = each_layer(trg_word_embed,encoder_output ,self_attention_mask, decoder_mask)

        #output = self.layer_norm(output)
        return output


# In[29]:


class Transformer(nn.Module):
    '''
    note.
        Shared Embeddings:When using BPE(토큰 단위로 나누는 방법) with shared vocabulary we can share the same weight vectors between the source / target / generator.
        See the (https://arxiv.org/abs/1608.05859) for details.
        
        BPE : https://arxiv.org/pdf/1508.07909.pdf
        # or 구글 sentencepiece
        
    '''
    def __init__(self, num_src_vocab, num_trg_vocab,src_pad_idx,trg_pad_idx, 
                 n_layers=6, d_model= 512, head_num =8 , fc_dim=2048,seq_len=200,dropout=0.1,
                 weight_sharing=False):
        super(Transformer,self).__init__()
        self.encoder = Stacked_Encoder(n_layers,d_model,head_num,fc_dim,
                                       num_src_vocab,src_pad_idx,seq_len,dropout)
        
        self.decoder = Stacked_Decoder(n_layers,d_model,head_num,fc_dim,
                                       num_trg_vocab,trg_pad_idx,seq_len,dropout)
        #마지막 linear 연산
        self.generator = nn.Linear(d_model,num_trg_vocab,bias=False)
        self.src_pad_idx=src_pad_idx
        self.trg_pad_idx=trg_pad_idx
        '''kaiming_he init'''
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.kaiming_uniform_(param)
        
        ''' BPE 구현 후 적용 할 예정.'''
        self.x_logit_scale = 1 
        if weight_sharing:
            # Share the weight between target word embedding & [last dense layer , source word embedding]
            self.generator.weight = self.Stacked_Decoder.trg_word_embedding.weight
            self.Stacked_Encoder.src_word_embedding.weight = self.Stacked_Decoder.trg_word_embedding.weight
            self.x_logit_scale = (d_model ** -0.5)

    def forward(self, src_seq, trg_seq):
        src_mask = get_pad_mask(src_seq,self.src_pad_idx)
        trg_mask = make_std_mask(trg_seq,self.trg_pad_idx)

        encoder_output = self.encoder(src_seq, src_mask)
        decoder_output = self.decoder(trg_seq, encoder_output,src_mask ,trg_mask)# 마스크 순서바꿈

        seq_logit = self.generator(decoder_output) * self.x_logit_scale
        '''
         [batch * seq_len, num_trg_vocab]
         => 각 단어에 대한 확률.
         '''
        return seq_logit.view(-1, seq_logit.size(2))

