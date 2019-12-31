import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Multi_Head_Attention(nn.Module):
    '''
    Args.
        Q : [batch, seq_len, d_model]
        
    calc.
        att : [batch, head_num, seq_len, self.dk]
    
        Q.views          => [batch, seq_len, head_num, self.dk].transpose(-1,-2)
        transpose(-1,-2) => [batch, head_num, seq_len, self.dk]
        
        
    note.
        -  View로 바꾸면 안됨. 반드시 transpose를 이용
        => seq_len을 쪼개서 dim을 바꾸게 될 수 있음
        
        contiguous() :
            보통 view함수를 써서 텐서모양을 고칠때 contiguous형식이 요구되는데 
            view함수는 reshape나 resize와는 다르게 어떤 경우에도 메모리 복사없이 이루어진다.
            따라서 contiguous형식이 아닐때는 텐서모양을 고칠수 없게되고 런타임에러가 발생한다.

            => 요약하자면 몇몇함수가 메모리 효율적인 연산을 위해 contiguous형식을 요구하니
               그 함수를 사용할때만 contiguous형식으로 맞춰서 사용하자.

            https://stackoverflow.com/questions/26998223/what-is-the-difference-between-contiguous-and-non-contiguous-arrays?fbclid=IwAR0HYP4D75GiswZ7WGLwW230CTWh7vcXPXPrGJSATJQtselBYsdY_gcvg-M
    '''
    def __init__(self,d_model, head_num=8, dropout=0.1):
        super(Multi_Head_Attention,self).__init__()
        self.dk = d_model // head_num
        
        self.Wq = nn.Linear(d_model,self.dk * head_num, bias = False)
        self.Wk = nn.Linear(d_model,self.dk * head_num, bias = False)
        self.Wv = nn.Linear(d_model,self.dk * head_num, bias = False)
        
        self.Wo = nn.Linear(d_model,self.dk * head_num, bias = False)
        self.scaled_dot_product_attention = Scaled_Dot_Product_Attention()
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head_num=head_num
    def forward(self,Q,K,V, mask=None):
        residual = Q
        batch_size = Q.size(0)
        
        q_w = self.Wq(Q).view(batch_size, -1, self.head_num,self.dk).transpose(1,2)
        k_w = self.Wk(K).view(batch_size, -1, self.head_num,self.dk).transpose(1,2)
        v_w = self.Wv(V).view(batch_size, -1, self.head_num,self.dk).transpose(1,2)
        ''' masks => [1,1,1,seq_len] : seq_k에 적용 되는 마스크.'''
        if mask is not None:
            mask = mask.unsqueeze(1)
            #mask = mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]
        
               
        att = self.scaled_dot_product_attention(q_w,k_w,v_w,self.dk,mask=mask) 
        att = att.transpose(1,2).contiguous().view(batch_size,-1,self.head_num * self.dk)
        
        Multi_head_attention_matrix = self.dropout(self.Wo(att))
        
        output = self.layer_norm(Multi_head_attention_matrix + residual)
        return output


class Scaled_Dot_Product_Attention(nn.Module):
    '''
    Args
        query : [batch, head_num, seq_len, self.dk]
        Attention_score : [batch, head_num, q_seq_len, k_seq_len]
        output : [batch, head_num, seq_len, self.dk] 
        
    calc
        matmul : ([batch, head_num, q_seq_len, self.dk], [batch, head_num, self.dk, k_seq_len])
               => [batch, head_num, q_seq_len, k_seq_len]
              
        output : [batch, head_num, seq_len, self.dk]
    '''       
    def __init__(self,dropout=0.1):
        super(Scaled_Dot_Product_Attention,self).__init__()
        self.dropout= nn.Dropout(dropout)
            
    def forward(self,query,key,value,d_k, mask=None):
        Attention_score = torch.matmul(query, key.transpose(-1,-2))
        Attention_score = Attention_score/ np.sqrt(d_k)
        
        '''
        mask: 입력 seq_len중 word이 아닌 경우를 구분하는 역할(word입력이 끝난 후 padding 처리와 동일)      
            tensor([[0.4402, 0.7203, 0.0000, 0.0000],
                    [0.2249, 0.4896, 0.0000, 0.0000],
                    [0.3616, 0.5888, 0.0000, 0.0000], 
                    [0.4716, 0.6443, 0.0000, 0.0000],          
            =>
            tensor([[0.4402, 0.7203, -inf,   -inf],
                    [0.2249, 0.4896, -inf,   -inf],
                    [0.3616, 0.5888, -inf,   -inf], 
                    [0.4716, 0.6443, -inf,   -inf],
        '''
        if mask is not None:
            Attention_score = Attention_score.masked_fill(mask==0, float("-Inf"))#-1e9)

        Att_softmax = self.dropout(F.softmax(Attention_score, dim=-1))

        output = torch.matmul(Att_softmax,value)
        
        return output


class Position_Wise_FFNN(nn.Module):
    def __init__(self, d_model, hidden_out, dropout = 0.1):
        super(Position_Wise_FFNN,self).__init__()
        self.FFNN = nn.Sequential(
            nn.Linear(d_model,hidden_out),
            nn.LeakyReLU(),
            nn.Linear(hidden_out,d_model),
            nn.Dropout(dropout)
        )

        self.layer_norm = nn.LayerNorm(d_model)
            
    def forward(self, x):
        residual = x
        x = self.FFNN(x)
        output = self.layer_norm(x + residual)
        
        return output

