#!/usr/bin/env python
# coding: utf-8

# In[3]:


import argparse
import math, copy, time
import os
import gc
from tqdm import tqdm
import pickle

import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data, datasets

from Models import Transformer
from apex import amp
#https://on-demand.gputechconf.com/ai-conference-2019/T1-1_Jack%20Han_Getting%20More%20DL%20Training%20with%20Tensor%20Cores%20and%20AMP_%ED%95%9C%EC%9E%AC%EA%B7%BC_%EB%B0%9C%ED%91%9C%EC%9A%A9.pdf


# In[ ]:


def re_batch(src, trg):
    src = src.transpose(0,1)
    trg = trg.transpose(0,1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return src, trg, gold


# In[ ]:


def calc_loss(pred, gold, trg_pad_idx):
    gold = gold.contiguous().view(-1)
    loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')# sum이 문제인가?
    non_pad_mask = gold.ne(trg_pad_idx)
    pred = pred.argmax(dim=1)
    '''
        1. pred.eq(gold) : equal check.
        2. masked_select : True인 부분만 남겨, padding 부분 제외시킴.
    '''   
    n_word_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return loss, n_word, n_word_correct


# In[ ]:


def train_epoch(model, data_loader,optimizer,device,args):
    model.train()
    total_loss, n_word_total, n_word_correct=0,0,0
    desc = '  - (Training)   '
    for batch in tqdm(data_loader,mininterval=2, desc=desc, leave=False):    
        src_seq, trg_seq, gold = map(lambda x : x.to(device), re_batch(batch.src, batch.trg))
        
        optimizer.zero_grad()    
        
        pred = model(src_seq, trg_seq)
        loss, n_word, n_correct = calc_loss(pred, gold, args.trg_pad_idx)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
            
        optimizer.step()
        
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()
        
    accuracy  = n_word_correct / n_word_total
    loss_per_word = total_loss/ n_word_total
    
    return accuracy , loss_per_word


# In[ ]:


def eval_epoch(model, data_loader,device,args):
    model.eval()
    total_loss, n_word_total, n_word_correct = 0,0,0
    desc = ' (Valadation) '
    with torch.no_grad():
        for batch in tqdm(data_loader,  desc= desc, leave = False):
            
            src_seq , trg_seq, gold = map(lambda x : x.to(device), re_batch(batch.src,batch.trg))
            
            pred = model(src_seq,trg_seq)
            
            loss , n_word,n_correct = calc_loss(pred,gold, args.trg_pad_idx)
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()   
            
    accuracy  = n_word_correct / n_word_total
    loss_per_word = total_loss/ n_word_total
    
    return accuracy , loss_per_word


# In[ ]:


def train(model, train_iter, val_iter, optimizer, device, args):
    
    val_loss =[]
    for e in range(args.epoch):
        start = time.time()
        checkpoint = {
                  'epoch':e,
                  'args': args, 
                  'model': model.state_dict()
                 }
        acc, loss_per_word = train_epoch(model, train_iter,optimizer,device,args)
        print('epoch : {e:3d}, accuracy : {acc:3.3f}%, elapse : {elapse:3.3f} min'.format(e=e+1,acc=acc*100, elapse=(time.time()-start)/60))
        
        val_acc, val_loss_per_word = eval_epoch(model, val_iter ,device,args)
        print('accuracy : {acc:3.3f}%, elapse : {elapse:3.3f} min , loss_per_word : {per_word:3.3f}%'.format(acc=val_acc*100, elapse=(time.time()-start)/60, per_word=val_loss_per_word))
        
        val_loss +=[loss_per_word]
        if loss_per_word <= min(val_loss):
            torch.save(checkpoint, './transformer_model.ckpt')
            print("- checkpoint update  ")


# In[4]:


def data_loader(args, device):
    if not os.path.exists(".data/iwslt/de-en"):
        print("Data download.")
        os.system("python -m spacy download en_core_web_sm")
        os.system("python -m spacy download de_core_news_sm")
    if not os.path.exists("./data"):
        print("create data folder")
        os.system("mkdir data")
    print("Data loading.")
    en_nlp = spacy.load("en_core_web_sm")
    de_nlp = spacy.load("de_core_news_sm")

    def tokenize_de(text):
        return [tok.text for tok in de_nlp.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in en_nlp.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    PAD_WORD = "<pad>"

    SRC = data.Field(tokenize=tokenize_de, lower=True, pad_token=PAD_WORD)
    TGT = data.Field(tokenize=tokenize_en, lower=True, init_token = BOS_WORD, eos_token = EOS_WORD, pad_token=PAD_WORD)

    MAX_LEN = args.seq_len
    BATCH_SIZE = args.batch
   
    '''
        exts – A tuple containing the extension to path for each language.
        fields – A tuple containing the fields that will be used for data in each language.
    '''

    train, test, val = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(SRC, TGT), 
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
            len(vars(x)['trg']) <= MAX_LEN)

    MIN_FREQ = 2

    '''단어장 생성'''
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

    args.num_src_vocab = len(SRC.vocab.stoi)
    args.num_trg_vocab = len(TGT.vocab.stoi)
    args.src_pad_idx = SRC.vocab.stoi['<pad>']
    args.trg_pad_idx = TGT.vocab.stoi['<pad>']
    
    with open('./data/SRC_vocab_pkl', 'wb') as f:
        pickle.dump(SRC.vocab, f)
    with open('./data/TGT_vocab_pkl', 'wb') as f:
        pickle.dump(TGT.vocab, f)

    '''
        EOFError: Compressed file ended before the end-of-stream marker was reached
        =>  rm -rf .data/
        재다운로드 하면된다.
    '''
    
    train_iter = data.BucketIterator(train, batch_size=BATCH_SIZE,device = device, sort_key=lambda x: len(x.src)) #,shuffle=True,repeat=False)
    val_iter = data.BucketIterator(val, batch_size=1,device = device, sort_key=lambda x: len(x.src)) #,repeat=False)
    return train_iter, val_iter


# In[13]:


def main(): 
    parser = argparse.ArgumentParser()
    
    #python train.py -epoch 10 -batch 128 ... 
    
    parser.add_argument('-epoch',type=int, default = 10)
    parser.add_argument('-batch',type=int, default = 32)
    parser.add_argument('-d_model',type=int, default= 512)
    parser.add_argument('-n_layers',type=int, default = 6)
    parser.add_argument('-head_num',type = int, default= 8)
    parser.add_argument('-fc_dim',type=int, default=2048)
    parser.add_argument('-seq_len',type=int, default=200)
    parser.add_argument('-dropout',type=float, default=0.1)
    parser.add_argument('-weight_sharing', action='store_true')
    
    args = parser.parse_args()
    
    args.weight_sharing = not args.weight_sharing
    device = torch.device('cuda')
    train_iter, val_iter = data_loader(args,device)

    transformer =  Transformer(args.num_src_vocab,
                              args.num_trg_vocab,
                              args.src_pad_idx,
                              args.trg_pad_idx,
                              n_layers = args.n_layers,
                              d_model = args.d_model,
                              head_num = args.head_num,
                              fc_dim = args.fc_dim,
                              seq_len = args.seq_len,
                              dropout=args.dropout)
    

    transformer = transformer.to(device)
    optimizer = optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09)
    transformer, optimizer = amp.initialize(transformer, optimizer, opt_level="O1")
    train(transformer, train_iter, val_iter, optimizer, device, args)
    
    torch.save(transformer.state_dict(), './transformer_model.ckpt')
    print("./transformer_model.ckpt - Model saved.")


# In[ ]:


if __name__ == '__main__':
    main()

