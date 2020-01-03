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
from apex.parallel import DistributedDataParallel as DDP
#https://on-demand.gputechconf.com/ai-conference-2019/T1-1_Jack%20Han_Getting%20More%20DL%20Training%20with%20Tensor%20Cores%20and%20AMP_%ED%95%9C%EC%9E%AC%EA%B7%BC_%EB%B0%9C%ED%91%9C%EC%9A%A9.pdf


def re_batch(src, trg):
    src = src.transpose(0,1)
    trg = trg.transpose(0,1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return src, trg, gold


def calc_loss(pred, gold, trg_pad_idx):
    gold = gold.contiguous().view(-1)
    
    '''label_smoothing'''
    eps = 0.1
    n_class = pred.size(1)
    
    #target word one-hot
    one_hot = torch.zeros_like(pred).scatter_(1, gold.view(-1,1),1)
    smoothing_one_hot = one_hot * (1 - eps) + (1- one_hot) * eps / (n_class-1)
    prob = F.log_softmax(pred, dim = 1)
    
    '''
        loss = - one_hot * prob 
        => 정확하게 예측하면 target값이 크기 때문에 곱셈 결과에 의해 loss 낮아짐
    '''
    #각 단어별 loss
    loss = -(smoothing_one_hot * prob).sum(dim=1)
    
    non_pad_mask = gold.ne(trg_pad_idx)
    loss = loss.masked_select(non_pad_mask).sum()
    #loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    pred = pred.argmax(dim=1)
    '''
        1. pred.eq(gold) : equal check.
        2. masked_select : True인 부분만 남겨, padding 부분 제외시킴.
    '''   
    n_word_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return loss, n_word, n_word_correct


def train_epoch(model, data_loader,optimizer,device,args):
    model.train()
    total_loss, n_word_total, n_word_correct=0,0,0
    desc = '  - (Training)   '
    for batch in tqdm(data_loader,mininterval=2, desc=desc, leave=False):    
        src_seq, trg_seq, gold = map(lambda x : x.to(device), re_batch(batch.src, batch.trg))
        
        optimizer.zero_grad()    
        
        pred = model(src_seq, trg_seq)
        #if loss : nan => zero division error
        eps= 1e-8
        pred = pred + eps
        loss, n_word, n_correct = calc_loss(pred, gold, args.trg_pad_idx)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
#        loss.backward()    
        optimizer.step()
        
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()
        
    accuracy  = n_word_correct / n_word_total
    loss_per_word = total_loss/ n_word_total
    
    return accuracy , loss_per_word


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

def print_status(epoch,loss, acc, start_time):
    print('epoch : {e:3d}, accuracy : {acc:3.3f}%, '\
          'loss: {loss:3.3f}, elapes : {time:3.3f}min'.format(
            e = epoch+1, acc = acc*100, loss = loss, time = (time.time()-start_time)/60))

def train(model, train_iter, val_iter, optimizer, device, args):
    
    val_losses =[]
    for e in range(args.epoch):
        start = time.time()
        checkpoint = {
                  'epoch':e,
                  'args': args, 
                  'model': model.state_dict()
                 }
        acc, loss = train_epoch(model, train_iter,optimizer,device,args)
        print_status(e,acc,loss,start)
 
        val_acc, val_loss = eval_epoch(model, val_iter ,device,args)
        print_status(e,acc,loss,start)       
 
        val_losses += [val_loss]
        if val_loss <= min(val_losses):
            torch.save(checkpoint, './transformer_model.ckpt')
            print("- checkpoint update  ")


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
#    transformer = nn.DataParallel(transformer)
    transformer, optimizer = amp.initialize(transformer, optimizer, opt_level="O1")
#    transformer = DDP(transformer)
    train(transformer, train_iter, val_iter, optimizer, device, args)


if __name__ == '__main__':
    main()

