import argparse
import time
import os
from tqdm import tqdm
import pickle

import spacy
import torch
from torchtext import data, datasets

from Models import Transformer, get_pad_mask, subsequent_mask, make_std_mask
from train import calc_loss,eval_epoch
from test import data_loader, load_model
import torch.nn as nn

def Greedy_Decoder(transformer, src_seq, src_mask ,max_len, start_symbol, end_symbol, trg_pad_idx, device):
    #nn.DataParallel => model.module. ...
    encoder_output = transformer.module.encoder(src_seq,src_mask)
    decoder_input = torch.ones(1, 1).fill_(start_symbol).type_as(src_seq.data)
    #decoder_input = torch.ones(1,max_len).type_as(src_seq.data)
    #next_word = start_symbol
    for i in range(max_len-1):
        #decoder_input[0][i] = next_word
        trg_mask = make_std_mask(decoder_input, trg_pad_idx)
        decoder_output = transformer.module.decoder(decoder_input, encoder_output, src_mask, trg_mask)
        prob = transformer.module.generator(decoder_output[:,-1])
        next_word = torch.argmax(prob, dim = 1).data[0]
        decoder_input = torch.cat([decoder_input,torch.ones(1, 1).type_as(src_seq.data).fill_(next_word)], dim=1)
        if next_word == end_symbol:
            break
    return decoder_input


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-epoch',type=int, default = 10)
    parser.add_argument('-batch',type=int, default = 1)
    parser.add_argument('-d_model',type=int, default= 512)
    parser.add_argument('-n_layers',type=int, default = 6)
    parser.add_argument('-head_num',type = int, default= 8)
    parser.add_argument('-fc_dim',type=int, default=2048)
    parser.add_argument('-seq_len',type=int, default=200)
    parser.add_argument('-dropout',type=float, default=0.1)
    parser.add_argument('-n_warmup_steps',type=int, default=4000)
    parser.add_argument('-n_gen_sentence',type=int, default=4)
    parser.add_argument('-weight_sharing', action='store_true')
    parser.add_argument('-model', default = './transformer_model.ckpt')
    parser.add_argument('-src_vocab',default = './data/SRC_vocab_pkl')
    parser.add_argument('-trg_vocab',default = './data/TGT_vocab_pkl')

    args = parser.parse_args()
    
    args.weight_sharing = not args.weight_sharing
    device = torch.device('cuda')
    
    test_iter, SRC, TGT = data_loader(args,device)
    transformer = load_model(args,device)
    
    start_symbol = TGT.vocab.stoi['<s>']
    end_symbol = TGT.vocab.stoi['</s>']
    
    '''Translate'''
    for number, batch in enumerate(test_iter):
        src_seq = batch.src.transpose(0,1)[:1]
        src_mask = get_pad_mask(src_seq, SRC.vocab.stoi['<pad>'])
        output_index = Greedy_Decoder(transformer,
                       src_seq, 
                       src_mask,
                       args.seq_len, 
                       start_symbol,
                       end_symbol,
                       args.trg_pad_idx,
                       device)
        
        print(" - Translation : ",end="\t")
        for i in range(1,output_index.size(1)):
            word = TGT.vocab.itos[output_index[0,i]]
            if word =='</s>' : 
                break
            print(word, end = " ")
        print()
        print(" Target sentence : ", end='\t')
        for i in range(1, batch.trg.size(0)):
            word = TGT.vocab.itos[batch.trg.data[i,0]]
            if word == '</s>' :
                break
            print(word, end = " ")
        print(end='\n\n')

        if number == args.n_gen_sentence:
            break
            

if __name__ == '__main__':
    main()

