import argparse
import time
import os
from tqdm import tqdm
import pickle

import spacy
import torch
from torchtext import data, datasets

from Models import Transformer
from train import re_batch,calc_loss,eval_epoch
import torch.nn as nn

def test(model, test_iter, device, args):
    epoch = args.epoch
    start = time.time()
            
    val_acc, val_loss_per_word = eval_epoch(model, test_iter ,device,args)
    print('-  (  Test Data set : WMT14  ) ')
    print('Accuracy : {acc:3.3f}%, Loss_per_word : {per_word:3.3f}%, elapse : {elapse:3.3f} min'.format(acc=val_acc*100,per_word=val_loss_per_word ,elapse=(time.time()-start)/60))


def data_loader(args, device):
    if not os.path.exists(".data/iwslt/de-en"):
        print("Data download.")
        os.system("python -m spacy download en_core_web_sm")
        os.system("python -m spacy download de_core_news_sm")
        
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
    
    test, _, _ = datasets.Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TGT),
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)
    
    '''단어장 로딩'''    
    SRC.build_vocab()
    TGT.build_vocab()
    SRC.vocab = pickle.load(open(args.src_vocab, 'rb'))
    TGT.vocab = pickle.load(open(args.trg_vocab, 'rb'))

    args.num_src_vocab = len(SRC.vocab.stoi)
    args.num_trg_vocab = len(TGT.vocab.stoi)
    args.src_pad_idx = SRC.vocab.stoi['<pad>']
    args.trg_pad_idx = TGT.vocab.stoi['<pad>']

    test_iter = data.BucketIterator(test, batch_size=BATCH_SIZE,device = device, sort_key=lambda x: len(x.src)) #,shuffle=True,repeat=False)
    return test_iter, SRC, TGT


def load_model(new_args, device):
    checkpoint = torch.load(new_args.model)
    args = checkpoint['args']
    
    transformer =  Transformer(args.num_src_vocab,
                          args.num_trg_vocab,
                          args.src_pad_idx,
                          args.trg_pad_idx,
                          n_layers = args.n_layers,
                          d_model = args.d_model,
                          head_num = args.head_num,
                          fc_dim = args.fc_dim,
                          seq_len = args.seq_len,
                          dropout=args.dropout).to(device)
    transformer = nn.DataParallel(transformer)
    transformer.load_state_dict(checkpoint['model'])
    
    return transformer


def main(): 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-epoch',type=int, default = 10)
    parser.add_argument('-batch',type=int, default = 512)
    parser.add_argument('-d_model',type=int, default= 512)
    parser.add_argument('-n_layers',type=int, default = 6)
    parser.add_argument('-head_num',type = int, default= 8)
    parser.add_argument('-fc_dim',type=int, default=2048)
    parser.add_argument('-seq_len',type=int, default=200)
    parser.add_argument('-dropout',type=float, default=0.1)
    parser.add_argument('-n_warmup_steps',type=int, default=4000)
    parser.add_argument('-model', default = './transformer_model.ckpt')
    parser.add_argument('-src_vocab',default = './data/SRC_vocab_pkl')
    parser.add_argument('-trg_vocab',default = './data/TGT_vocab_pkl')

    args = parser.parse_args()
    
    args.weight_sharing = not args.weight_sharing
    device = torch.device('cuda')
    
    test_iter,_,_ = data_loader(args,device)
    transformer = load_model(args,device)
    
    test(transformer, test_iter, device, args)


if __name__ == '__main__':
    main()

