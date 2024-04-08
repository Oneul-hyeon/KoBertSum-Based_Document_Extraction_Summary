import sys
import os
sys.path.append(os.getcwd() + '/src')

import copy
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_
from transformers import AutoModel, AutoTokenizer

from models.model_builder import *
from models.encoder import *

import argparse
import os
import math
import numpy as np

from transformers import BertTokenizer, BertModel
from prepro.tokenization_kobert import *
from prepro.tokenization_kobert import KoBertTokenizer
from kss import split_sentences

def make_parser() :
    parser = argparse.ArgumentParser()
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
    # parser.add_argument("-bert_data_path", default='../bert_data_new/cnndm')
    parser.add_argument("-model_path", default='../models/')
    # parser.add_argument("-result_path", default='../results/cnndm')
    parser.add_argument("-temp_dir", default='../temp')

    parser.add_argument("-batch_size", default=140, type=int)
    parser.add_argument("-test_batch_size", default=200, type=int)

    parser.add_argument("-max_pos", default=512, type=int)
    parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-large", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-load_from_extractive", default='', type=str)

    parser.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-lr_bert", default=2e-3, type=float)
    parser.add_argument("-lr_dec", default=2e-3, type=float)
    parser.add_argument("-use_bert_emb", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)

    parser.add_argument("-pretrained_model", default='bert', type=str)

    parser.add_argument("-mode", default='', type=str)
    parser.add_argument("-select_mode", default='greedy', type=str)
    parser.add_argument("-map_path", default='../../data/')
    parser.add_argument("-raw_path", default='../../line_data')
    parser.add_argument("-save_path", default='../../data/')

    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument('-min_src_nsents', default=3, type=int)    # 3
    parser.add_argument('-max_src_nsents', default=120, type=int)    # 100
    parser.add_argument('-min_src_ntokens_per_sent', default=1, type=int)    # 5
    parser.add_argument('-max_src_ntokens_per_sent', default=300, type=int)    # 200
    parser.add_argument('-min_tgt_ntokens', default=1, type=int)    # 5
    parser.add_argument('-max_tgt_ntokens', default=500, type=int)    # 500

    parser.add_argument("-lower", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-use_bert_basic_tokenizer", type=str2bool, nargs='?',const=True,default=False)

    # parser.add_argument('-log_file', default='../../logs/cnndm.log')

    parser.add_argument('-dataset', default='')

    parser.add_argument('-n_cpus', default=2, type=int)

    # params for EXT
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=2, type=int)
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha",  default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-max_tgt_len", default=140, type=int)

    args = parser.parse_args('')
    
    return args

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'

        self.sep_vid = self.tokenizer.convert_tokens_to_ids(self.sep_token)
        self.cls_vid = self.tokenizer.convert_tokens_to_ids(self.cls_token)
        self.pad_vid = self.tokenizer.convert_tokens_to_ids(self.pad_token)

    def preprocess(self, src):

        if (len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]
        idxs = [i for i, s in enumerate(src) if (len(s) > 1)]

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        src = src[:self.args.max_src_nsents]

        if (len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        #src_subtokens = src_subtokens[:4094]  ## 512가 최대인데 [SEP], [CLS] 2개 때문에 510
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        labels = None
        src_txt = [original_src_txt[i] for i in idxs]
        tgt_txt = None
        
        return src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt_txt

def summarize(text):
    def txt2input(text):
        bertdata = BertData(args)
        txt_data = bertdata.preprocess(text)
        # print(f'txt_data : {txt_data}')
        data_dict = {"src":txt_data[0],
                    "labels":[0,1,2],
                    "segs":txt_data[2],
                    "clss":txt_data[3],
                    "src_txt":txt_data[4]}
        input_data = []
        input_data.append(data_dict)
        return input_data
    
    input_data = txt2input(text)
    device = torch.device("cuda")
    
    def _pad(data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data
    
    pre_src = [x['src'] for x in input_data]
    pre_segs = [x['segs'] for x in input_data]
    pre_clss = [x['clss'] for x in input_data]

    src = torch.tensor(_pad(pre_src, 0)).cuda()
    segs = torch.tensor(_pad(pre_segs, 0)).cuda()
    mask_src = ~(src == 0)

    clss = torch.tensor(_pad(pre_clss, -1)).cuda()
    mask_cls = ~(clss == -1)
    clss[clss == -1] = 0

    clss.to(device).long()
    mask_cls.to(device).long()
    segs.to(device).long()
    mask_src.to(device).long()
    
    model_path = MODEL_DIR + MODEL_FILE
    checkpoint = torch.load(model_path)
    model = ExtSummarizer(args, device, checkpoint)
    model.eval()

    with torch.no_grad():
        sent_scores, mask = model(src, segs, clss, mask_src, mask_cls)
        sent_scores = sent_scores + mask.float()
        sent_scores = sent_scores.cpu().data.numpy()
        selected_ids = np.argsort(-sent_scores, 1)
    
    return selected_ids

def get_summary(text, sentence_cnt) :
    split_sentence = split_sentences(text, backend = 'Mecab')
    result = summarize(split_sentence)[0]
    return ' '.join([split_sentence[i] for i in result[:sentence_cnt]])

PROBLEM = 'ext'
PROJECT_DIR = os.getcwd()
DATA_DIR = f'{PROJECT_DIR}/{PROBLEM}/data'
RAW_DATA_DIR = DATA_DIR + '/raw'
JSON_DATA_DIR = DATA_DIR + '/json_data'
BERT_DATA_DIR = DATA_DIR + '/bert_data' 
LOG_DIR = f'{PROJECT_DIR}/{PROBLEM}/logs'
LOG_PREPO_FILE = LOG_DIR + '/preprocessing.log' 
MODEL_DIR = f'{PROJECT_DIR}/{PROBLEM}/models/' 
RESULT_DIR = f'{PROJECT_DIR}/{PROBLEM}/results'   
MODEL_FILE = "Oneul_KoBERT/model_step_28000.pt"
model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
            'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']

args = make_parser()
