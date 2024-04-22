import sys
import os
sys.path.append(os.getcwd() + '/src')

import argparse
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers import AutoModel
from torch.nn.init import xavier_uniform_

from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder
from models.optimizers import Optimizer

from kss import split_sentences
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer
import numpy as np

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
parser = argparse.ArgumentParser()
parser.add_argument("-task", default='ext', type=str, choices=['ext', 'abs'])
parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
parser.add_argument("-bert_data_path", default='../bert_data_new/cnndm')
parser.add_argument("-model_path", default='../models/')
parser.add_argument("-result_path", default='../results/cnndm')
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

parser.add_argument("-param_init", default=0, type=float)
parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
parser.add_argument("-optim", default='adam', type=str)
parser.add_argument("-lr", default=1, type=float)
parser.add_argument("-beta1", default= 0.9, type=float)
parser.add_argument("-beta2", default=0.999, type=float)
parser.add_argument("-warmup_steps", default=8000, type=int)
parser.add_argument("-warmup_steps_bert", default=8000, type=int)
parser.add_argument("-warmup_steps_dec", default=8000, type=int)
parser.add_argument("-max_grad_norm", default=0, type=float)

parser.add_argument("-save_checkpoint_steps", default=5, type=int)
parser.add_argument("-accum_count", default=1, type=int)
parser.add_argument("-report_every", default=1, type=int)
parser.add_argument("-train_steps", default=1000, type=int)
parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)

parser.add_argument('-visible_gpus', default='-1', type=str)
parser.add_argument('-gpu_ranks', default='0', type=str)
parser.add_argument('-log_file', default='../logs/cnndm.log')
parser.add_argument('-seed', default=666, type=int)

parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
parser.add_argument("-test_from", default='')
parser.add_argument("-test_start_from", default=-1, type=int)
parser.add_argument("-make_gold", default="false", type=str)
parser.add_argument("-train_from", default='')
parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("-model", default=None, type=str, choices=["KoBERT", "KoBigBird"])

parser.add_argument("-shard_size", default=2000, type=int)
parser.add_argument('-min_src_nsents', default=1, type=int)    # 3
parser.add_argument('-max_src_nsents', default=120, type=int)    # 100
parser.add_argument('-min_src_ntokens_per_sent', default=1, type=int)    # 5
parser.add_argument('-max_src_ntokens_per_sent', default=300, type=int)    # 200
parser.add_argument('-min_tgt_ntokens', default=1, type=int)    # 5
parser.add_argument('-max_tgt_ntokens', default=500, type=int)    # 500

args = parser.parse_args()

MODEL_DIR = "ext/models"
BEST_MODEL = "model_step_25000.pt"
class BertData():
    def __init__(self, args):
        self.args = args
        # self.tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert", do_lower_case=True)
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base") if args.model == "KoBigBird" else BertTokenizer.from_pretrained("monologg/kobert")
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '¶' # '[unused0]'   204; 314[ 315]
        self.tgt_eos = '----------------' # '[unused1]'
        self.tgt_sent_split = ';' #'[unused2]'

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
        src_subtokens = src_subtokens[:args.max_pos-2]

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
    
class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()

        self.model = BertModel.from_pretrained("monologg/kobert", cache_dir=temp_dir)
        self.finetune = finetune

    def forward(self, x, segs, mask):
        if(self.finetune):
            output = self.model(x, token_type_ids=segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                output = self.model(x, token_type_ids=segs, attention_mask=mask)
        return output["last_hidden_state"]

class BigBird(nn.Module) :
    def __init__(self, large, temp_dir, finetune=False) :
        super(BigBird, self).__init__()
        self.model = AutoModel.from_pretrained("monologg/kobigbird-bert-base", attention_type = "original_full")
        self.finetune = finetune
    
    def forward(self, x, mask) :
        if self.finetune :
            output = self.model(x, attention_mask=mask).last_hidden_state
        else :
            self.eval()
            with torch.no_grad() :
                output = self.model(x, attention_mask=mask).last_hidden_state
        return output
    
class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert) if args.model == "KoBERT" else BigBird(args.large, args.temp_dir, args.finetune_bert)
        self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers)

        if(args.model == "KoBERT") and (args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src) if self.args.model == "KoBERT" else self.bert(src, mask = mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls
    
def summarize(text):
    def txt2input(text):
        bertdata = BertData(args)
        txt_data = bertdata.preprocess(text)
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
    
    model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']
    
    pt = f'{MODEL_DIR}/{args.model}/{BEST_MODEL}'
    test_from = ""
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])

    model = ExtSummarizer(args, device, checkpoint)
    model.eval()

    with torch.no_grad():
        sent_scores, mask = model(src, segs, clss, mask_src, mask_cls)
        sent_scores = sent_scores + mask.float()
        sent_scores = sent_scores.cpu().data.numpy()
        selected_ids = np.argsort(-sent_scores, 1)
    
    return selected_ids

def get_summary(text, sentence_cnt, model) :
    global args, MODEL_FILE
    if model == "KoBigBird" : 
        args.model = "KoBigBird"
        args.max_pos = 1024
    elif model == "KoBERT" :
        args.model = "KoBERT"
        args.max_pos = 512

    split_sentence = split_sentences(text, backend = 'Mecab')
    result = summarize(split_sentence)[0]
    return ' '.join([split_sentence[i] for i in result[:sentence_cnt]])