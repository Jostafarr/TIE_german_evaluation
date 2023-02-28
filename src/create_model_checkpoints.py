
from __future__ import absolute_import, division, print_function
import sys
sys.path.append("/Users/jostgotte/Documents/Uni/WS2223/rtiai/TIE/")

import argparse
import logging
import os
import random
import glob
import timeit

from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from tensorboardX import SummaryWriter
from transformers import (
    WEIGHTS_NAME,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from markuplmft.models.markuplm import MarkupLMConfig, MarkupLMTokenizer, MarkupLMForQuestionAnswering

from model import TIEConfig, TIE
from dataset import StrucDataset, PiecedDataset
from utils import read_examples, convert_examples_to_features, RawResult, RawTagResult,\
    write_tag_predictions, write_predictions_provided_tag
from utils_evaluate import EVAL_OPTS, main as evaluate_on_squad


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_type", type=str, default="markuplm", 
                        help="Model type selected from the models supported by huggingface, such as roberta, markuplm.")
    parser.add_argument("--model_name_or_path", default="microsoft/markuplm-large-finetuned-websrc", type=str, 
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument('--num_node_block', type=int, default=3,
                        help='The number of GAT layers of the Struct Encoder')
    parser.add_argument('--merge_weight', default=None, type=float,
                        help='If not none, share the parameters of Content Encoder and the model used for answer '
                             'refining stage and jointly train the models with the two losses and the specified weight '
                             'between them.')
    parser.add_argument('--mask_method', type=int, default=0,
                        help='How the GAT implement. 0: DOM+NPR; 1: NPR; 2: DOM; 3: Origin DOM.')
    parser.add_argument('--direction', default='B', choices=['V', 'H', 'B'],
                        help='The relations used in the NPR graph. V: only vertical relations (up & down); '
                             'H: only horizontal relations (left & right); B: both vertical and horizontal relations')
            
            
    args = parser.parse_args()   
            
    config = MarkupLMConfig.from_pretrained("microsoft/markuplm-large-finetuned-websrc", cache_dir= "data/cached")
    tokenizer = MarkupLMTokenizer.from_pretrained("microsoft/markuplm-large-finetuned-websrc",
                                                    do_lower_case=False, cache_dir="data/cached")

    tie_config = TIEConfig(args, **config.__dict__)
    model = TIE(tie_config, init_plm=True)
    state_dict = torch.load("checkpoints/TIE_WebSRC.bin", map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_output_dir = os.path.join("result/TIE_MarkupLM/", 'checkpoint-best')
    model_to_save.save_pretrained(model_output_dir )
    tokenizer.save_pretrained("result/TIE_MarkupLM/")


if __name__ == "__main__":
    main()



    