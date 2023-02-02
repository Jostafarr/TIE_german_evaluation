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

logger = logging.getLogger(__name__)

def set_seed(args):
    r"""
    Fix the random seed for reproduction.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()

def evaluate(args, model, tokenizer, prefix="", write_pred=True):
    r"""
    Evaluate the model
    """
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, split=args.evaluate_split)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    start_time = timeit.default_timer()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            if args.provided_tag_pred is not None:
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2]}
            else:
                inputs = {'input_ids'      : batch[0],
                          'attention_mask' : batch[1],
                          'token_type_ids' : batch[2],
                          'dom_mask'       : batch[-2],
                          'tag_to_tok'     : batch[-1]}
                if args.mask_method < 2:
                    inputs.update({'spa_mask': batch[-3]})
            if args.model_type == 'markuplm':
                inputs.update({
                    'xpath_tags_seq': batch[4],
                    'xpath_subs_seq': batch[5],
                })
                del inputs['token_type_ids']
            if args.model_type == 'roberta':
                del inputs['token_type_ids']
            feature_indices = batch[3]
            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)
            if args.merge_weight is not None:
                result = (RawResult(unique_id=unique_id,
                                    start_logits=to_list(outputs[1][i]),
                                    end_logits=to_list(outputs[2][i])),
                          RawTagResult(unique_id=unique_id,
                                       tag_logits=to_list(outputs[0][i]))
                          )
            else:
                if args.provided_tag_pred is not None:
                    result = RawResult(unique_id=unique_id,
                                       start_logits=to_list(outputs[0][i]),
                                       end_logits=to_list(outputs[1][i]))
                else:
                    result = RawTagResult(unique_id=unique_id,
                                          tag_logits=to_list(outputs[0][i]))
            all_results.append(result)

    eval_time = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", eval_time, eval_time / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_tag_prediction_file = os.path.join(args.output_dir, "tag_predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    output_result_file = os.path.join(args.output_dir, "qas_eval_results_{}.json".format(prefix))
    output_file = os.path.join(args.output_dir, "eval_matrix_results_{}".format(prefix))

    if args.merge_weight is not None:
        returns, _ = write_tag_predictions(examples, features,
                                           [r[1] for r in all_results], 1, args.model_type, output_tag_prediction_file,
                                           output_nbest_file, write_pred=False)
        returns = write_predictions_provided_tag(examples, features,
                                                 [r[0] for r in all_results], args.n_best_size,
                                                 args.max_answer_length, args.do_lower_case, output_prediction_file,
                                                 returns, output_tag_prediction_file, output_nbest_file,
                                                 args.verbose_logging, write_pred=write_pred)
    else:
        if args.provided_tag_pred is not None:
            returns = write_predictions_provided_tag(examples, features, all_results, args.n_best_size,
                                                     args.max_answer_length, args.do_lower_case, output_prediction_file,
                                                     args.provided_tag_pred, output_tag_prediction_file,
                                                     output_nbest_file, args.verbose_logging, write_pred=write_pred)
        else:
            returns = write_tag_predictions(examples, features, all_results, 1, args.model_type,
                                            output_tag_prediction_file, output_nbest_file, write_pred=write_pred)
            output_prediction_file = None

    if not write_pred:
        output_prediction_file, output_tag_prediction_file = returns

    # Evaluate with the official SQuAD script
    evaluate_options = EVAL_OPTS(data_file=args.predict_file,
                                 root_dir=args.root_dir,
                                 pred_file=output_prediction_file,
                                 tag_pred_file=output_tag_prediction_file,
                                 result_file=output_result_file if write_pred else None,
                                 out_file=output_file)
    results = evaluate_on_squad(evaluate_options)
    return results


def load_and_cache_examples(args, tokenizer, evaluate=False, split='train'):
    r"""
    Load and process the raw data.
    """
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache

    # Load data features from cache or dataset file
    input_file = args.predict_file if evaluate else args.train_file
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached', 'cached_{}_{}_{}'.format(
        split,
        list(filter(None, args.model_name_or_path.split('/'))).pop().split('_')[0],
        str(args.max_seq_length)))

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        if args.separate_read is not None and not evaluate:
            total = torch.load(cached_features_file + '_total')
            features = None
        else:
            features = torch.load(cached_features_file)
        examples, tag_list = read_examples(input_file=input_file,
                                           root_dir=args.root_dir,
                                           is_training=not evaluate,
                                           tokenizer=tokenizer,
                                           base_mode=args.model_type,
                                           simplify=False if evaluate else True)
        if not evaluate:
            tag_list = list(tag_list)
            tag_list.sort()
            tokenizer.add_tokens(tag_list)

    else:
        logger.info("Creating features from dataset file at %s", input_file)

        if not evaluate:
            examples, tag_list = read_examples(input_file=input_file,
                                               root_dir=args.root_dir,
                                               is_training=not evaluate,
                                               tokenizer=tokenizer,
                                               base_mode=args.model_type,
                                               simplify=True)
            tag_list = list(tag_list)
            tag_list.sort()
            tokenizer.add_tokens(tag_list)

        examples, _ = read_examples(input_file=input_file,
                                    root_dir=args.root_dir,
                                    is_training=not evaluate,
                                    tokenizer=tokenizer,
                                    base_mode=args.model_type,
                                    simplify=False)

        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                max_tag_length=args.max_tag_length,
                                                is_training=not evaluate,
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                pad_token=tokenizer.pad_token_id,)
        if args.local_rank in [-1, 0] and args.save_features:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
            if args.separate_read is not None and not evaluate:
                random.shuffle(features)
                total = len(features)
                num = ((total // 64) // args.separate_read) * 64
                for i in range(args.separate_read - 1):
                    torch.save(features[i * num:(i + 1) * num], cached_features_file + '_sub_{}'.format(i + 1))
                torch.save(features[(args.separate_read - 1) * num:],
                           cached_features_file + '_sub_{}'.format(args.separate_read))
                torch.save(total, cached_features_file + '_total')

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache

    if args.separate_read is not None and not evaluate:
        dataset = PiecedDataset(examples, evaluate, total, cached_features_file, args)
        return dataset

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_app_tags = [f.app_tags for f in features]
    all_example_index = [f.example_index for f in features]
    all_html_trees = [e.html_tree for e in examples]
    all_base_index = [f.base_index for f in features]
    all_tag_to_token = [f.tag_to_token_index for f in features]
    all_page_id = [f.page_id for f in features]
    if args.model_type == 'markuplm':
        all_xpath_tags_seq = torch.tensor([f.xpath_tags_seq for f in features], dtype=torch.long)
        all_xpath_subs_seq = torch.tensor([f.xpath_subs_seq for f in features], dtype=torch.long)
    else:
        all_xpath_tags_seq, all_xpath_subs_seq,  = None, None

    if evaluate:
        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = StrucDataset(all_input_ids, all_input_mask, all_segment_ids, all_feature_index,
                               all_xpath_tags_seq, all_xpath_subs_seq,
                               gat_mask=(all_app_tags, all_example_index, all_html_trees), base_index=all_base_index,
                               tag2tok=all_tag_to_token, shape=(args.max_tag_length, args.max_seq_length),
                               training=False, page_id=all_page_id, mask_method=args.mask_method,
                               mask_dir=os.path.dirname(input_file), direction=args.direction)
    else:
        all_answer_tid = torch.tensor([f.answer_tid for f in features], dtype=torch.long)
        if args.merge_weight is not None:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        else:
            all_start_positions, all_end_positions = None, None
        dataset = StrucDataset(all_input_ids, all_input_mask, all_segment_ids, all_answer_tid,
                               all_xpath_tags_seq, all_xpath_subs_seq, all_start_positions, all_end_positions,
                               gat_mask=(all_app_tags, all_example_index, all_html_trees), base_index=all_base_index,
                               tag2tok=all_tag_to_token, shape=(args.max_tag_length, args.max_seq_length),
                               training=True, page_id=all_page_id, mask_method=args.mask_method,
                               mask_dir=os.path.dirname(input_file), direction=args.direction)

    if evaluate:
        dataset = (dataset, examples, features)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str, required=True,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--root_dir", default=None, type=str, required=True,
                        help="the root directory of the raw WebSRC dataset, which contains the HTML files.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected from the models supported by huggingface, such as roberta, markuplm.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending "
                             "with step number")
    parser.add_argument('--eval_from_checkpoint', default=0, type=int,
                        help="Only evaluate the checkpoints with prefix larger than or equal to it, beside the final "
                             "checkpoint with no prefix")
    parser.add_argument('--eval_to_checkpoint', default=None, type=int,
                        help="Only evaluate the checkpoints with prefix smaller than it, beside the final checkpoint "
                             "with no prefix")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=float,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output "
                             "file.")

    parser.add_argument('--logging_steps', type=int, default=3000,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=3000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--save_features', default=True, type=bool,
                        help="whether or not to save the processed features, default is True")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    # added parameters
    parser.add_argument('--max_tag_length', default=512, type=int,
                        help="The maximum total tag length after the HTML based pooling. "
                             "Violation will cause error.")
    parser.add_argument('--separate_read', default=None, type=int,
                        help='if not none, shuffle and split the dataset into corresponding number of pieces.')
    parser.add_argument('--load_epoch', action='store_true')

    parser.add_argument('--mask_method', type=int, default=0,
                        help='How the GAT implement. 0: DOM+NPR; 1: NPR; 2: DOM; 3: Origin DOM.')
    parser.add_argument('--direction', default='B', choices=['V', 'H', 'B'],
                        help='The relations used in the NPR graph. V: only vertical relations (up & down); '
                             'H: only horizontal relations (left & right); B: both vertical and horizontal relations')

    parser.add_argument('--num_node_block', type=int, default=3,
                        help='The number of GAT layers of the Struct Encoder')
    parser.add_argument('--merge_weight', default=None, type=float,
                        help='If not none, share the parameters of Content Encoder and the model used for answer '
                             'refining stage and jointly train the models with the two losses and the specified weight '
                             'between them.')

    parser.add_argument('--evaluate_split', type=str, default='dev', choices=['dev', 'test', 'train'],
                        help='The part of dataset used for evaluation')
    parser.add_argument('--provided_tag_pred', type=str, default=None,
                        help='In the answer refining stage, the file that contain the answer tag predictions generated '
                             'in the node locating stage.')
    parser.add_argument('--state_dict', type=str, default='',
                        help='path to the state dict of the model')

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    # if args.server_ip and args.server_port:
    #     # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    #     import ptvsd
    #     print("Waiting for debugger attach")
    #     ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    #     ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    if args.model_type == 'markuplm':
        config = MarkupLMConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                                cache_dir=args.cache_dir)
        tokenizer = MarkupLMTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name
                                                      else args.model_name_or_path,
                                                      do_lower_case=args.do_lower_case, cache_dir=args.cache_dir)
    else:
        config = AutoConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                            cache_dir=args.cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name
                                                  else args.model_name_or_path,
                                                  do_lower_case=args.do_lower_case, cache_dir=args.cache_dir)
    if args.provided_tag_pred is not None:
        assert not args.do_train
        if args.model_type == 'markuplm':
            model = MarkupLMForQuestionAnswering.from_pretrained(args.model_name_or_path,
                                                                 from_tf=bool('.ckpt' in args.model_name_or_path),
                                                                 config=config, cache_dir=args.cache_dir)
        else:
            model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path,
                                                                  from_tf=bool('.ckpt' in args.model_name_or_path),
                                                                  config=config, cache_dir=args.cache_dir)
    else:
        tie_config = TIEConfig(args, **config.__dict__)
        model = TIE(tie_config, init_plm=True)
        state_dict = torch.load(args.state_dict, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model_to_save = model.module if hasattr(model, 'module') else model
        output_dir = os.path.join(args.output_dir, 'checkpoint-best')
        model_to_save.save_pretrained(output_dir )
        tokenizer.save_pretrained(args.output_dir)

    if args.local_rank == 0:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is
    # set. Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running
    # `--fp16_opt_level="O2"` will remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.provided_tag_pred is not None:
            logger.info("Evaluate the PLM provided with tags: %s", args.model_name_or_path)
            model.to(args.device)
            result = evaluate(args, model, tokenizer)
            results.update(result)
        else:
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME,
                                                                 recursive=True))
                )
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
            else:
                checkpoints = [args.output_dir]
    
            logger.info("Evaluate the following checkpoints: %s", checkpoints)

            if 'checkpoint' in args.output_dir:
                tokenizer_dir = os.path.split(args.output_dir[:-1])[0]
            else:
                tokenizer_dir = args.output_dir
            if args.model_type == 'markuplm':
                tokenizer = MarkupLMTokenizer.from_pretrained(tokenizer_dir, do_lower_case=args.do_lower_case)
            else:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, do_lower_case=args.do_lower_case)
            if model.config.vocab_size != len(tokenizer):
                model.resize_token_embeddings(len(tokenizer))

            for checkpoint in checkpoints:
                # Reload the model
                global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
                try:
                    int(global_step)
                except ValueError:
                    global_step = ""
                if global_step and int(global_step) < args.eval_from_checkpoint:
                    continue
                if global_step and args.eval_to_checkpoint and int(global_step) >= args.eval_to_checkpoint:
                    continue
                tie_config = TIEConfig.from_pretrained(checkpoint)
                model = TIE.from_pretrained(checkpoint, config=tie_config)
                model.to(args.device)

                # Evaluate
                result = evaluate(args, model, tokenizer, prefix=global_step)

                result = dict((k + ('_{}'.format(global_step) if global_step else ''), v) for k, v in result.items())
                results.update(result)

    logger.info("Results: {}".format(results))

    return results


if __name__ == "__main__":
    main()