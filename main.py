import os
import argparse
from utils.setup_utils import *
from train import train, evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='mwoz')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu', default="0", type=str,
                    help="Use CUDA on the device.")
parser.add_argument('--base_model_name', default='bert-base-uncased',
                    type=str, help="Base model name.")

parser.add_argument("--batch_size", default=16, type=int,
                    help="Batch size for training.")
parser.add_argument("--epoch_num", default=5, type=int,
                    help="Maximum number of training epochs.")
parser.add_argument("--max_seq_len", default=64, type=int,
                    help="Maximum input sequence length.")
parser.add_argument("--max_dial_len", default=16, type=int,
                    help="Maximum input dialogue length.")
parser.add_argument("--eval_steps", default=100, type=int)

parser.add_argument("--lr", default=2e-5, type=float,
                    help="Learning rate.")
parser.add_argument("--gamma", default=0.5, type=float,
                    help="Weighting parameter.")

parser.add_argument("--dropout", default=0.1, type=float,
                    help="Dropout rate.")
parser.add_argument("--sat_num", default=3, type=int)
parser.add_argument("--warmup", default=0.1, type=float,
                    help="Warmup ratio.")

parser.add_argument('--rewrite_data', action='store_true',
                    help='Rewrite data pickle.')
parser.add_argument('--use_hp', action='store_false',
                    help='Whether to use the Hawkes transformer.')

parser.add_argument('--train_base', action='store_true',
                    help='If true, the ground-truth labels will be used to optimize the base estimator and the entire model simultaneously. This can help the base model make better predictions. As a result, the Hawkes process module may learn the satisfaction dynamics patterns more accurately. ') 

parser.add_argument('--eval', action='store_true',
                    help='Evaluation only.') 
parser.add_argument('--eval_set', default='test',
                    type=str, help='Evaluation Split Set.')  

parser.add_argument("--tf_heads", default=12, type=int,
                    help="Number of heads in turn-level transformer.")
parser.add_argument("--tf_layers", default=2, type=int,
                    help="Number of layers in turn-level transformer.")

parser.add_argument("--hp_heads", default=12, type=int,
                    help="Number of heads in Hawkes transformer.")
parser.add_argument("--hp_layers", default=2, type=int,
                    help="Number of layers in Hawkes transformer.")

args = parser.parse_args()

print('train data', args.data)

# Setup CUDA, GPU & distributed training
device, device_id = set_cuda(args)
args.device = device
args.device_id = device_id

if args.eval:
    evaluate(args)
else:
    train(args)
    