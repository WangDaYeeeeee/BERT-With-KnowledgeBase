import argparse
import time

import torch

from trainer import Trainer
from utils import init_logger, load_tokenizer, set_seed
from data_loader import load_and_cache_dataset

parser = argparse.ArgumentParser()

parser.add_argument("--task", default='atis', type=str, choices=['atis', 'snips'],
                    help="The name of the task to train.")
parser.add_argument("--kb", default='both', type=str, choices=['both', 'wn', 'nell', 'none'],
                    help="The composition of knowledge base.")
parser.add_argument("--decoder", default="stack", type=str, choices=['stack', 'unstack', 'none'],
                    help="Knowledge decoder type.")
parser.add_argument("--schedule", default="linear", type=str, choices=['linear', 'cosine', 'constant'],
                    help="The schedule of training process.")
parser.add_argument("--attn", default="general", type=str,
                    choices=['dot', 'scaled_dot', 'general', 'concat', 'perceptron'], help="Attention method.")
parser.add_argument("--data_dir", default="../data", type=str, help="The input data dir")

parser.add_argument("--num_epochs", default=20, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size for training and evaluation.")
parser.add_argument("--max_seq_len", default=50, type=int,
                    help="The maximum total input sequence length after tokenization.")
parser.add_argument("--max_wn_concepts_count", default=50, type=int,
                    help="The maximum WordNet concepts count of a word.")
parser.add_argument("--max_nell_concepts_count", default=30, type=int,
                    help="The maximum NELL concepts count of a word.")
parser.add_argument("--top_concepts_count", default=80, type=int,
                    help="Top concepts count in knowledge attention.")
parser.add_argument("--weight_decay", default=0., type=float, help="Weight decay if we apply some.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--adam_lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--num_warmup_steps", default=0, type=float,
                    help="Warmup steps of schedule. If less than 1, that means the proportion of total training steps.")

parser.add_argument('--seed', type=int, default=0, help="Random seed for initialization.")
parser.add_argument("--eval_epochs", default=5, type=int,
                    help="Evaluate model every X epochs 5. Only evaluate the last epoch if it's 0.")

parser.add_argument("--intent_decoder_dropout", "--id_dr", default=0., type=float, help="Intent decoder dropout.")
parser.add_argument("--knowledge_attn_dropout", "--ka_dr", default=0., type=float, help="Knowledge attention dropout.")
parser.add_argument("--context_attn_dropout", "--ca_dr", default=0., type=float, help="Knowledge context attn dropout.")
parser.add_argument("--knowledge_decoder_dropout", "--kd_dr", default=0., type=float, help="Knowledge decoder dropout.")
parser.add_argument("--classifier_dropout", "--c_dr", default=0.1, type=float, help="Classifier layers dropout.")

# parser.add_argument("--full", action="store_true", help="Whether to use full-scale dataset.")
parser.add_argument("--pos", action="store_true", help="Whether to enable the pos-embedding in knowledge integrator.")
parser.add_argument("--uni_intent", action="store_true", help="Whether to disable the intent decoder.")
parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available.")
parser.add_argument('--log_to_file', action="store_true", help="Whether to make logger output log info to file.")

parser.add_argument("--ignore_index", default=-100, type=int,
                    help='Specifies a target value that is ignored and does not contribute to the input gradient.')

parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')

# For eval and pred.
parser.add_argument("--do_eval", default=None, type=str, help="Path of saved model. '{record_path}/models/{epoch}'")
parser.add_argument("--pred_dir", default="./preds", type=str, help="The input prediction dir.")
parser.add_argument("--pred_input_file", default="preds.txt", type=str,
                    help="The input text file of lines for prediction.")
parser.add_argument("--pred_output_file", default="outputs.txt", type=str, help="The output file of prediction.")

# CRF option
parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
parser.add_argument("--pad_label", default="PAD", type=str,
                    help="Pad token for slot label pad. (to be ignore when calculate loss)")

args = parser.parse_args()
args.model_name_or_path = '../bert/'
args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
args.do_train = args.do_eval is None

if args.kb == 'nell':
    args.max_wn_concepts_count = 0
elif args.kb == 'wn':
    args.max_nell_concepts_count = 0
elif args.kb == 'none':
    args.max_wn_concepts_count = 0
    args.max_nell_concepts_count = 0

args.record_path = '{}_{}_{}_{}_{}_{}'.format(
    args.task,
    f'{args.decoder}{"+uni" if args.uni_intent else ""}{"+pos" if args.pos else ""}',
    f'seed{args.seed}' if args.do_train else 'eval-pred',
    f'seq{args.max_seq_len}',
    f'{args.kb}',
    time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime(time.time()))
)


if __name__ == '__main__':
    init_logger(args)
    set_seed(args.seed)
    tokenizer = load_tokenizer(args)

    trainer = Trainer(args=args,
                      train_data=load_and_cache_dataset(args, tokenizer, mode='train'),
                      dev_data=load_and_cache_dataset(args, tokenizer, mode='dev'),
                      test_data=load_and_cache_dataset(args, tokenizer, mode='test'))

    if args.do_train:
        trainer.train()
        if args.eval_epochs == 0 or args.num_epochs % args.eval_epochs != 0:
            trainer.evaluate('dev', args.num_epochs, tensorboard_enabled=False)
        trainer.evaluate('test', args.num_epochs, tensorboard_enabled=False)
        trainer.save_model(args.record_path, args.num_epochs)

    if args.do_eval is not None:
        trainer.load_model(args.do_eval)
        trainer.evaluate('dev', args.num_epochs, tensorboard_enabled=False)
        trainer.evaluate('test', args.num_epochs, tensorboard_enabled=False)
