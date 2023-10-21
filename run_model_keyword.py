import json
import random

import numpy as np
import argparse
from bm25 import BM25
import os
import torch

from transformers import ElectraTokenizer, BertTokenizer, \
    get_linear_schedule_with_warmup, get_constant_schedule
from data_keyword import *

from utils import *

import time
from torch.optim import AdamW
from tqdm import tqdm
from model_keyword import KeyWordModel


def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def strtime(datetime_checkpoint):
    diff = datetime.now() - datetime_checkpoint
    return str(diff).rsplit('.')[0]  # Ignore below seconds


def load_model(is_init, device, tokenizer, args):
    model = KeyWordModel(args.pretrained_model, args.keyword_num, device)

    if not is_init:
        state_dict = torch.load(args.model) if device.type == 'cuda' else \
            torch.load(args.model, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['sd'])
    return model


def configure_optimizer(args, model, num_train_examples):
    # https://github.com/google-research/bert/blob/master/optimization.py#L25
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr,
                      eps=args.adam_epsilon)

    num_train_steps = int(num_train_examples / args.B /
                          args.gradient_accumulation_steps * args.epochs)
    num_warmup_steps = int(num_train_steps * args.warmup_proportion)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps)

    return optimizer, scheduler, num_train_steps, num_warmup_steps


def configure_optimizer_simple(args, model, num_train_examples):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_train_steps = int(num_train_examples / args.B /
                          args.gradient_accumulation_steps * args.epochs)
    num_warmup_steps = 0

    scheduler = get_constant_schedule(optimizer)

    return optimizer, scheduler, num_train_steps, num_warmup_steps


def evaluate(model, data_loader, processed_kb, device, tokenizer, k, args):
    data_loader = tqdm(data_loader)
    corpus = [kb[0] for kb in processed_kb]
    ids = [kb[1] for kb in processed_kb]
    bmModel = BM25(corpus)
    labels = []
    res_label = []
    predict_time, whole_time = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            time_start = time.time()
            model.eval()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mention_tokens = batch["mention_tokens"]
            mention_tokens = [mention_token[0] for mention_token in mention_tokens]
            kb_id = batch["label"][0]
            token_ids = model(input_ids, attention_mask)
            time_end1 = time.time()
            tokens = tokenizer.convert_ids_to_tokens(token_ids.view(-1))

            tokens = clear_tokens_electra(tokens)
            key_words = []
            token_index = 0
            while len(key_words) < args.keyword_num and token_index < len(tokens):
                token = tokens[token_index]
                token_index += 1
                if token not in key_words:
                    key_words.append(token)
            if args.only_bm25:
                key_words = mention_tokens
            else:
                key_words += mention_tokens

            scores = bmModel.get_scores(key_words)
            scores = torch.tensor(scores)
            indices = scores.sort(-1, True).indices.tolist()
            sorted_ids = [ids[indice] for indice in indices]
            predicts = []
            for item in sorted_ids:
                if len(predicts) == k: break
                if item not in predicts:
                    predicts.append(item)
            res_label.append({kb_id: predicts})
            labels.append(kb_id in predicts)
            time_end2 = time.time()
            predict_time += time_end1 - time_start
            whole_time += time_end2 - time_start
    hitK = sum(labels) / len(labels)
    return hitK


def train(args):
    set_seeds(args)
    samples = read_data(args.data)
    random.shuffle(samples)
    samples_train = samples[:50]
    samples_dev = samples[50:100]
    samples_test = samples[100:]

    kb_train = read_kb(args.kb)
    kb_val = kb_test = kb_train


    logger = Logger(args.model + '.log', on=True)
    logger.log(str(args))
    args.logger = logger

    best_val_perf = float('-inf')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    logger.log(f'Using device: {str(device)}', force=True)
    logger.log('number of train entities {:d}'.format(len(kb_train)))
    logger.log('number of val entities {:d}'.format(len(kb_val)))
    logger.log('number of test entities {:d}'.format(len(kb_test)))
    tokenizer = ElectraTokenizer.from_pretrained(args.pretrained_model)
    special_tokens = ["[e1]", "[\e1]", '[stop]', "[deduce]"]
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    args.tokenizer = tokenizer

    model = load_model(True, device, tokenizer, args)

    num_train_samples = len(samples_train)
    if args.simpleoptim:
        optimizer, scheduler, num_train_steps, num_warmup_steps \
            = configure_optimizer_simple(args, model, num_train_samples)
    else:
        optimizer, scheduler, num_train_steps, num_warmup_steps \
            = configure_optimizer(args, model, num_train_samples)

    model.to(device)

    train_tokens, train_labels = process_train_data(samples_train, kb_train, tokenizer, args.max_text_len,
                                                    args.keyword_num)
    processed_dev_kb = process_test_kb(kb_val, tokenizer, args.window_size, "dev")
    processed_test_kb = processed_dev_kb
    train_loader = get_train_mention_loader(train_tokens, train_labels, tokenizer, args)
    dev_loader = get_test_mention_loader(samples_dev, tokenizer, args.max_text_len)
    test_loader = get_test_mention_loader(samples_test, tokenizer, args.max_text_len)
    effective_bsz = args.B * args.gradient_accumulation_steps
    # train
    logger.log('***** train *****')
    logger.log('# train samples: {:d}'.format(num_train_samples))
    logger.log('# val samples: {:d}'.format(len(samples_dev)))
    logger.log('# test samples: {:d}'.format(len(samples_test)))
    logger.log('# epochs: {:d}'.format(args.epochs))
    logger.log(' batch size : {:d}'.format(args.B))
    logger.log(' gradient accumulation steps {:d}'
               ''.format(args.gradient_accumulation_steps))
    logger.log(
        ' effective training batch size with accumulation: {:d}'
        ''.format(effective_bsz))
    logger.log(' # training steps: {:d}'.format(num_train_steps))
    logger.log(' # warmup steps: {:d}'.format(num_warmup_steps))
    logger.log(' learning rate: {:g}'.format(args.lr))
    logger.log(' # parameters: {:d}'.format(count_parameters(model)))

    step_num = 0
    tr_loss, logging_loss = 0.0, 0.0
    start_epoch = 1
    args.raw_kb = kb_test

    for epoch in range(start_epoch, args.epochs + 1):
        logger.log('\nEpoch {:d}'.format(epoch))

        epoch_start_time = datetime.now()

        epoch_train_start_time = datetime.now()
        train_loader = tqdm(train_loader)
        for step, batch in enumerate(train_loader):
            model.train()

            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch

            loss = model(input_ids, attention_mask, labels)

            loss_avg = loss / args.gradient_accumulation_steps

            loss_avg.backward()
            tr_loss += loss_avg.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                step_num += 1
        logger.log('training time for epoch {:3d} '
                   'is {:s}'.format(epoch, strtime(epoch_train_start_time)))

        hitK = evaluate(model, dev_loader, processed_dev_kb, device, tokenizer, args.k,args)
        logger.log('Done with epoch {:3d} | train loss {:8.4f} | '

                   'recall@k {:8.4f}'
                   ' epoch time {} '.format(
            epoch,
            tr_loss / step_num,

            hitK,
            strtime(epoch_start_time)
        ))

        save_model = (hitK >= best_val_perf)
        if save_model:
            current_best = hitK
            logger.log('------- new best val perf: {:g} --> {:g} '
                       ''.format(best_val_perf, current_best))

            best_val_perf = current_best
            torch.save({'opt': args,
                        'sd': model.state_dict(),

                        'perf': best_val_perf, 'epoch': epoch,
                        'opt_sd': optimizer.state_dict(),
                        'scheduler_sd': scheduler.state_dict(),
                        'tr_loss': tr_loss, 'step_num': step_num,
                        'logging_loss': logging_loss},
                       args.model)
        else:
            logger.log('')

    model = load_model(False, device, tokenizer, args)
    model.to(device)
    hitK = evaluate(model, test_loader, processed_test_kb, device, tokenizer,
                    args.k, args)
    logger.log(' '
               'recall@k {:8.4f}'
               ''.format(hitK))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="retriever_model/keyword_model.pt")
    parser.add_argument("--pretrained_model", default="google/electra-base-discriminator/")
    parser.add_argument("--data", default="zeshel/data/forgotten_realms.json")
    parser.add_argument("--kb", default="zeshel/kb/forgotten_realms.json")

    parser.add_argument("--max_text_len", default=128, type=int)
    parser.add_argument("--window_size", default=128)
    parser.add_argument("--keyword_num", default=32, type=int)
    parser.add_argument("--max_ent_len", default=128, type=int)

    parser.add_argument("--simpleoptim", default=False)
    parser.add_argument("--B", default=8, type=int)
    parser.add_argument("--epochs", default=12, type=int)
    parser.add_argument('--warmup_proportion', type=float, default=0.2,
                        help='proportion of training steps to perform linear '
                             'learning rate warmup for [%(default)g]')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay [%(default)g]')
    parser.add_argument('--adam_epsilon', type=float, default=1e-6,
                        help='epsilon for Adam optimizer [%(default)g]')
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--rands_ratio", default=0.9, type=float)
    parser.add_argument("--k", default=64, type=int)
    parser.add_argument("--use_gpu_index", default=False)
    parser.add_argument("--clip", default=1, type=int)
    parser.add_argument('--gpus', default='1', type=str,
                        help='GPUs separated by comma [%(default)s]')
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--only_bm25", action="store_true")

    args = parser.parse_args()
    # Set environment variables before all else.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus  # Sets torch.cuda behavior

    train(args)
