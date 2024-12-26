# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import time

from datetime import timedelta

from tqdm import tqdm
from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader

import jittor as jt
from jittor import nn
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    # print("preds shape: ", preds.shape)
    # print("labels shape: ", labels.shape)
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)
    return (preds == labels).mean()

def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    checkpoint = {
        'model': model_to_save.state_dict(),
    }
    jt.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    config.split = args.split
    config.slide_step = args.slide_step

    if args.dataset == "CUB_200_2011":
        num_classes = 200
    elif args.dataset == "car":
        num_classes = 196
    elif args.dataset == "nabirds":
        num_classes = 555
    elif args.dataset == "dog":
        num_classes = 120
    elif args.dataset == "INat2017":
        num_classes = 5089

    model = VisionTransformer(config, args.no_contrastive, args.alpha, args.img_size, zero_head=True, num_classes=num_classes, smoothing_value=args.smoothing_value)
    
    model.load_from(np.load(args.pretrained_dir))
    # for p in model.parameters():
        # p.requires_grad = True
    print("model num params before load pretrained model: ", count_parameters(model))
    if args.pretrained_model is not None:
        pretrained_model = jt.load(args.pretrained_model)['model']
        model.load_state_dict(pretrained_model)
        print("Load pretrained model from %s" % args.pretrained_model)
    # model.to(args.device)
    # all need grad
    # for p in model.parameters():
        # p.requires_grad = True
    num_params = count_parameters(model)
    # print("num_params: ", num_params)
    
    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model
    
def count_parameters(model):
    # params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = sum(p.numel() for p in model.parameters() if p.is_stop_grad() == False)
    return params/1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    
def valid(args, model, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []

    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                        #   bar_format="{l_bar}{r_bar}",
                          bar_format="{l_bar}{bar:20}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = nn.CrossEntropyLoss()
    # print("Length of test_loader: ", len(epoch_iterator))
    import itertools
    
    print(f"Running validation, global_step: {global_step}")
    for step, batch in enumerate(epoch_iterator):
        # batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with jt.no_grad():
            logits = model(x)

            eval_loss = loss_fct(logits, y)
            eval_loss = eval_loss.mean()
            eval_losses.update(eval_loss.item())

            preds = jt.argmax(logits, dim=-1)[0]

        # if len(all_preds) == 0:
        #     all_preds.append(preds.numpy())
        #     all_label.append(y.numpy())
        # else:
        #     all_preds[0] = np.append(all_preds[0], preds.numpy(), axis=0)
        #     all_label[0] = np.append(all_label[0], y.numpy(), axis=0)
        all_preds += preds.numpy().tolist()
        all_label += y.numpy().tolist()

        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

        # break

    # all_preds, all_label = all_preds[0], all_label[0]
    # accuracy = simple_accuracy(all_preds, all_label)
    print("all_label shape: ", len(all_label))
    print("all_preds shape: ", len(all_preds))
    accuracy = accuracy_score(all_label, all_preds)
    
    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)
        
    return accuracy

def train(args, model, global_step=0):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
    
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    
    # print("这是调试信息 -----")
    # print("train_batch_size: ", args.train_batch_size)
    # print(model)
    
    # Prepare dataset
    train_loader, test_loader = get_loader(args)
    # print("train loader len: ", len(train_loader))
    # print("test loader len: ", len(test_loader))
    
    # Prepare optimizer and scheduler
    optimizer = jt.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    # model.zero_grad()
    # optimizer.zero_grad()
    set_seed(args)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    # best_acc = 0
    start_time = time.time()
    while True:
        # print("这是调试信息 train 1 -----")
        model.train()
        # print("这是调试信息 train 2 -----")
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        all_preds, all_label = [], []
        for step, batch in enumerate(epoch_iterator):
            # batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            loss, logits = model(x, y)
            loss = loss.mean()
            preds = jt.argmax(logits, dim=-1)[0]
        
            all_preds += preds.numpy().tolist()
            all_label += y.numpy().tolist()
            # if len(all_preds) == 0:
            #     to_append = preds.numpy()
            #     all_preds.append(to_append)
            #     all_label.append(y.numpy())
            # else:
            #     all_preds[0] = np.append(all_preds[0], preds.numpy(), axis=0)
            #     all_label[0] = np.append(all_label[0], y.numpy(), axis=0)
                
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            # loss.backward()
            optimizer.step(loss)
                
            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)
                optimizer.clip_grad_norm(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )

                if global_step % args.eval_every == 0:
                    with jt.no_grad():
                        accuracy = valid(args, model, test_loader, global_step)
                    if args.local_rank in [-1, 0]:
                        if best_acc < accuracy:
                            save_model(args, model)
                            best_acc = accuracy
                        logger.info("best accuracy so far: %f" % best_acc)
                    model.train()

                if global_step % t_total == 0:
                    break

        # all_preds, all_label = all_preds[0], all_label[0]
        # accuracy = simple_accuracy(all_preds, all_label)
        print("all_label shape: ", len(all_label))
        print("all_preds shape: ", len(all_preds))
        accuracy = accuracy_score(all_label, all_preds)
        logger.info("train accuracy so far: %f" % accuracy)
        losses.reset()
        if global_step % t_total == 0:
            break

    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    end_time = time.time()
    logger.info("Total Training Time: \t%f" % ((end_time - start_time) / 3600))
    
def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--name", required=True, help="Name of this run. Used for monitoring.")
    parser.add_argument("--name", default="test_inat_memory", help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "dog", "nabirds", "INat2017"], default="INat2017", help="Which dataset.")
    parser.add_argument("--data_root", type=str, default="../TransFG/data")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16", "ViT-L_32", "ViT-H_14"], default="ViT-B_16", help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="../TransFG/data/ViT-B_16.npz", help="Where to search for pretrained ViT models.")
    parser.add_argument("--pretrained_model", type=str, default=None, help="load pretrained model")
    parser.add_argument("--output_dir", default="./output", type=str, help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=448, type=int, help="Resolution size")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=1000, type=int, help="Run prediction on validation set every so many steps."
                        "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float, help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=30000, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--global_step", default=0, type=int, help="Global step to start training from.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine", help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int, help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2', help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0, help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")
    
    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")

    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")
    
    parser.add_argument('--no_contrastive', action='store_true', help="Whether to use contrastive learning")
    
    parser.add_argument('--alpha', type=float, default=0.4, help="Contrastive loss weight")

    args = parser.parse_args()
    
    args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    # args.device = 'cuda' if jt.flags.use_cuda else 'cpu'

    args.n_gpu = 1
    jt.flags.use_cuda = 1
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
                        filename=f'log/{args.name}.log'
                    )
    logger.warning("Process rank: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.n_gpu, False, False))

    # Set seed
    set_seed(args)

    # Setup model
    args, model = setup(args)

    # checkpoint_path = os.path.join(args.output_dir, f"{args.name}_checkpoint.bin")
    # if os.path.exists(checkpoint_path):
    #     logger.info(f"Loading checkpoint from {checkpoint_path}")
    #     checkpoint = jt.load(checkpoint_path)
    #     model.load_state_dict(checkpoint['model'])
    #     # global_step = checkpoint.get('global_step', 0)
    #     global_step = args.global_step
    #     logger.info(f"Resuming training from global step {global_step}")
    # else:
    #     logger.info("No checkpoint found. Starting from scratch.")
    #     global_step = 0

    # Train!
    # train(args, model, global_step)
    train(args, model, args.global_step)

if __name__ == "__main__":
    main()
