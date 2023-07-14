# -*- coding: utf-8 -*-
import os
import time
import torch

from argparse import ArgumentParser

import torch.multiprocessing as mp
import torch.distributed as dist

import routines
from utils.saving import Saver
from utils.visualize import Dashboard, print_optimized_model_params
from utils.loading import load_model_from_file, load_checkpoint


def process(gpu, args):

    ############################################################
    rank = args.nr * args.gpus + gpu
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    ############################################################

    torch.manual_seed(0)

    board = None
    if gpu == 0:
        if args.visualize and args.steps_plot > 0:
            board = Dashboard(args)

    # Save arguments for reference
    saver = Saver(args)
    checkpoint = None
    if args.resume:
        checkpoint = load_checkpoint(saver.save_dir, args.shallow_dec)

    model_path = os.path.join("models", args.model)     # Loading Model file, i.e., mvnet.py
    model = load_model_from_file(args, model_path, board, gpu, checkpoint)

    if gpu == 0:
        # Load Model and save a copy of it for reference
        saver.save_model_copy(model_path)
        print_optimized_model_params(model)
        saver.save_txtmodel(model, args.shallow_dec)

    # Use that model for training and/or inference
    if args.training:
        routines.train(args, board, saver, model, gpu, rank, checkpoint)
    else:
        routines.eval(args, board, saver, model, gpu, rank)

def main(args):

    mp.spawn(process, nprocs=args.gpus, args=(args, ))

    print("========== PROCESSING FINISHED ===========")


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset', default="MVSeg")

    # Model selection
    parser.add_argument('--model', default="mvnet.py")
    #parser.add_argument('--weights', default="trained_models/erfnet_pretrained.pth")
    parser.add_argument('--weights', default=False)
    parser.add_argument('--backbone', type=str, default="deeplab50")

    # Network structure
    parser.add_argument('--backbone-nobn', action='store_true')
    parser.add_argument('--encoder-eval-mode', action='store_true')
    parser.add_argument('--decoder-eval-mode', action='store_true')
    parser.add_argument('--train-decoder', action='store_true')
    parser.add_argument('--train-encoder', action='store_true')
    parser.add_argument('--train-erfnet-shallow-dec', action='store_true', default=False)
    parser.add_argument('--shallow-dec', action='store_true', default=False)
    parser.add_argument('--use-orig-res', action='store_true')
    parser.add_argument('--always-decode', action='store_true')

    # Training options
    parser.add_argument('--eval-mode', action='store_true')
    parser.add_argument('--split-mode', default='test')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--steps-loss', type=int, default=100)
    parser.add_argument('--steps-plot', type=int, default=50)
    parser.add_argument('--epochs-save', type=int, default=0)
    parser.add_argument('--lr-strategy', default='pow_09')
    parser.add_argument('--lr-start', default='5e-5')
    parser.add_argument('--loss', default="")

    # Multi GPU training
    parser.add_argument('--nodes', default=1, type=int, metavar='N')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--mport', type=str, default="8888")

    # Debug output / Visdom options
    parser.add_argument('--savedir', required=True)
    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--save-images', action='store_true')

    # Evaluation options
    parser.add_argument('--iouTrain', action='store_true', default=False)
    parser.add_argument('--iouVal', action='store_true', default=True)
    parser.add_argument('--save-all-vals', action='store_true')
    parser.add_argument('--baseline-mode', action='store_true')
    parser.add_argument('--baseline-path', default="<TO DEFINE>")

    # LMANet related options
    parser.add_argument('--win-size', type=int, default=-1, help='if not given, mem size + 1')
    parser.add_argument('--model-struct', type=str, default="original")
    parser.add_argument('--align-weights', action='store_true', default=True)
    parser.add_argument('--local-correlation', action='store_true', default=False)
    parser.add_argument('--corr-size', type=int, default=21)
    parser.add_argument('--learnable-constant', action='store_true')
    parser.add_argument('--memorize-first', action='store_true')
    parser.add_argument('--fusion-strategy', type=str, default="sigmoid-do1")

    parser.add_argument('--memory-strategy', type=str, default="all", help='all or random')
    parser.add_argument('--stm-queue-size', type=int, default=3)
    parser.add_argument('--sample-rate', type=int, default=1)

    # Augment related options
    parser.add_argument('--random-crop', action='store_true', default=True)

    args = parser.parse_args()

    #########################################################
    args.world_size = args.gpus * args.nodes                #
    os.environ['MASTER_ADDR'] = 'localhost'                 #
    os.environ['MASTER_PORT'] = args.mport                  #
    #########################################################

    if args.eval_mode == args.training:
        print("Cannot be at the same time in training and evaluation mode!")
        exit(1)

    # Options simplifications
    if args.eval_mode:
        args.num_epochs = 1
        args.batch_size = 1

    if args.win_size == -1:
        args.win_size = args.stm_queue_size + 1

    start_time = time.time()
    
    main(args)
    
    print("Total time: {0:6.4f} s".format(time.time() - start_time))
