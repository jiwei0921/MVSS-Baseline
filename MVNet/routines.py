import time
import torch
import statistics
import scipy.ndimage

from datasets.mvss_dataset import cmap
from utils.utils import class_to_RGB
from PIL import Image
import os
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import torch.distributed as dist

from utils.metrics import averageMeter, runningScore
from datasets.helpers import DATASETS_DICT
from utils.saving import save_labels_images
from utils.visualize import print_classes_report
from utils.iou_eval import IouEval
from models.losses import TrainingLoss



def scheduler_generator(args, optimizer):

    strategy = args.lr_strategy

    if strategy == "plateau_03":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3)
    elif strategy == "plateau_05":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)
    elif strategy == "plateau_09":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, min_lr=1e-9)
    elif strategy == "plateau_08":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, min_lr=1e-9)
    elif strategy == "plateau_07":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.7, min_lr=1e-9)
    elif strategy == "pow_07":
        lambda1 = lambda epoch: pow((1-((epoch-1)/args.num_epochs)), 0.7)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif strategy == "pow_09":
        lambda1 = lambda epoch: pow((1-((epoch-1)/args.num_epochs)), 0.9)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif strategy == "pow_09_b": # DEFAULT
        lambda1 = lambda epoch: pow((1-((epoch-1)/min(args.num_epochs, 50))), 0.9) if epoch < 50 else pow(0.02, 0.9)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    return scheduler

def epoch_routine(args, epoch, model, loader, optimizer,
                  criterion, is_train, shallow_dec, board, dataset, print_all_logs=True, gpu=None):

    epoch_loss = []
    times = []
    if is_train:
        tag = "TRAINING"
        tag_short = "TRAIN"
        evaluate_iou = args.iouTrain
        grad_context = torch.enable_grad()
        model.train() if model.module.is_training else model.eval()
    else:
        tag = "VALIDATE"
        tag_short = "VAL"
        evaluate_iou = args.iouVal
        grad_context = torch.no_grad()
        model.eval()

    if print_all_logs:
        print("-----", tag, "- EPOCH", epoch, "-----")

    n_eval_objs = 1 if args.dataset == "MVSeg" else args.win_size
    if evaluate_iou:        # For eval-mode
        iou_eval_objs = [IouEval(dataset.num_classes, -1) for i in range(n_eval_objs)]
        if args.use_orig_res:
            iou_eval_objs_full_res = [IouEval(dataset.num_classes, -1) for i in range(n_eval_objs)]

    total_img_count = 0
    for step, (images, thermals, labels, orig_images, orig_thermals, orig_labels,
               file_path, file_path_gt,
               images_filenames, labels_filenames) in enumerate(loader):

        step_time_start = time.time()
        seq_len = images.size(1)
        assert(seq_len == args.win_size)
        images = images.cuda(gpu)                               # images: batch, seq_len, channels, h, w
        thermals=thermals.cuda(gpu)
        labels = labels.cuda(gpu)                               # labels: batch, 1, channels, h, w
        step_img_count = images.size(1) * images.size(0)        # batch_size * seq_length
        total_img_count += step_img_count

        with grad_context:
            model.module.reset_hidden_state()
            probabilities, probabilities_aux, probabilities_thermal, probabilities_fusion, total_feas = \
                model(images, thermals, step, epoch)
            assert(probabilities.size(2) == dataset.num_classes)    # batch, seq_len, num_classes, h, w

        if is_train:
            optimizer.zero_grad()
            loss = criterion(probabilities, labels, probabilities_aux, probabilities_thermal, probabilities_fusion, total_feas)
            loss.backward()
            optimizer.step()
        else:
            loss = criterion(probabilities, labels, probabilities_aux, probabilities_thermal, probabilities_fusion, total_feas)
            # bs x classes x h x w | bs x h x w

        epoch_loss.append(loss.data)
        # Append time per image on that step
        times.append((time.time() - step_time_start) / step_img_count)

        # probabilities / pred_labels sizes:
        # torch.Size([b, t, 20, 512, 1024])
        # torch.Size([b, t, 1, 512, 1024])
        pred_labels = probabilities.max(dim=2, keepdim=True)[1].data

        if args.use_orig_res:
            pred_labels_numpy = probabilities.max(dim=2, keepdim=True)[1].data.cpu()
            pred_labels_full_res = scipy.ndimage.zoom(pred_labels_numpy, [1,1,1,2,2], order=0)
            pred_labels_full_res = torch.tensor(pred_labels_full_res).cuda()

        if args.always_decode:
            assert(pred_labels.size(1) == seq_len)
        else:
            assert(pred_labels.size(1) == 1)

        if evaluate_iou:
            for i in range(n_eval_objs):
                if args.dataset == "MVSeg":
                    assert(labels.size(1) == 1)
                    if args.always_decode:
                        assert(pred_labels.size(1) == seq_len)
                        t = seq_len - 1
                    else:
                        assert(pred_labels.size(1) == 1)
                        t = 0
                else:
                    assert(pred_labels.size(1) == seq_len)
                    t = i
                iou_eval_objs[i].add_batch(pred_labels[:, t, :, :, :], labels[:, i, :, :, :])
                if args.use_orig_res:
                    iou_eval_objs_full_res[i].add_batch(pred_labels_full_res[:, t, :, :, :], orig_labels[:, i, :, :, :])

                if print_all_logs and args.save_images and not shallow_dec:
                    if args.use_orig_res:
                        save_labels_images(dataset, args.savedir, epoch, t, orig_images, orig_labels, pred_labels_full_res)
                    else:
                        save_labels_images(dataset, args.savedir, epoch, t, images, labels, pred_labels)


        if args.visualize and print_all_logs:
            board.update_board_images(images, pred_labels, labels, epoch, step, tag_short, shallow_dec, dataset.colorize)

        mean_loss = sum(epoch_loss) / len(epoch_loss)

        if print_all_logs:
            if (args.steps_loss > 0 and step % args.steps_loss == 0) or step == len(loader) - 1:
                gpu_multiplier = args.world_size if is_train else 1
                report_str = "{} loss: {:.4f} | " \
                             "Epoch: {:3d} | Step: {:4d} | seq_len {:2d} | " \
                             "Total images for 1 gpu {:5d} | " \
                             "Median step time per image {:.4f} ms | Total images: {:5d}" \
                    .format(tag_short, mean_loss, epoch, step, seq_len,
                            total_img_count, 1000 * statistics.median(times), gpu_multiplier * total_img_count)
                print(report_str)

    epoch_miou = 0
    if print_all_logs and evaluate_iou:
        print("PREDICTION LOW  RESOLUTION:", pred_labels.shape)
        print("LABELS     LOW  RESOLUTION:", labels.shape)
        epoch_miou = print_classes_report(dataset, iou_eval_objs, tag_short, is_train)
        if args.use_orig_res:
            print("PREDICTION HIGH RESOLUTION:", pred_labels_full_res.shape)
            print("LABELS     HIGH RESOLUTION:", orig_labels.shape)
            _ = print_classes_report(dataset, iou_eval_objs_full_res, tag_short, is_train)

    return step, epoch_miou, mean_loss

def train(args, board, saver, model, gpu, rank, checkpoint):

    print_all_logs = gpu == 0

    interval = [args.win_size - 1, 0]   # interval = [3, 0], where args.win_size = 4.
    shallow_dec = args.shallow_dec
    dataset_train = DATASETS_DICT[args.dataset](args, 'train',
                                                co_transform=True, shallow_dec=shallow_dec, augment=True, interval=interval, print_all_logs=print_all_logs)
    ################################################################
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_train,
        num_replicas=args.world_size,
        rank=rank
    )
    ################################################################


    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, pin_memory=True)

    if print_all_logs:
        dataset_val = DATASETS_DICT[args.dataset](args, 'test',
                                                  co_transform=False, shallow_dec=shallow_dec, augment=False, interval=interval, print_all_logs=print_all_logs)

        loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    criterion = TrainingLoss(args, gpu)
    criterion.cuda(gpu)

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), float(args.lr_start), (0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    start_epoch = 1
    best_acc = 0

    if checkpoint is not None:
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        if print_all_logs:
            print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))

    scheduler = scheduler_generator(args, optimizer)

    end_epoch = args.num_epochs + 1

    dist.barrier()

    for epoch in range(start_epoch, end_epoch):
        train_sampler.set_epoch(epoch)
        used_lr = 0
        for param_group in optimizer.param_groups:
            used_lr = float(param_group['lr'])
            if print_all_logs:
                print("LEARNING RATE: ", used_lr)

        step, mean_iou_train, mean_loss_train = epoch_routine(
            args, epoch, model, loader, optimizer, criterion,
            is_train=True, shallow_dec=shallow_dec, board=board, dataset=dataset_train, print_all_logs=print_all_logs, gpu=gpu)

        dist.barrier()
        if print_all_logs:
            # Validate on 500 val images after each epoch of training
            step, mean_iou_val, mean_loss_val = epoch_routine(
                args, epoch, model, loader_val, optimizer, criterion,
                is_train=False, shallow_dec=shallow_dec, board=board, dataset=dataset_val, gpu=gpu)

            # Remember best valIoU and save checkpoint
            if mean_iou_val == 0:
                current_acc = -mean_loss_val
            else:
                current_acc = mean_iou_val
            is_best = current_acc > best_acc
            best_acc = max(current_acc, best_acc)

            checkpoint_state_dict = {
                'epoch': epoch + 1,
                'arch': str(model),
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            saver.save_checkpoint(checkpoint_state_dict, is_best, shallow_dec)
            saver.save_model(model, shallow_dec, is_best, epoch, step, mean_iou_val)
            saver.save_epoch_report(shallow_dec, epoch, mean_loss_train, mean_loss_val, mean_iou_train, mean_iou_val, used_lr)

            if args.visualize:
                board.update_graphs(shallow_dec, epoch, float(mean_loss_train), float(mean_loss_val),
                                    float(mean_iou_val), used_lr)
                board.save_dashboard(saver.save_dir)

        # With pow strategies, should be
        #scheduler.step(epoch)
        scheduler.step(mean_loss_train)


def epoch_routine_test(args, epoch, model, loader, optimizer,
                  criterion, is_train, shallow_dec, board, dataset, print_all_logs=True, gpu=None):

    epoch_loss = []
    times = []

    tag = "TEST"
    tag_short = "TEST"
    evaluate_iou = args.iouVal
    grad_context = torch.no_grad()
    model.eval()


    print("-----", tag, "-------", tag, "-----")

    # MVSeg: only one true label for a video clip.
    n_eval_objs = 1 if args.dataset == "MVSeg" else args.win_size

    iou_eval_objs = [IouEval(dataset.num_classes, -1) for i in range(n_eval_objs)]
    if args.use_orig_res:
        iou_eval_objs_full_res = [IouEval(dataset.num_classes, -1) for i in range(n_eval_objs)]

    running_metrics_val = runningScore(dataset.num_classes, ignore_index=None)
    running_metrics_val.reset()

    total_img_count = 0
    for step, (images, thermals, labels, orig_images, orig_thermals, orig_labels,
               file_path, file_path_gt,
               images_filenames, labels_filenames) in enumerate(loader):

        step_time_start = time.time()
        seq_len = images.size(1)
        assert(seq_len == args.win_size)
        images = images.cuda(gpu)                               # images: batch, seq_len, channels, h, w
        thermals=thermals.cuda(gpu)
        labels = labels.cuda(gpu)                               # labels: batch, 1, channels, h, w
        step_img_count = images.size(1) * images.size(0)        # batch_size * seq_length
        total_img_count += step_img_count

        with grad_context:
            model.module.reset_hidden_state()
            probabilities, probabilities_aux, probabilities_thermal, probabilities_fusion, total_feas = \
                model(images, thermals, step, epoch)
            assert(probabilities.size(2) == dataset.num_classes)    # batch, seq_len, num_classes, h, w


        loss = criterion(probabilities, labels, probabilities_aux, probabilities_thermal, probabilities_fusion, total_feas)
        # bs x classes x h x w | bs x h x w

        epoch_loss.append(loss.data)
        # Append time per image on that step
        times.append((time.time() - step_time_start) / step_img_count)

        # probabilities / pred_labels sizes:
        # torch.Size([b, t, 20, 512, 1024])
        # torch.Size([b, t, 1, 512, 1024])
        pred_labels = probabilities.max(dim=2, keepdim=True)[1].data

        if args.use_orig_res:
            pred_labels_numpy = probabilities.max(dim=2, keepdim=True)[1].data.cpu()
            pred_labels_full_res = scipy.ndimage.zoom(pred_labels_numpy, [1,1,1,2,2], order=0)
            pred_labels_full_res = torch.tensor(pred_labels_full_res).cuda()

        if args.always_decode:
            assert(pred_labels.size(1) == seq_len)
        else:
            assert(pred_labels.size(1) == 1)


        for i in range(n_eval_objs):
            if args.dataset == "MVSeg":
                assert(labels.size(1) == 1)
                if args.always_decode:
                    assert(pred_labels.size(1) == seq_len)
                    t = seq_len - 1
                else:
                    assert(pred_labels.size(1) == 1)
                    t = 0
            else:
                assert(pred_labels.size(1) == seq_len)
                t = i
            iou_eval_objs[i].add_batch(pred_labels[:, t, :, :, :], labels[:, i, :, :, :])
            if args.use_orig_res:
                iou_eval_objs_full_res[i].add_batch(pred_labels_full_res[:, t, :, :, :], orig_labels[:, i, :, :, :])


            if not args.use_orig_res:
                predict = pred_labels[:, t, :, :, :].cpu().numpy()
            else:
                predict = pred_labels_full_res[:, t, :, :, :].cpu().numpy()

            # i.e., only one GT for eval. We compute the related metrics.
            if args.dataset == "MVSeg":
                running_metrics_val.update(labels[:, i, :, :, :].cpu().numpy(), predict)
            else:
                print('The metrics only support MVSeg!')
                exit(1)


            #Prediction_Get_Colored = True
            Prediction_Get_Colored = False

            if Prediction_Get_Colored:
                # Saving Predict
                color_map = cmap()
                predict = predict.squeeze()
                predict = class_to_RGB(predict, N=len(color_map), cmap=color_map)
                predict = Image.fromarray(predict)
                filename = "{}l.png".format(images_filenames[-1][0][:-5])
                predict_path = os.path.join(args.savedir, 'predicts')
                if not os.path.exists(predict_path):
                    os.mkdir(predict_path)
                video_name = file_path[0].split('/')[-3]
                save_predict_path = os.path.join(predict_path, video_name)
                if not os.path.exists(save_predict_path):
                    os.mkdir(save_predict_path)
                image_path = os.path.join(save_predict_path, filename)
                predict.save(image_path)
            else:
                predict = predict.squeeze()
                filename = "{}l.png".format(images_filenames[-1][0][:-5])
                predict_path = os.path.join(args.savedir, 'predicts')
                if not os.path.exists(predict_path):
                    os.mkdir(predict_path)
                video_name = file_path[0].split('/')[-3]
                save_predict_path = os.path.join(predict_path, video_name)
                if not os.path.exists(save_predict_path):
                    os.mkdir(save_predict_path)
                image_path = os.path.join(save_predict_path, filename)
                import cv2
                cv2.imwrite(image_path, predict)


        if args.visualize and print_all_logs:
            board.update_board_images(images, pred_labels, labels, epoch, step, tag_short, shallow_dec, dataset.colorize)

        mean_loss = sum(epoch_loss) / len(epoch_loss)

        if print_all_logs:
            if (args.steps_loss > 0 and step % args.steps_loss == 0) or step == len(loader) - 1:
                gpu_multiplier = args.world_size if is_train else 1
                report_str = "{} loss: {:.4f} | " \
                             "Epoch: {:3d} | Step: {:4d} | seq_len {:2d} | " \
                             "Total images for 1 gpu {:5d} | " \
                             "Median step time per image {:.4f} ms | Total images: {:5d}" \
                    .format(tag_short, mean_loss, epoch, step, seq_len,
                            total_img_count, 1000 * statistics.median(times), gpu_multiplier * total_img_count)
                print(report_str)

    epoch_miou = 0
    if print_all_logs and evaluate_iou:
        print("PREDICTION LOW  RESOLUTION:", pred_labels.shape)
        print("LABELS     LOW  RESOLUTION:", labels.shape)
        epoch_miou = print_classes_report(dataset, iou_eval_objs, tag_short, is_train)
        if args.use_orig_res:
            print("PREDICTION HIGH RESOLUTION:", pred_labels_full_res.shape)
            print("LABELS     HIGH RESOLUTION:", orig_labels.shape)
            _ = print_classes_report(dataset, iou_eval_objs_full_res, tag_short, is_train)


    # Print all metrics
    metrics = running_metrics_val.get_scores()
    print('overall metrics .....')
    for k, v in metrics[0].items():
        print(k, f'{v:.4f}')

    print('iou for each class .....')
    for k, v in metrics[1].items():
        print(k, f'{v:.4f}')
    print('acc for each class .....')
    for k, v in metrics[2].items():
        print(k, f'{v:.4f}')


    return step, epoch_miou, mean_loss


def eval(args, board, saver, model, gpu, rank):

    print_all_logs = gpu == 0

    interval = [args.win_size - 1, 0]
    shallow_dec = args.shallow_dec
    assert(args.num_epochs == 1)

    split = args.split_mode

    dataset_val = DATASETS_DICT[args.dataset](args, split,
                                              co_transform=False, shallow_dec=shallow_dec, augment=False, interval=interval, print_all_logs=print_all_logs)

    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    criterion = TrainingLoss(args, gpu)
    criterion.cuda(gpu)

    step, mean_iou_eval, mean_loss_eval = epoch_routine_test(
        args, args.num_epochs, model, loader_val, None, criterion,
        is_train=False, shallow_dec=shallow_dec, board=board, dataset=dataset_val, gpu=gpu)


