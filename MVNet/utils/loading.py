import torch
import os

from models.mvnet import MVNet

models = {
    "mvnet": MVNet
}


def load_model_from_file(args, model_path, board, gpu, checkpoint):

    if not os.path.exists(model_path):
        print("Could not load model, file does not exist: ", model_path)
        exit(1)

    print_all_logs = gpu == 0

    model_name = str(os.path.basename(model_path).split('.')[0])
    model = models[model_name](args, print_all_logs=print_all_logs, board=board)
    if print_all_logs:
        print("Loaded model file: ", model_path)

    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    if args.gpus > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    weights_path = args.weights
    if weights_path:
        if os.path.exists(weights_path):
            if print_all_logs:
                print("Loading weights file: ", weights_path)
        else:
            print("Could not load weights, file does not exist: ", weights_path)

        if checkpoint is not None:
            weights_dict = checkpoint
        else:
            weights_dict = torch.load(weights_path, map_location=lambda storage, loc: storage)


        model.load_state_dict(weights_dict)

        if print_all_logs:
            print("Loaded weights:", weights_path)

    if args.training:
        model.train()
    else:
        model.eval()

    #filter_model_params_optimization(args, model)

    return model



def load_checkpoint(save_dir, enc):
    tag = "_enc" if enc else ""
    checkpoint_path = os.path.join(save_dir, "checkpoint{}.pth.tar".format(tag))

    if os.path.exists(checkpoint_path):
        print("Loading checkpoint file: ", checkpoint_path)
    else:
        print("Could not load checkpoint, file does not exist: ", checkpoint_path)

    return torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
