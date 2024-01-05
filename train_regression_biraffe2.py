import sys
import os
import torch
import torch.distributed as dist
import warnings
import torch.distributed
import random
import faulthandler
import time

from data_regression_biraffe2 import Biraffe2Dataset
from data_regression_biraffe2_test import Biraffe2DatasetTest
from models.networks_regression_biraffe2 import HyperRegression
from args import get_args
from torch.backends import cudnn
from utils import AverageValueMeter, set_random_seed, resume, save
import matplotlib.pyplot as plt
from torch import optim
import torch.multiprocessing as mp


faulthandler.enable()


def main_worker(gpu, save_dir, args):
    # basic setup
    cudnn.benchmark = True
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    train_data = Biraffe2Dataset(args.data_dir)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    test_data = Biraffe2DatasetTest(args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=1, shuffle=True, num_workers=0, pin_memory=True
    )

    args.input_size = len(train_data.x_labels)
    args.output_size = len(train_data.y_labels)
    args.input_dim = 2

    model = HyperRegression(args)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    start_epoch = 0
    optimizer = model.make_optimizer(args)
    if args.resume_checkpoint is None and os.path.exists(os.path.join(save_dir, "checkpoint-latest.pt")):
        args.resume_checkpoint = os.path.join(save_dir, "checkpoint-latest.pt")  # use the latest checkpoint
    if args.resume_checkpoint is not None:
        if args.resume_optimizer:
            model, optimizer, start_epoch = resume(
                args.resume_checkpoint, model, optimizer, strict=(not args.resume_non_strict)
            )
        else:
            model, _, start_epoch = resume(
                args.resume_checkpoint, model, optimizer=None, strict=(not args.resume_non_strict)
            )
        print("Resumed from: " + args.resume_checkpoint)

    
    # initialize the learning rate scheduler
    if args.scheduler == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.exp_decay)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 2, gamma=0.1)
    elif args.scheduler == 'linear':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - 0.5 * args.epochs) / float(0.5 * args.epochs)
            return lr_l
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:
        assert 0, "args.schedulers should be either 'exponential' or 'linear'"

    # main training loop
    start_time = time.time()
    point_nats_avg_meter = AverageValueMeter()
    if args.distributed:
        print("[Rank %d] World size : %d" % (args.rank, dist.get_world_size()))

    print("Start epoch: %d End epoch: %d" % (start_epoch, args.epochs))
    for epoch in range(start_epoch, args.epochs):
        # adjust the learning rate
        if (epoch + 1) % args.exp_decay_freq == 0:
            scheduler.step(epoch=epoch)

        print("Epoch starts:")
        for bidx, data in enumerate(train_loader):
            x, y = data
            x = x.float().to(args.gpu)
            y = y.float().to(args.gpu).unsqueeze(1)
            step = bidx + len(train_loader) * epoch
            model.train()
            recon_nats = model(x, y, optimizer, step, None)
            point_nats_avg_meter.update(recon_nats.item())
            if step % args.log_freq == 0:
                duration = time.time() - start_time
                start_time = time.time()
                print(
                    "[Rank %d] Epoch %d Batch [%2d/%2d] Time [%3.2fs] PointNats %2.5f"
                    % (args.rank, epoch, bidx, len(train_loader), duration, point_nats_avg_meter.avg)
                )
        # save visualizations
        if (epoch + 1) % args.viz_freq == 0:
            # reconstructions
            model.eval()
            for bidx, data in enumerate(test_loader):
                x, _, subject = data
                x = x.squeeze().float().to(args.gpu)
                _, y_pred = model.decode(x, 100)
                y_pred = y_pred.cpu().detach().numpy()
                y_pred = y_pred.squeeze()
                valence = y_pred[:, 0]
                arousal = y_pred[:, 1]

                plt.xlim([-1, 1])
                plt.ylim([-1, 1])
                plt.scatter(valence, arousal)

                filepath = os.path.join(save_dir, "images", "result_epoch%d_%s.png" % (epoch, subject[0]))
                plt.savefig(filepath)
                plt.close()

        if (epoch + 1) % args.save_freq == 0:
            save(model, optimizer, epoch + 1, os.path.join(save_dir, "checkpoint-%d.pt" % epoch))
            save(model, optimizer, epoch + 1, os.path.join(save_dir, "checkpoint-latest.pt"))


def main():
    # command line args
    args = get_args()

    # override args in order to debug
    args.gpu = 0
    args.log_name = "biraffe2_v2"
    args.data_dir = "data/BIRAFFE2"

    args.num_blocks = 1
    args.log_freq = 1
    args.viz_freq = 1
    args.save_freq = 1

    args.lr = 1e-3
    args.weight_decay = 1e-5
    args.epochs = 100
    args.batch_size = 1024
    args.dims = '16-16-16'
    args.hyper_dims = '64-16'

    save_dir = os.path.join("checkpoints", args.log_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, "images"))

    with open(os.path.join(save_dir, "command.sh"), "w") as f:
        f.write("python -X faulthandler " + " ".join(sys.argv))
        f.write("\n")

    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    set_random_seed(args.seed)

    if args.gpu is not None:
        warnings.warn("You have chosen a specific GPU. This will completely " "disable data parallelism.")

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if args.sync_bn:
        assert args.distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(save_dir, ngpus_per_node, args))
    else:
        main_worker(args.gpu, save_dir, args)


if __name__ == "__main__":
    main()
