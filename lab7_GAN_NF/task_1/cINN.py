"""Train Glow on CIFAR-10.

Train script adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
import argparse
import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import util
from dataloader2 import *
from evaluator import evaluation_model
from flow_model import Glow
from tqdm import tqdm


def main(args):
    # Set up main device and scale batch size
    device = 'cuda' if torch.cuda.is_available() and args.gpu_ids else 'cpu'
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    trainloader = data.DataLoader(ICLEVRLoader('./'), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_condition = get_iCLEVR_data('./', 'test')[1]
    test_condition = torch.Tensor(test_condition).float()
    test_condition = test_condition.to(device)

    # Model
    print('Building model..')
    net = Glow(num_channels=args.num_channels,
               num_levels=args.num_levels,
               num_steps=args.num_steps,
               img_shape=(3,64,64),
               mode=args.mode)
    net = net.to(device)
    evaluator = evaluation_model()
    
    loss_fn = util.NLLLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = sched.LambdaLR(optimizer, lambda s: min(1., s / args.warm_up))
    start_epoch = 0

    if args.resume:
        # Load checkpoint.
        print('Resuming from checkpoint')
        checkpoint = torch.load('savemodel/cINN/checkpoint_18.tar')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        global best_loss
        global global_step
        # best_loss = checkpoint['test_loss']
        start_epoch = checkpoint['epoch']
        global_step = start_epoch * len(trainloader.dataset)


    score_list = []

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train(epoch, net, trainloader, device, optimizer, scheduler,
              loss_fn, args.max_grad_norm)
        # test(epoch, net, test_condition, device, loss_fn, args.mode)
        score = test(epoch, net, test_condition, device, evaluator)
        score_list.append(score)
    
    score_list = np.asarray(score_list)
    print('Best epoch: %d\nBest score: %f' % (np.argmax(score_list), np.max(score_list)))


@torch.enable_grad()
def train(epoch, net, trainloader, device, optimizer, scheduler, loss_fn, max_grad_norm):
    global global_step
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = util.AverageMeter()
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, cond_x in trainloader:
            x , cond_x = x.to(device), cond_x.to(device)
            optimizer.zero_grad()
            z, sldj = net(x, cond_x, reverse=False)
            loss = loss_fn(z, sldj)
            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
            if max_grad_norm > 0:
                util.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()
            scheduler.step(global_step)

            progress_bar.set_postfix(nll=loss_meter.avg,
                                     bpd=util.bits_per_dim(x, loss_meter.avg),
                                     lr=optimizer.param_groups[0]['lr'])
            progress_bar.update(x.size(0))
            global_step += x.size(0)
    
    print('Saving...')
    state = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, 'savemodel/cINN/checkpoint_' + str(epoch) + '.tar')


@torch.no_grad()
def sample(net, test_condition, device, sigma=0.6):
    # B, C, W, H = gray_img.shape
    B = len(test_condition)
    z = torch.randn((B, 3, 64, 64), dtype=torch.float32, device=device) * sigma
    x, _ = net(z, test_condition, reverse=True)
    x = torch.sigmoid(x)

    return x


@torch.no_grad()
# def test(epoch, net, test_condition, device, loss_fn, mode='color'):
def test(epoch, net, test_condition, device, evaluator):
    global best_loss
    net.eval()
    # loss_meter = util.AverageMeter()
    gen_img = sample(net, test_condition, device)
    score = evaluator.eval(gen_img, test_condition)
    print('Score: ', score)
    images_concat = torchvision.utils.make_grid(gen_img, nrow=8)
    torchvision.utils.save_image(images_concat, 'visualization/cINN/epoch_{}.png'.format(epoch))

    # with tqdm(total=len(testloader.dataset)) as progress_bar:
    #     for x, x_cond in testloader:
    #         x, x_cond = x.to(device), x_cond.to(device)
    #         z, sldj = net(x, x_cond, reverse=False)
    #         loss = loss_fn(z, sldj)
    #         loss_meter.update(loss.item(), x.size(0))
    #         progress_bar.set_postfix(nll=loss_meter.avg,
    #                                  bpd=util.bits_per_dim(x, loss_meter.avg))
    #         progress_bar.update(x.size(0))

    # Save checkpoint
    # if loss_meter.avg < best_loss:
    

    return score

    # origin_img, gray_img = next(iter(testloader))
    # B = gray_img.shape[0]
    # # Save samples and data
    # images = sample(net, gray_img, device)
    # os.makedirs('samples', exist_ok=True)
    # os.makedirs('ref_pics', exist_ok=True)
    # if mode == 'sketch':
    #     gray_img = (~gray_img.type(torch.bool)).type(torch.float)
    # images_concat = torchvision.utils.make_grid(images, nrow=int(B ** 0.5), padding=2, pad_value=255)
    # origin_concat = torchvision.utils.make_grid(origin_img, nrow=int(B ** 0.5), padding=2, pad_value=255)
    # gray_concat = torchvision.utils.make_grid(gray_img, nrow=int(B ** 0.5), padding=2, pad_value=255)

    # torchvision.utils.save_image(images_concat, 'samples/epoch_{}.png'.format(epoch))
    # torchvision.utils.save_image(origin_concat, 'ref_pics/origin_{}.png'.format(epoch))
    # torchvision.utils.save_image(gray_concat, 'ref_pics/gray_{}.png'.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Glow on CIFAR-10')

    def str2bool(s):
        return s.lower().startswith('t')

    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU')
    parser.add_argument('--benchmark', type=str2bool, default=True, help='Turn on CUDNN benchmarking')
    parser.add_argument('--gpu_ids', default=[0], type=eval, help='IDs of GPUs to use')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=-1., help='Max gradient norm for clipping')
    parser.add_argument('--num_channels', '-C', default=128, type=int, help='Number of channels in hidden layers')
    parser.add_argument('--num_levels', '-L', default=3, type=int, help='Number of levels in the Glow model')
    parser.add_argument('--num_steps', '-K', default=8, type=int, help='Number of steps of flow in each level')
    parser.add_argument('--num_epochs', default=20, type=int, help='Number of epochs to train')
    # parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', type=str2bool, default=False, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--warm_up', default=500000, type=int, help='Number of steps for lr warm-up')
    parser.add_argument('--mode', default="sketch", choices=['gray', 'sketch'])
    best_loss = float('inf')
    global_step = 0

    main(parser.parse_args())
