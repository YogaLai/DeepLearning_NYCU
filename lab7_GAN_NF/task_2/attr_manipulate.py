from numpy.core.numeric import zeros_like
from model import Glow
import torch
import argparse
from PIL import Image
from torchvision import transforms, utils
from dataset import *

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_block", default=3, type=int, help="number of blocks")
    parser.add_argument(
        "--no_lu",
        action="store_true",
        help="use plain convolution instead of LU decomposed version",
    )
    parser.add_argument(
        "--affine", action="store_true", help="use affine coupling instead of additive"
    )
    parser.add_argument(
        "--n_flow", default=32, type=int, help="number of flows in each block"
    )
    parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
    parser.add_argument("--img_size", default=64, type=int)
    parser.add_argument("--batch_size", default=18, type=int)
    args = parser.parse_args()

    model_single = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    model = torch.nn.DataParallel(model_single)
    model = model.to(device)
    ckt = torch.load('savemodel/checkpoint_44.tar')
    model.load_state_dict(ckt['net'])

    dataset = CelebALoader('./', args.img_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    with_attr_cnt = 0
    wo_attr_list = []
    with_attr_z = None

    n_bins = 2.0 ** args.n_bits
    for iters, (image, attr) in enumerate(dataloader):
        with_attr_idx = []
        wo_attr_idx = []
        for i in range(len(attr)):
            if attr[i][-1]==1 and attr[i][8]==1: # -1 young, 8 black hair
                with_attr_idx.append(i)
            else:
                wo_attr_idx.append(i)
        with_attr_idx = torch.tensor(with_attr_idx, dtype=torch.int64)
        wo_attr_idx = torch.tensor(wo_attr_idx, dtype=torch.int64)
        with_attr_cnt += len(with_attr_idx)

        image = image.to(device)
        image = image * 255
        if args.n_bits < 8:
            image = torch.floor(image / 2 ** (8 - args.n_bits))
        image = image / n_bins - 0.5
        log_p, logdet, z = model(image + torch.rand_like(image) / n_bins)
        if with_attr_z == None:
            with_attr_z = []
            for i in range(len(z)):
                z[i] = z[i].cpu()
                with_attr_z.append(torch.zeros_like(z[i][0]))

        for i in range(len(z)):
            z[i] = z[i].cpu()
            tmp_z = torch.index_select(z[i], 0, with_attr_idx)
            tmp_z = torch.sum(tmp_z, dim=0)
            with_attr_z[i] += tmp_z
            # tmp_z = tmp_z.unsqueeze(0)
            # with_attr_z = torch.cat((with_attr_z, tmp_z), dim=0)
            # with_attr_z.append(tmp_z)
            # tmp_z = torch.index_select(z[i], 0, wo_attr_idx)

        if iters == 2:
            break
    
    
    for i in range(len(with_attr_z)):
        with_attr_z[i] = with_attr_z[i] / with_attr_cnt
