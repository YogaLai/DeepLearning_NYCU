from model import Glow
import torch
import argparse
from PIL import Image
from torchvision import transforms, utils

def preprocess(image, transform, n_bins):
    image = transform(image)
    image = image.unsqueeze(0)
    image = image * 255

    if args.n_bits < 8:
        image = torch.floor(image / 2 ** (8 - args.n_bits))

    image = image / n_bins - 0.5

    return image

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
    args = parser.parse_args()

    model_single = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    model = torch.nn.DataParallel(model_single)
    ckt = torch.load('savemodel/checkpoint_10.tar')
    model.load_state_dict(ckt['net'])
    # model = model.to(device)

    path = '/media/jojorge/NTFS/yoga/109_2/DeepLearning/DeepLearning_NYCU/lab7_GAN_NF/task_2/CelebA-HQ-img/'
    transform = transforms.Compose(
        [
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
        ]
    )
    n_bins = 2.0 ** args.n_bits
    img1 = Image.open(path + '0.jpg')
    img2 = Image.open(path + '1.jpg')
    img1 = preprocess(img1, transform, n_bins)
    img2 = preprocess(img2, transform, n_bins)
    images = [img1, img2]
    z_list = []
    for image in images:
        log_p, logdet, z = model(image + torch.rand_like(image) / n_bins)
        z_list.append(z)

    z1 = z_list[0]
    z2 = z_list[1]
    all_rate = [0.1, 0.3, 0.5, 0.7, 0.9]
    img_concat = img1
    for rate in all_rate:
        interpolate_z = []
        for i in range(len(z1)):
            interpolate_z.append(torch.lerp(z1[i], z2[i], rate))
        with torch.no_grad():
            reconstruct_img = model_single.reverse(interpolate_z).cpu().data
        img_concat = torch.cat((img_concat, reconstruct_img), dim=0)

    img_concat = torch.cat((img_concat, img2), dim=0)
    utils.save_image(
        img_concat,
        "interpolate.png",
        normalize=True,
        nrow=7,
        range=(-0.5, 0.5),
    )