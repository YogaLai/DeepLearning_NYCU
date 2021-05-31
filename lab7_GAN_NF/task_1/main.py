from dataset import ICLEVRLoader
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=24)
args = parser.parse_args()

if __name__ == '__main__':
    train_loader = torch.utils.data.DataLoader(ICLEVRLoader('./'), batch_size = args.batch_size, shuffle = True)
    for data in train_loader:
        print(data)