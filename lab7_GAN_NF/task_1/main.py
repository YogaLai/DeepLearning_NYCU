from cINN import test
from dataloader import ICLEVRLoader, get_iCLEVR_data
import torch
import argparse
from model import *
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from evaluator import evaluation_model
import numpy as np
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=150)
parser.add_argument('--exp_name', type=str, default='')
args = parser.parse_args()

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64
    # condition dimension
    nc = 200
    # Size of z latent vector (i.e. size of generator input)
    nz = 100
    # Size of feature maps in generator
    ngf = 64
    # Size of feature maps in discriminator
    ndf = 64
    # Number of training epochs
    num_epochs = args.num_epochs
    # Learning rate for optimizers
    lr = 0.0002
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5
    batch_size = args.batch_size

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    netG = Generator(nc, nz, ngf).to(device)
    netD = Discriminator((64,64,3), ndf).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)
    evaluation_model = evaluation_model()

    criterion = nn.BCELoss()
    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator

    real_label = 1.
    fake_label = 0.

    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.SGD(netD.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    writer = SummaryWriter('logs/' + args.exp_name)
    train_loader = torch.utils.data.DataLoader(ICLEVRLoader('./'), batch_size = batch_size, shuffle = True)
    # test_loader = torch.utils.data.DataLoader(ICLEVRLoader('./', mode='test'), batch_size = 2, shuffle = True)
    test_condition = get_iCLEVR_data('./', 'test')[1]
    fixed_noise = torch.randn(len(test_condition), nz, 1, 1, device=device)
    iters = 0
    score_list = []

    for epoch in range(num_epochs):
        total_lossD = 0 
        total_lossG = 0 
        netG.train()
        netD.train()
        for i, data in enumerate(train_loader):
            condition = data[1].to(device)
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            real_data = data[0].to(device)
            b_size = real_data.shape[0]
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_data, condition).view(-1)
            # Calculate loss on all-real batch
            lossD_real = criterion(output, label)
            lossD_real.backward()
            # D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake_data = netG(noise, condition)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake_data.detach(), condition).view(-1)
            lossD_fake = criterion(output, label)
            lossD_fake.backward()
            # D_G_z1 = output.mean().item()
            lossD = lossD_real + lossD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            label.fill_(real_label)  # fake labels are real for generator cost
            for i in range(4):
                netG.zero_grad()
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake_data = netG(noise, condition)
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake_data, condition).view(-1)
                lossG = criterion(output, label)
                lossG.backward()
                # D_G_z2 = output.mean().item()
                optimizerG.step()

            writer.add_scalar('Train/D_loss', lossD.item(), iters)
            writer.add_scalar('Train/G_loss', lossG.item(), iters)
            iters += 1
            total_lossD += lossD.item()
            total_lossG += lossG.item()

        writer.add_scalar('Train/total_D_loss', total_lossD/len(train_loader), epoch)
        writer.add_scalar('Train/total_G_loss', total_lossG/len(train_loader), epoch)
        print('-----------------------')
        print('Epoch: ', epoch)
        print('Total D loss: ', total_lossD/len(train_loader))
        print('Total G loss: ', total_lossG/len(train_loader))

        # evaluate
        netG.eval()
        with torch.no_grad():
            test_condition = torch.tensor(test_condition, dtype=torch.float32)
            test_condition = test_condition.to(device)
            generate_img = netG(fixed_noise, test_condition)
        images_concat = torchvision.utils.make_grid(generate_img, nrow=8, normalize=True)
        torchvision.utils.save_image(images_concat, 'visualization/cGAN/epoch_{}.png'.format(epoch))
        score = evaluation_model.eval(generate_img, test_condition)
        score_list.append(score)
        print('Score: ', score)
        print('-----------------------\n')

        # save model
        savefilename = 'savemodel/checkpoint_' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'generator_state_dict': netG.state_dict(),
            'discriminator_state_dict': netD.state_dict(),
        }, savefilename)

    score_list = np.asarray(score_list)
    print('Best epoch: %d\nBest score: %f' % (max(score_list), np.argmax(score_list)))