import argparse
import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, dataset
from torchvision import datasets
from torch.autograd import Variable

from model import Generator, Discriminator

import torch
from torch.utils.tensorboard import SummaryWriter
import random

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=9999999999, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
parser.add_argument("--gpu", type=str, default='cuda:0', help='Choose which gpu to run a training. If you have a single GPU, ignore this flag. If you have multiple gpus, cuda:0 will let a model to run on gpu 0, cuda:1 will let a model to run on gpu1, and so on.')
parser.add_argument("--mode", type=int, default=0, choices=[0, 1, 2], help='0 for no augmentation, 1 for simple augmentation, 2 for probabilistic augmentation')
parser.add_argument("--probability", type=float, default=0.2, help='Probability of augmentation. Only effectivate when mode flag is set to 2')
parser.add_argument("--n_training_images", type=int, default=1000, help='Number of images to use for training')
opt = parser.parse_args()

# Loss function
adversarial_loss = torch.nn.BCELoss()

device = opt.gpu if torch.cuda.is_available() else 'cpu'

os.makedirs(f"images/{device}", exist_ok=True)
writer = SummaryWriter()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def validation(generator, discriminator, dataloader, Tensor, batches_done):
    g_losses = []
    d_losses = []
    for i, (imgs, _) in tqdm(enumerate(dataloader), desc="Validation"):
        
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False).to(device)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor)).to(device)
        
        #--------------------
        # Validate Generator
        #--------------------

        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))).to(device)
        
        with torch.no_grad():
            gen_imgs = generator(z)
        
        g_losses.append(adversarial_loss(discriminator(gen_imgs), valid))

        #--------------------
        # Validate Discriminator
        #--------------------

        with torch.no_grad():
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        
        d_losses.append((real_loss + fake_loss) / 2)

    
    average_g_loss = sum(g_losses) / len(g_losses)
    average_d_loss = sum(d_losses) / len(d_losses)
    print("Validation Generator Loss : {}".format(average_g_loss))
    print("Validation Discriminator Loss : {}".format(average_d_loss))
    writer.add_scalar('Validation/Generator Loss', average_g_loss, batches_done)
    writer.add_scalar('Validation/Discriminator Loss', average_d_loss, batches_done)
    save_image(gen_imgs.data[:25], f"images/{opt.gpu}/val_{batches_done}.png", nrow=5, normalize=True)
        
def train(generator, discriminator, optimizer_G, optimizer_D, train_dataloader, val_dataloader, Tensor):
    
    pbar = tqdm(range(opt.n_epochs))
    for epoch in pbar:
        for i, (imgs, _) in enumerate(train_dataloader):
            identity_mapping = [lambda img: img]
            full_augmentations = [
                lambda img: img,
                transforms.RandomAffine(degrees=15, shear=15),
                transforms.ColorJitter()
            ]
            orig_dim = imgs.shape[0]

            if opt.mode == 1: # Simple Augmentation
                # TODO : uniformly select augmentations
                augmentations = full_augmentations
            elif opt.mode == 2: # Stochastic Discriminator Augmentation
                # TODO : stochastically select augmentations (select augmentation probabilistically)
                if np.random.choice([True, False], p=[opt.probability, 1 - opt.probability]):
                    augmentations = full_augmentations
                else:
                    augmentations = identity_mapping

            if opt.mode == 1 or opt.mode == 2: 
                # TODO : Apply selected augmentations to a image
                imgs = torch.cat([augmentation(imgs) for augmentation in augmentations], dim=0)
                print(f"imgs.shape : {imgs.shape}")

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False).to(device)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor)).to(device)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (orig_dim, opt.latent_dim)))).to(device)

            # Generate a batch of images
            gen_imgs = generator(z)
            orig_gen_imgs = gen_imgs.detach()

            if opt.mode == 2: # Stochastic Discriminator Augmentation
                # TODO : Apply selected augmentations to a generaeted image
                gen_imgs = torch.cat([augmentation(gen_imgs) for augmentation in augmentations], dim=0)
                print(f"gen_imgs.shape :{gen_imgs.shape}")

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()


            batches_done = epoch * len(train_dataloader) + i
            
            if batches_done % opt.sample_interval == 0:
                writer.add_scalar('Train/Generator Loss', g_loss, batches_done)
                writer.add_scalar('Train/Discriminator Loss', d_loss, batches_done)
                save_image(orig_gen_imgs.data[:25], f"images/{opt.gpu}/{batches_done}.png", nrow=5, normalize=True)
                validation(generator, discriminator, val_dataloader, Tensor, batches_done)
        pbar.set_description(f"G Loss : {g_loss.item()} D Loss : {d_loss.item()}")

if __name__ == '__main__':

    # Initialize generator and discriminator
    generator = Generator(opt.latent_dim, opt.img_size, opt.channels)
    discriminator = Discriminator(opt.img_size, opt.channels)

    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Configure dataloader
    os.makedirs("data/mnist", exist_ok=True)

    transform = transforms.Compose(
        [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )

    # Get only 1000 training images and labels
    t_dataset = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
    indices = torch.randperm(len(t_dataset))[:opt.n_training_images]

    train_dataset = torch.utils.data.Subset(t_dataset, indices)

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size,shuffle=True)
    
    val_dataloader = DataLoader(
        datasets.MNIST(
            "data/mnist",
            train=False,
            download=True,
            transform = transform,
        ),
        batch_size = opt.batch_size,
        shuffle=False
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if device != 'cpu' else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    train(generator, discriminator, optimizer_G, optimizer_D, train_dataloader, val_dataloader, Tensor)
