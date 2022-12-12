'''Rxy_Final_Job/train.py

This module contains the training process of the project. 
You can run this module to train the model with parameters
you want.

Example: python train.py --batch_size 32 --lr 0.001 --epoch 100 --mode='pca-gm' --class='all'

@author: Rao Xiangyun
@version: 1.3
@date: 2022-12-08
'''
import jittor as jt
from jittor import Var, models, nn
import pygmtools
from pygmtools.benchmark import Benchmark
from PIL import Image
import scipy.io as sio
import numpy as np
import argparse
from GMNet import GMNet
from dataloader import image_dataset
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
parser.add_argument('--lr_scheduling', type=bool, default=False, help='whether to use lr scheduling')
parser.add_argument('--epochs', type=int, default=300, help='epochs')
parser.add_argument('--epoch_iters', type=int, default=1, help='epoch iters')
parser.add_argument('--mode', type=str, default='pca_gm', help='mode,pca_gm/ipca_gm/cie')
parser.add_argument('--type', type=str, default='WillowObject', help='type of dataset')
parser.add_argument('--obj_resize', type=tuple, default=(256, 256), help='resize the object')
parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
parser.add_argument('--classes', type=str, default='all', help='class of dataset')
args = parser.parse_args()

# Define the model'
vgg16 = models.vgg16_bn(False)
gmnet = GMNet(vgg16,args.mode)

# Define super parameters
batch_size = args.batch_size
lr = args.lr
epochs = args.epochs
obj_resize = args.obj_resize

# Define the optimizer
if args.optimizer == 'Adam':
    optimizer = nn.Adam(gmnet.parameters(), lr=lr)
else:
    optimizer = nn.SGD(gmnet.parameters(), lr=lr, momentum=0.9)

# Define the loss function
# loss_func = pygmtools.utils.permutation_loss()

# Define the Benchmark
ds_dict = {}

if args.type == 'WillowObject':
   name = 'WillowObject'
elif args.type == 'SPair71k':
    name = 'SPair71k'
elif args.type == 'PascalVOC':
    name = 'PascalVOC'
elif args.type == 'CUB2011':
    name = 'CUB2011'
elif args.type == 'IMC_PT_SparseGM':
    name = 'IMC_PT_SparseGM'

benchmark = {
    x: Benchmark(name=name,
                    sets=x,
                    problem='2GM',
                    obj_resize=obj_resize,
                    filter='intersection',
                    **ds_dict)
    for x in ('train', 'test')}

if args.classes == 'all':
    train_dataset = image_dataset('WillowObject', benchmark['train'])
else:
    train_dataset = image_dataset('WillowObject', benchmark['train'], cls=args.classes)
train_dataset.set_attrs(batch_size=batch_size, shuffle=False)

# Define the training process
loss_all = []
for epoch in range(epochs):
    if (epoch + 1) % 10 == 0 and args.lr_scheduling:
        lr = lr / 10
        if args.optimizer == 'Adam':
            optimizer = nn.Adam(gmnet.parameters(), lr=lr)
        else:
            optimizer = nn.SGD(gmnet.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for j in range(args.epoch_iters):
        losses = []
        for i, (img1, img2, P1, P2, A1, A2, perm_mat) in enumerate(train_dataset):
            y_pred = gmnet(img1, img2, P1, P2, A1, A2)
            y = perm_mat
            if batch_size == 1:
                y = y[0]
            loss = pygmtools.utils.permutation_loss(y_pred, y)
            optimizer.step(loss)
            # fresh the printing
            print('\rEpoch: {}/{}, Step: {}/{}, Loss: {:.4f}, current_lr:{}'.format(epoch + 1, epochs, j * len(train_dataset) + i + 1, args.epoch_iters * len(train_dataset), loss.item(), lr), end='')
            # plot the loss
            if (i+1) % 10 == 0:
                losses.append(loss.item())
        print('\r')
        loss_all.append(min(losses))

# Save the model
jt.save(gmnet.state_dict(), './model/gmnet_{}_{}_{}_{}_{}.pkl'.format(args.classes, args.mode, args.type, args.batch_size, args.lr))
jt.save(vgg16.state_dict(), './model/vgg16_{}_{}_{}_{}_{}.pkl'.format(args.classes, args.mode, args.type, args.batch_size, args.lr))

# Plot the loss
plt.plot(loss_all)

# Save the loss with parameter
plt.savefig('./loss/loss_{}_{}_{}_{}.png'.format(args.classes, args.mode, batch_size, lr))
