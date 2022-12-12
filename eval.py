'''Rxy_Final_Job/eval.py

This module contains the evaluation process of the project.
You can run this module to evaluate the model with parameters
you want.

Example: python eval.py --vggmodel='vgg16.pkl' --gmnetmodel='gmnet.pkl' --mode='pca-gm'

@author: Rao Xiangyun
@version: 1.1
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

parser = argparse.ArgumentParser(description='Eval the model')
parser.add_argument('--vggmodel', type=str, default='./model/vgg16_100_0.001.pkl', help='vgg16 model')
parser.add_argument('--gmnetmodel', type=str, default='./model/gmnet_100_0.001.pkl', help='gmnet model')
parser.add_argument('--mode', type=str, default='pca_gm', help='mode,pca_gm/ipca_gm/cie')
parser.add_argument('--type', type=str, default='WillowObject', help='type of dataset')
parser.add_argument('--obj_resize', type=tuple, default=(256, 256), help='resize the object')
parser.add_argument('--classes', type=str, default='all', help='class of dataset')
args = parser.parse_args()

# Define the model'
vgg16_state_dict = jt.load(args.vggmodel)
gmnet_state_dict = jt.load(args.gmnetmodel)
vgg16 = models.vgg16_bn(False)
vgg16.load_state_dict(vgg16_state_dict)
gmnet = GMNet(vgg16,args.mode)
gmnet.load_state_dict(gmnet_state_dict)

# Create the dataset
obj_resize = args.obj_resize
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
    train_dataset = image_dataset('WillowObject', benchmark['test'])
else:
    train_dataset = image_dataset('WillowObject', benchmark['test'], cls=args.classes)
train_dataset.set_attrs(batch_size=1, shuffle=True)

# Define the eval process
vgg16.eval()
gmnet.eval()
prediction = []
classes = ['Duck','Car','Face','Motorbike','Winebottle']
for i, (img1, img2, P1, P2, A1, A2, ids, cur_cls) in enumerate(train_dataset):
        y_pred = gmnet(img1, img2, P1, P2, A1, A2)
        # 计算perm_mat
        perm_mat = pygmtools.hungarian(y_pred)
        # generate prediction dict
        current_pred = {}
        current_pred['ids'] = [ids[0][0], ids[1][0]]
        current_pred['cls'] = cur_cls[0]
        current_pred['perm_mat'] = perm_mat.numpy()
        prediction.append(current_pred)
        print('\rStep: {}/{}'.format(i+1, len(train_dataset), end=''))
if args.classes == 'all':
    benchmark['test'].eval(prediction, classes, verbose=True)
else:
    benchmark['test'].eval_cls(prediction, args.classes, verbose=True)
# benchmark['test'].eval_cls(prediction, cls='Car')
# benchmark['test'].eval_cls(prediction, cls='Duck')
# benchmark['test'].eval_cls(prediction, cls='Face')
# benchmark['test'].eval_cls(prediction, cls='Motorbike')
# benchmark['test'].eval_cls(prediction, cls='Winebottle')

