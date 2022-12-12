'''Rxy_Final_Job/dataloader.py

This module contains the dataset of the project.

@Author: Rxy
@version: 1.6
@date: 2022-12-08
'''
import jittor as jt
import numpy as np
import random
import pygmtools
from jittor import Var, models, nn, transform
from utils import delaunay_triangulation
from pygmtools.benchmark import Benchmark
from GMNet import GMNet, GMNet2


class image_dataset(jt.dataset.dataset.Dataset):
    def __init__(self, name, bm, cls=None, length=None):
        super().__init__()
        self.bm = bm
        self.obj_resize = self.bm.obj_resize
        self.test = True if self.bm.sets == 'test' else False
        self.cls = cls
        self.classes = self.bm.classes
        if length != None:
            self.length = length
        else:
            self.id_combination, self.length = self.bm.get_id_combination(self.cls)
        self.ids = []
        for i in range(len(self.id_combination)):
            self.ids.extend(self.id_combination[i])
        self.set_attrs(total_len=self.length)
        self.length_list = []
        for cls_ in self.classes:
            cls_length = self.bm.compute_length(cls_)
            self.length_list.append(cls_length)
        
    def __getitem__(self, index):
        if self.cls == None:
            if not self.test:
                data_list, perm_mat_dict, ids = self.bm.get_data(list(self.ids[index]))
            else:
                data_list, ids =  self.bm.get_data(list(self.ids[index]),test=True)
        else:
            if not self.test:
                data_list, perm_mat_dict, ids = self.bm.rand_get_data(self.cls)
            else:
                data_list, ids = self.bm.rand_get_data(self.cls, test=True)
        #### process the image ####            
        img1, img2 = data_list[0]['img'], data_list[1]['img']
        trans = transform.Compose([
                    transform.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        img1 = np.array(img1, dtype=np.float32)
        img2 = np.array(img2, dtype=np.float32)
        img1 = img1.transpose(2, 0, 1)/256.0
        img2 = img2.transpose(2, 0, 1)/256.0
        img1, img2 = [trans(img) for img in [img1, img2]]
        img1 = jt.array(img1)
        img2 = jt.array(img2)
        #### process the keypoints ####
        P1 = jt.float32([(kp['x'], kp['y']) for kp in data_list[0]['kpts']])
        P2 = jt.float32([(kp['x'], kp['y']) for kp in data_list[1]['kpts']])
        P1 = jt.transpose(P1)
        P2 = jt.transpose(P2)
        #### Process the A ####
        A1 = delaunay_triangulation(P1)
        A2 = delaunay_triangulation(P2)

        if not self.test:
            perm_mat = perm_mat_dict[(0, 1)].toarray()
            perm_mat = jt.array(perm_mat)
            return img1, img2, P1, P2, A1, A2, perm_mat
        else:    
            cur_cls = data_list[0]['cls']
            return img1, img2, P1, P2, A1, A2, ids, cur_cls

if __name__ == '__main__':
    vgg16 = models.vgg16_bn(False)
    gmnet = GMNet(vgg16)
    benchmark = {
    x: Benchmark(name='WillowObject',
                    sets=x)
    for x in ('train', 'test')}
    train_dataset = image_dataset('train', benchmark['train'])
    train_dataset.set_attrs(batch_size=1, shuffle=False)
    for i, data in enumerate(train_dataset):
        img1, img2, P1, P2, A1, A2, perm_mat = data
        perm_mat = jt.squeeze(perm_mat, 0)
        pred = gmnet(img1, img2, P1, P2, A1, A2)
        pred = jt.squeeze(pred, 0)
        print(perm_mat)
        print(pred)
        print(pygmtools.utils.permutation_loss(pred,perm_mat))
        print(img1.shape, img2.shape, P1.shape, P2.shape, A1.shape, A2.shape, perm_mat.shape)
        break