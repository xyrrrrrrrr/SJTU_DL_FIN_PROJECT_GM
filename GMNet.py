'''Rxy_Final_Job/GMNet.py

This module contains the GMNet class, which is used to create a
GMNet object. This object is used to create a GMNet object, which

@author: Rao Xiangyun
@version: 1.8
@date: 2022-12-06
'''
import jittor as jt 
from jittor import Var, models, nn
import pygmtools as pygm
from utils import *
pygm.BACKEND = 'jittor'
jt.flags.use_cuda = jt.has_cuda

obj_resize = (256, 256) # Default size of image (height, width).

class CNNNet(jt.nn.Module):
    def __init__(self, vgg16_module):
        super(CNNNet, self).__init__()
        # The naming of the layers follow ThinkMatch convention to load pretrained models.
        self.node_layers = jt.nn.Sequential(*[_ for _ in list(vgg16_module.features)[:31]])
        self.edge_layers = jt.nn.Sequential(*[_ for _ in list(vgg16_module.features)[31:38]])

    def execute(self, inp_img):
        feat_local = self.node_layers(inp_img)
        feat_global = self.edge_layers(feat_local)
        return feat_local, feat_global


class GMNet(jt.nn.Module):
    def __init__(self, vgg16_cnn, mode='pca_gm'):
        super(GMNet, self).__init__()
        self.mode = mode
        if mode == 'pca_gm':
            self.gm_net = pygm.utils.get_network(pygm.pca_gm, pretrain=False) # fetch the network object
        if mode == 'ipca_gm':
            self.gm_net = pygm.utils.get_network(pygm.ipca_gm, pretrain=False)
        if mode == 'cie':
            self.gm_net = pygm.utils.get_network(pygm.cie, pretrain=False)
        self.cnn = CNNNet(vgg16_cnn)

    def execute(self, img1, img2, kpts1, kpts2, A1, A2):
        # CNN feature extractor layers
        feat1_local, feat1_global = self.cnn(img1)
        feat2_local, feat2_global = self.cnn(img2)

        feat1_local = l2norm(feat1_local)
        feat1_global = l2norm(feat1_global)
        feat2_local = l2norm(feat2_local)
        feat2_global = l2norm(feat2_global)

        # upsample feature map
        feat1_local_upsample = jt.nn.interpolate(feat1_local, obj_resize, mode='bilinear')
        feat1_global_upsample = jt.nn.interpolate(feat1_global, obj_resize, mode='bilinear')
        feat2_local_upsample = jt.nn.interpolate(feat2_local, obj_resize, mode='bilinear')
        feat2_global_upsample = jt.nn.interpolate(feat2_global, obj_resize, mode='bilinear')
        feat1_upsample = jt.concat((feat1_local_upsample, feat1_global_upsample), dim=1)
        feat2_upsample = jt.concat((feat2_local_upsample, feat2_global_upsample), dim=1)

        # assign node features
        rounded_kpts1 = jt.round(kpts1).long()
        rounded_kpts2 = jt.round(kpts2).long()

        if len(feat1_upsample) == 1:
            node1 = feat1_upsample[0, :, rounded_kpts1[0][1], rounded_kpts1[0][0]].t()
            node2 = feat2_upsample[0, :, rounded_kpts2[0][1], rounded_kpts2[0][0]].t()
            A1 = A1[0]
            A2 = A2[0]
        else:
            N_ = rounded_kpts1[0][0].shape[0]
            node1 = jt.zeros((feat1_upsample.shape[0], N_, feat1_upsample.shape[1]))
            node2 = jt.zeros((feat1_upsample.shape[0], N_, feat1_upsample.shape[1]))
            for i in range(len(node1)):
                node1[i] = feat1_upsample[i, :, rounded_kpts1[i][1], rounded_kpts1[i][0]].t()
                node2[i] = feat2_upsample[i, :, rounded_kpts2[i][1], rounded_kpts2[i][0]].t()

        if self.mode == 'pca_gm':
            X = pygm.pca_gm(node1, node2, A1, A2, network=self.gm_net) # the network object is reused
        if self.mode == 'ipca_gm':
            X = pygm.ipca_gm(node1, node2, A1, A2, network=self.gm_net)
        if self.mode == 'cie':
            kpts1_dis = (kpts1.unsqueeze(1) - kpts1.unsqueeze(2))
            kpts1_dis = jt.norm(kpts1_dis, p=2, dim=0).detach()
            kpts2_dis = (kpts2.unsqueeze(1) - kpts2.unsqueeze(2))
            kpts2_dis = jt.norm(kpts2_dis, p=2, dim=0).detach()

            Q1 = jt.exp(-kpts1_dis / obj_resize[0]).unsqueeze(-1).float32()
            Q2 = jt.exp(-kpts2_dis / obj_resize[0]).unsqueeze(-1).float32()
            X = pygm.cie(node1, node2, A1, A2, Q1, Q2, network=self.gm_net)
        return X

class GMNet2(jt.nn.Module):
    def __init__(self, vgg16_cnn):
        super(GMNet2, self).__init__()
        self.gm_net = pygm.utils.get_network(pygm.pca_gm, pretrain=False) # fetch the network object
        self.cnn = CNNNet(vgg16_cnn)

    def execute(self, img1, img2, kpts1, kpts2, A1, A2):
        # CNN feature extractor layers
        feat1_local, feat1_global = self.cnn(img1)
        feat2_local, feat2_global = self.cnn(img2)
        feat1_local = l2norm(feat1_local)
        feat1_global = l2norm(feat1_global)
        feat2_local = l2norm(feat2_local)
        feat2_global = l2norm(feat2_global)

        # upsample feature map
        feat1_local_upsample = jt.nn.interpolate(feat1_local, obj_resize, mode='bilinear')
        feat1_global_upsample = jt.nn.interpolate(feat1_global, obj_resize, mode='bilinear')
        feat2_local_upsample = jt.nn.interpolate(feat2_local, obj_resize, mode='bilinear')
        feat2_global_upsample = jt.nn.interpolate(feat2_global, obj_resize, mode='bilinear')
        feat1_upsample = jt.concat((feat1_local_upsample, feat1_global_upsample), dim=1)
        feat2_upsample = jt.concat((feat2_local_upsample, feat2_global_upsample), dim=1)

        # assign node features
        rounded_kpts1 = jt.round(kpts1).long()
        rounded_kpts2 = jt.round(kpts2).long()
        node1 = feat1_upsample[0, :, rounded_kpts1[0], rounded_kpts1[1]].t()  # shape: NxC
        node2 = feat2_upsample[0, :, rounded_kpts2[0], rounded_kpts2[1]].t()  # shape: NxC

        # PCA-GM matching layers
        X = pygm.pca_gm(node1, node2, A1, A2, network=self.gm_net) # the network object is reused
        return X

