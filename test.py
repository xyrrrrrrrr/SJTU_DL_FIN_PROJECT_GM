import jittor as jt
import numpy as np
import random
import pygmtools as pygm
from utils import *
from jittor import Var, models, nn, transform
from utils import delaunay_triangulation
from pygmtools.benchmark import Benchmark
from dataloader import image_dataset

vgg16_cnn = models.vgg16_bn(False)

obj_resize = (256,256)

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
    def __init__(self):
        super(GMNet, self).__init__()
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

model = GMNet()

benchmark = {
x: Benchmark(name='WillowObject',
                sets=x)
for x in ('train', 'test')}
train_dataset = image_dataset('train', benchmark['train'])
train_dataset.set_attrs(batch_size=1)
for i, data in enumerate(train_dataset):
        img1, img2, P1, P2, A1, A2, perm_mat = data
        perm_mat = jt.squeeze(perm_mat, 0)
        pred = model(img1, img2, P1[0], P2[0], A1[0], A2[0])
        # pred = jt.squeeze(pred, 0)
        print(perm_mat)
        print(pred)
        print(pygm.utils.permutation_loss(pred,perm_mat))
        print(img1.shape, img2.shape, P1.shape, P2.shape, A1.shape, A2.shape, perm_mat.shape)
        break