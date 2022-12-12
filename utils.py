'''Rxy_Final_Job/utils.py

This module contains the utils functions used in the project.

@author: Rao Xiangyun
@version: 1.0
@date: 2022-11-29

'''
import jittor as jt 
from jittor import Var, models, nn
import scipy.spatial as spa
import itertools

def local_response_norm(input: Var, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0) -> Var:
    '''
    This function is used to implement the local response normalization.
    Parameters:
        input(Var): the input feature map
        size(int): the size of the local response normalization
        alpha(float): the alpha of the local response normalization
        beta(float): the beta of the local response normalization
        k(float): the k of the local response normalization
    Returns:
        output(Var): the output feature map
    '''
    dim = input.ndim
    assert dim >= 3

    if input.numel() == 0:
        return input

    div = input.multiply(input).unsqueeze(1)
    if dim == 3:
        div = nn.pad(div, (0, 0, size // 2, (size - 1) // 2))
        div = nn.avg_pool2d(div, (size, 1), stride=1).squeeze(1)
    else:
        sizes = input.size()
        div = div.view(sizes[0], 1, sizes[1], sizes[2], -1)
        div = nn.pad(div, (0, 0, 0, 0, size // 2, (size - 1) // 2))
        div = nn.AvgPool3d((size, 1, 1), stride=1)(div).squeeze(1)
        div = div.view(sizes)
    div = div.multiply(alpha).add(k).pow(beta)
    return input / div

def l2norm(node_feat):
    '''
    Get node_feat that has been l2 normalized.
        Parameters:
            node_feat(Var): the node feature matrix
        Returns:
            node_feat(Var): the node feature matrix that has been l2 normalized
    '''
    return local_response_norm(
        node_feat, node_feat.shape[1] * 2, alpha=node_feat.shape[1] * 2, beta=0.5, k=0)

def delaunay_triangulation(kpt):
    '''
    Get the delaunay triangulation of the keypoints.
        Parameters:
            kpt(Var): the keypoints
        Returns:
            tri(Var): the delaunay triangulation of the keypoints
    '''
    d = spa.Delaunay(kpt.numpy().transpose())
    A = jt.zeros((len(kpt[0]), len(kpt[0])))
    for simplex in d.simplices:
        for pair in itertools.permutations(simplex, 2):
            A[pair] = 1
    return A

