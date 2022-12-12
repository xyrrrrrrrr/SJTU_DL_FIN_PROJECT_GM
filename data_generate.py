'''Rxy_Final_Job/data_generate.py

This module contains the data generating process of the project.
You can run this module to generate the data with parameters
you want.

Example: python data_generate.py --obj_resize=(256,256) --type='WillowObject'

@author: Rao Xiangyun
@version: 1.0
@date: 2022-11-29
'''
import pygmtools
import argparse

parser = argparse.ArgumentParser(description='Generate the data')
parser.add_argument('--obj_resize', type=tuple, default=(256, 256), help='resize the object')
parser.add_argument('--type', type=str, default='WillowObject', help='type of dataset')
args = parser.parse_args()

obj_resize = args.obj_resize

def main():
    if args.type == 'WillowObject':
        train_obj = pygmtools.dataset.WillowObject('train', obj_resize)
        test_obj = pygmtools.dataset.WillowObject('test', obj_resize)
    elif args.type == 'SPair71k':
        train_obj = pygmtools.dataset.SPair71k('train', obj_resize)
        test_obj = pygmtools.dataset.SPair71k('test', obj_resize)
    elif args.type == 'PascalVOC':
        train_obj = pygmtools.dataset.PascalVOC('train', obj_resize)
        test_obj = pygmtools.dataset.PascalVOC('test', obj_resize)
    elif args.type == 'CUB2011':
        train_obj = pygmtools.dataset.CUB2011('train', obj_resize)
        test_obj = pygmtools.dataset.CUB2011('test', obj_resize)
    elif args.type == 'IMC_PT_SparseGM':
        train_obj = pygmtools.dataset.IMC_PT_SparseGM('train', obj_resize)
        test_obj = pygmtools.dataset.IMC_PT_SparseGM('test', obj_resize)

if __name__ == '__main__':
    main()