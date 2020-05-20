import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import pandas as pd
from util import pose_utils
import numpy as np
import torch

class FashionChangeClothDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
        if is_train:
            parser.set_defaults(load_size=256.)
        else:
            parser.set_defaults(load_size=256.)
        parser.set_defaults(old_size=(256., 256.))
        parser.set_defaults(structure_nc=18)
        parser.set_defaults(image_nc=3)
        parser.set_defaults(display_winsize=256)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'test' if opt.phase == 'val' else opt.phase
        imgLst = os.path.join(root, '%s.lst' % phase)
        imgnameLst = self.init_categories(imgLst)
        
        image_dir = os.path.join(root, '%s' % phase)
        mask_dir = os.path.join(root, '%s_mask' % phase)
        bonesLst = os.path.join(root, 'fasion-annotation-%s.csv' % phase)
        return image_dir, mask_dir, bonesLst, imgnameLst

    def init_categories(self, imgLst):
        lst = []
        with open(imgLst,'r') as f:
            for line in f:
                line = line.strip()
                lst.append(line)

    
        if self.opt.dataset_size == 0:
            size = len(lst)
        else:
            size = self.opt.dataset_size

        pairs = []
        print('Loading data pairs ...')
        for i in range(size):
            for j in range(size):
            pair = [lst[i], lst[j]]
            pairs.append(pair)

        print('Loading data pairs finished ...{}pairs'.format(len(pairs)))  
        return pairs    


    def name(self):
        return "FashionChangeClothDataset"