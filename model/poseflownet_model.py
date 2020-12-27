import torch
from model.base_model import BaseModel
from model.networks import base_function, external_function
import model.networks as network
from util import task, util,pose_utils
import itertools
import data as Dataset
import numpy as np
from itertools import islice
import random
import os


class PoseFlowNet(BaseModel):
    def name(self):
        return "Pre-train flow estimator for human pose image generation"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--netG', type=str, default='poseflownet', help='The name of net Generator')
        parser.add_argument('--init_type', type=str, default='orthogonal', help='Initial type')

        parser.add_argument('--attn_layer', action=util.StoreList, metavar="VAL1,VAL2...", help="The number layers away from output layer") 
        parser.add_argument('--kernel_size', action=util.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...", help="Kernel Size of Local Attention Block")

        parser.add_argument('--lambda_correct', type=float, default=5.0, help='weight for the Sampling Correctness loss')
        parser.add_argument('--lambda_regularization', type=float, default=0.01, help='weight for Regularization loss')
        parser.add_argument('--use_spect_g', action='store_false')
        parser.set_defaults(use_spect_g=False)
        return parser


    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.keys = ['head','body','leg']
        self.mask_id = {'head':[1,2,4,13],'body':[3,5,6,7,10,11,14,15],'leg':[8,9,12,16,17,18,19]}
        self.channel_id = {'head':[0,14,15,16,17],'body':[1,2,3,4,5,6,7,8,11],'leg':[8,9,10,11,12,13]}
        # self.target_mask_id = {'backgrand':[0],'hair':[1,2],'head':[4,13],'arm':[14,15],'body':[3,5,6,7,10,11],'leg':[16,17],'pants':[9,12],'shoes':[8,18,19]}
        self.target_mask_id = {'backgrand':[0],'head':[1,2,4,13],'body':[3,5,6,7,10,11,14,15],'leg':[8,9,12,16,17,18,19]}
        self.GPU = torch.device('cuda:0')

        self.loss_names = ['correctness', 'regularization', 'layout']
        self.visual_names = ['input_P1','input_P2', 'warp', 'flow_fields',
                            'masks','input_BP1', 'input_BP2', 'layout', 'target_layout']
        self.model_names = {'G':['flow_net','layout']}

        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids)>0 \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if len(self.gpu_ids)>0 \
            else torch.ByteTensor

        self.net_G = network.define_g(opt, structure_nc=opt.structure_nc, ngf=32, img_f=256, 
                                       layers=5, norm='instance', activation='LeakyReLU', 
                                       attn_layer=self.opt.attn_layer, use_spect=opt.use_spect_g,
                                       )
        self.flow2color = util.flow2color()
        self.layout2color = util.layout2color

        self.l1loss = torch.nn.L1Loss()

        if self.isTrain:
            # define the loss functions
            self.Correctness = external_function.PerceptualCorrectness().to(opt.device)
            self.Regularization = external_function.MultiAffineRegularizationLoss(kz_dic=opt.kernel_size).to(opt.device)
            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_G.parameters())),
                                                lr=opt.lr, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_G)
        # load the pretrained model and schedulers
        self.setup(opt)


    def set_input(self,input):
        # move to GPU and change data types
        random.shuffle(self.keys)
        
        if len(self.gpu_ids) > 0:
            self.input_fullP1 = input['P1'].cuda()
            self.input_fullP2 = input['P2'].cuda()
            input_P1mask = input['P1masks'].cuda()
            input_P2mask = input['P2masks'].cuda()

        res_mask = []
        for key in self.target_mask_id.keys():
            tmpmask = []
            for k in self.target_mask_id[key]:
                tmpmask.append(torch.where(input_P2mask==k,torch.ones_like(input_P2mask),torch.zeros_like(input_P2mask)))
            tmpmask = torch.stack(tmpmask)
            res_mask.append(torch.sum(tmpmask,axis=0))
        self.target_layout = torch.stack(res_mask,1)
        self.target_layout = self.target_layout.float()

        input_P1mask,_ = pose_utils.obtain_mask(input_P1mask,self.mask_id,self.keys)
        input_P2mask,_ = pose_utils.obtain_mask(input_P2mask,self.mask_id,self.keys)
        self.input_P1 = self.input_fullP1.repeat(3,1,1,1)*input_P1mask
        self.input_P2 = self.input_fullP2.repeat(3,1,1,1)*input_P2mask
        self.input_BP1 = pose_utils.cords_to_map(input['BP1'],input['P1masks'],self.channel_id,self.keys,self.GPU,self.opt,input['affine'])
        self.input_BP2 = pose_utils.cords_to_map(input['BP2'],input['P2masks'],self.channel_id,self.keys,self.GPU,self.opt)


        self.image_paths=[]
        for i in range(self.opt.batchSize):
            self.image_paths.append(os.path.splitext(input['P1_path'][i])[0] + '_2_' + input['P2_path'][i])


    def forward(self):
        """Run forward processing to get the inputs"""
        self.flow_fields, self.masks, self.layout = self.net_G(self.input_P1, self.input_BP1, self.input_BP2)
        self.warp  = self.visi(self.flow_fields[0][-1])

    def visi(self, flow_field):
        [b,_,h,w] = flow_field.size()

        source_copy = torch.nn.functional.interpolate(self.input_P1[:self.opt.batchSize], (h,w))

        x = torch.arange(w).view(1, -1).expand(h, -1).float()
        y = torch.arange(h).view(-1, 1).expand(-1, w).float()
        x = 2*x/(w-1)-1
        y = 2*y/(h-1)-1
        grid = torch.stack([x,y], dim=0).float().cuda()
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        flow_x = (2*flow_field[:,0,:,:]/(w-1)).view(b,1,h,w)
        flow_y = (2*flow_field[:,1,:,:]/(h-1)).view(b,1,h,w)
        flow = torch.cat((flow_x,flow_y), 1)

        grid = (grid+flow).permute(0, 2, 3, 1)
        warp = torch.nn.functional.grid_sample(source_copy, grid, align_corners=True)
        return  warp


    def backward_G(self):
        """Calculate training loss for the generator"""
        loss_correctness = self.Correctness(self.input_P2[:self.opt.batchSize], self.input_P1[:self.opt.batchSize], self.flow_fields[0], self.opt.attn_layer)+\
            self.Correctness(self.input_P2[self.opt.batchSize:2*self.opt.batchSize], self.input_P1[self.opt.batchSize:2*self.opt.batchSize], self.flow_fields[1], self.opt.attn_layer)+\
                self.Correctness(self.input_P2[2*self.opt.batchSize:], self.input_P1[2*self.opt.batchSize:], self.flow_fields[2], self.opt.attn_layer)
        self.loss_correctness = loss_correctness * self.opt.lambda_correct

        loss_regularization = self.Regularization(self.flow_fields[0])+self.Regularization(self.flow_fields[1])+self.Regularization(self.flow_fields[2])
        self.loss_regularization = loss_regularization * self.opt.lambda_regularization

        loss_layout = self.l1loss(self.layout,self.target_layout)
        self.loss_layout = loss_layout*1.0

        total_loss = 0
        for name in self.loss_names:
            total_loss += getattr(self, "loss_" + name)
        total_loss.backward()


    def optimize_parameters(self):
        """update netowrk weights"""
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
