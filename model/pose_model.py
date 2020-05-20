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

class Pose(BaseModel):
    """
       Deep Spatial Transformation For Pose Based Image Generation
    """
    def name(self):
        return "Human Pose-based Image Generation Task"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--attn_layer', action=util.StoreList, metavar="VAL1,VAL2...", help="The number layers away from output layer") 
        parser.add_argument('--kernel_size', action=util.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...", help="Kernel Size of Local Attention Block")

        parser.add_argument('--layers', type=int, default=3, help='number of layers in G')
        parser.add_argument('--netG', type=str, default='pose', help='The name of net Generator')
        parser.add_argument('--netD', type=str, default='res', help='The name of net Discriminator')
        parser.add_argument('--init_type', type=str, default='orthogonal', help='Initial type')

        # if is_train:
        parser.add_argument('--ratio_g2d', type=float, default=0.1, help='learning rate ratio G to D')
        parser.add_argument('--lambda_rec', type=float, default=5.0, help='weight for image reconstruction loss')
        parser.add_argument('--lambda_g', type=float, default=2.0, help='weight for generation loss')
        parser.add_argument('--lambda_correct', type=float, default=5.0, help='weight for the Sampling Correctness loss')
        parser.add_argument('--lambda_style', type=float, default=500.0, help='weight for the VGG19 style loss')
        parser.add_argument('--lambda_content', type=float, default=0.5, help='weight for the VGG19 content loss')
        parser.add_argument('--lambda_regularization', type=float, default=0.0025, help='weight for the affine regularization loss')

        parser.add_argument('--use_spect_g', action='store_false', help="whether use spectral normalization in generator")
        parser.add_argument('--use_spect_d', action='store_false', help="whether use spectral normalization in discriminator")
        parser.add_argument('--save_input', action='store_false', help="whether save the input images when testing")

        parser.set_defaults(use_spect_g=False)
        parser.set_defaults(use_spect_d=True)
        parser.set_defaults(save_input=False)

        return parser


    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['app_gen', 'correctness_gen', 'content_gen', 'style_gen', 'regularization',
                           'ad_img_gen', 'ad_seg_gen', 'dis_img_gen', 'dis_seg_gen', 'cx', 'layout_l1']

        self.visual_names = ['input_P1','input_P2', 'img_gen', 'flow_fields', 'masks', 'layout', 'target_layout']
        self.model_names = {'G':['source','target','flow_net'],'imgD':[],'segD':[]}

        self.keys = ['head','body','leg']
        self.mask_id = {'head':[1,2,4,13],'body':[3,5,6,7,10,11,14,15],'leg':[8,9,12,16,17,18,19]}
        self.channel_id = {'head':[0,14,15,16,17],'body':[1,2,3,4,5,6,7,8,11],'leg':[8,9,10,11,12,13]}
        # self.target_mask_id = {'backgrand':[0],'hair':[1,2],'head':[4,13],'arm':[14,15],'body':[3,5,6,7,10,11],'leg':[16,17],'pants':[9,12],'shoes':[8,18,19]}
        self.target_mask_id = {'backgrand':[0],'head':[1,2,4,13],'body':[3,5,6,7,10,11,14,15],'leg':[8,9,12,16,17,18,19]}
        self.GPU = torch.device('cuda:0')

        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids)>0 \
            else torch.FloatTensor

        # define the generator
        self.net_G = network.define_g(opt, batchSize=opt.batchSize, image_nc=opt.image_nc, structure_nc=opt.structure_nc, ngf=64, img_f=512,
                                      layers=opt.layers, num_blocks=2, use_spect=opt.use_spect_g, attn_layer=opt.attn_layer, 
                                      norm='instance', activation='LeakyReLU', extractor_kz=opt.kernel_size)

        # define the discriminator 
        if self.opt.dataset_mode == 'fashion':
            self.net_imgD = network.define_d(opt, ndf=32, img_f=128, layers=4, use_spect=opt.use_spect_d)
            self.net_segD = network.define_d(opt, input_nc=4, ndf=32, img_f=128, layers=4, use_spect=opt.use_spect_d)
        elif self.opt.dataset_mode== 'market':
            self.net_D = network.define_d(opt, ndf=32, img_f=128, layers=3, use_spect=opt.use_spect_d)
        self.flow2color = util.flow2color()
        self.layout2color = util.layout2color

        if self.isTrain:
            # define the loss functions
            self.GANloss = external_function.AdversarialLoss(opt.gan_mode).to(opt.device)
            self.L1loss = torch.nn.L1Loss()
            # self.Correctness = external_function.PerceptualCorrectness().to(opt.device)
            self.Regularization = external_function.MultiAffineRegularizationLoss(kz_dic=opt.kernel_size).to(opt.device)
            self.Vggloss = external_function.VGGLoss(self.opt.batchSize).to(opt.device)

            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                                               filter(lambda p: p.requires_grad, self.net_G.parameters())),
                                               lr=opt.lr, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_G)

            self.optimizer_imgD = torch.optim.Adam(itertools.chain(
                                filter(lambda p: p.requires_grad, self.net_imgD.parameters())),
                                lr=opt.lr*opt.ratio_g2d, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_imgD)

            self.optimizer_segD = torch.optim.Adam(itertools.chain(
                                filter(lambda p: p.requires_grad, self.net_segD.parameters())),
                                lr=opt.lr*opt.ratio_g2d, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_segD)

        # load the pre-trained model and schedulers
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

        input_P1mask,input_P1backmask = pose_utils.obtain_mask(input_P1mask,self.mask_id,self.keys)
        input_P2mask,input_P2backmask = pose_utils.obtain_mask(input_P2mask,self.mask_id,self.keys)
        self.input_P1 = self.input_fullP1.repeat(3,1,1,1)*input_P1mask
        self.input_P1_back = self.input_fullP1*input_P1backmask
        self.input_P2 = self.input_fullP2.repeat(3,1,1,1)*input_P2mask
        self.input_P2mask = input_P2mask.float()
        self.input_P1backmask = input_P1backmask
        self.input_P2backmask = input_P2backmask
        self.input_BP1 = pose_utils.cords_to_map(input['BP1'],self.channel_id,self.keys,self.GPU,self.opt,input['affine'])
        self.input_BP2 = pose_utils.cords_to_map(input['BP2'],self.channel_id,self.keys,self.GPU,self.opt)
        self.target_BP = pose_utils.cords_to_map_full(input['BP2'],self.GPU,self.opt)

        self.image_paths=[]
        for i in range(self.opt.batchSize):
            self.image_paths.append(os.path.splitext(input['P1_path'][i])[0] + '_2_' + input['P2_path'][i])


    def test(self):
        """Forward function used in test time"""
        img_gen, flow_fields, masks = self.net_G(self.input_P1, self.input_BP1, self.input_BP2, self.target_BP, self.input_fullP1, (1.0-self.input_P1backmask), self.input_P2mask ,self.input_P2backmask)
        res_img = torch.cat([self.input_fullP1*(1.0-self.input_P1backmask),self.input_fullP2*(1.0-self.input_P2backmask),img_gen],3)
        self.save_results(res_img,self.opt.results_dir, data_name='vis')
        if self.opt.calcfid:
            self.save_results(img_gen,'calc_fid_dir', data_name='gen')

    def forward(self):
        """Run forward processing to get the inputs"""
        self.img_gen, self.flow_fields, self.masks, self.layout = self.net_G(self.input_P1, self.input_BP1, self.input_BP2, self.target_BP, self.input_fullP1, (1.0-self.input_P1backmask), self.input_P2mask ,self.input_P2backmask)


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        # gradient penalty for wgan-gp
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
            D_loss += gradient_penalty

        D_loss.backward()

        return D_loss

    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        base_function._unfreeze(self.net_imgD)
        base_function._unfreeze(self.net_segD)
        self.loss_dis_img_gen = self.backward_D_basic(self.net_imgD, self.input_fullP2, self.img_gen) #注意有无背景！
        self.loss_dis_seg_gan = self.backward_D_basic(self.net_segD, self.target_layout, self.layout)

    def backward_G(self):
        """Calculate training loss for the generator"""
        # Calculate l1 loss 
        loss_app_gen = self.L1loss(self.img_gen, self.input_fullP2) #注意有无背景！
        self.loss_app_gen = loss_app_gen * self.opt.lambda_rec
        
        loss_layout_l1 = self.L1loss(self.layout,self.target_layout)
        self.loss_layout_l1 = loss_layout_l1*1.0

        # Calculate Sampling Correctness Loss        
        # loss_correctness_gen = self.Correctness(self.input_P2[:self.opt.batchSize], self.input_P1[:self.opt.batchSize], self.flow_fields[0], self.opt.attn_layer)+\
        #     self.Correctness(self.input_P2[self.opt.batchSize:2*self.opt.batchSize], self.input_P1[self.opt.batchSize:2*self.opt.batchSize], self.flow_fields[1], self.opt.attn_layer)+\
        #         self.Correctness(self.input_P2[2*self.opt.batchSize:], self.input_P1[2*self.opt.batchSize:], self.flow_fields[2], self.opt.attn_layer)
        # self.loss_correctness_gen = loss_correctness_gen * self.opt.lambda_correct        

        # Calculate GAN loss
        base_function._freeze(self.net_imgD)
        base_function._freeze(self.net_segD)
        D_img_fake = self.net_imgD(self.img_gen)
        D_seg_fake = self.net_segD(self.layout)
        self.loss_ad_img_gen = self.GANloss(D_img_fake, True, False) * self.opt.lambda_g
        self.loss_ad_seg_gen = self.GANloss(D_seg_fake, True, False) * self.opt.lambda_g

        # Calculate regularization term 
        loss_regularization = self.Regularization(self.flow_fields[0])+self.Regularization(self.flow_fields[1])+self.Regularization(self.flow_fields[2])
        self.loss_regularization = loss_regularization * self.opt.lambda_regularization

        # Calculate perceptual loss
        loss_content_gen, loss_style_gen, loss_correctness_gen, loss_cx = self.Vggloss(self.img_gen, self.input_fullP2, self.input_P2, self.input_P1, self.flow_fields, self.opt.attn_layer) #注意有无背景！
        self.loss_style_gen = loss_style_gen*self.opt.lambda_style
        self.loss_content_gen = loss_content_gen*self.opt.lambda_content
        self.loss_correctness_gen = loss_correctness_gen * self.opt.lambda_correct
        self.loss_cx = loss_cx * 0.05

        total_loss = 0

        for name in self.loss_names:
            if name not in ['dis_img_gen','dis_seg_gen']:
                total_loss += getattr(self, "loss_" + name)
        total_loss.backward()


    def optimize_parameters(self):
        """update network weights"""
        self.forward()

        self.optimizer_imgD.zero_grad()
        self.optimizer_segD.zero_grad()
        self.backward_D()
        self.optimizer_imgD.step()
        self.optimizer_segD.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
