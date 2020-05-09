import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.networks.base_network import BaseNetwork
from model.networks.resample2d_package.resample2d import Resample2d
from model.networks.base_function import *
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
from .inpaintor import InpaintSANet
from collections import OrderedDict

######################################################################################################
# Human Pose Image Generation 
######################################################################################################
class PoseGenerator(BaseNetwork):
    def __init__(self,  batchSize, image_nc=3, structure_nc=18, output_nc=3, ngf=64, img_f=1024, layers=6, num_blocks=2, 
                norm='batch', activation='ReLU', attn_layer=[1,2], extractor_kz={'1':5,'2':5}, use_spect=True, use_coord=False):  
        super(PoseGenerator, self).__init__()
        # self.backgrand = InpaintSANet(c_dim=4)

        self.source = PoseSourceNet(image_nc, ngf, img_f, layers, 
                                                    norm, activation, use_spect, use_coord)
        self.target = PoseTargetNet(batchSize, image_nc, structure_nc, output_nc, ngf, img_f, layers, num_blocks, 
                                                norm, activation, attn_layer, extractor_kz, use_spect, use_coord)
        self.flow_net = PoseFlowNet(image_nc, structure_nc, ngf=32, img_f=256, encoder_layer=5, 
                                    attn_layer=attn_layer, norm=norm, activation=activation,
                                    use_spect=use_spect, use_coord=use_coord)       


    def forward(self, source, source_B, target_B, source_full, source_body_mask, target_mask, target_backgrand_mask):
        source_a,source_b,source_c = torch.chunk(source,3)
        feature_list_a = self.source(source_a)
        feature_list_b = self.source(source_b)
        feature_list_c = self.source(source_c)

        # source_backgrand = self.backgrand(source_full,masks=source_body_mask,only_x=True)

        source_B_a,source_B_b,source_B_c = torch.chunk(source_B,3)
        target_B_a,target_B_b,target_B_c = torch.chunk(target_B,3)
        flow_fields_a, masks_a = self.flow_net(source_a, source_B_a, target_B_a)
        flow_fields_b, masks_b = self.flow_net(source_b, source_B_b, target_B_b)
        flow_fields_c, masks_c = self.flow_net(source_c, source_B_c, target_B_c)

        b,c,h,w = target_B.size()
        target_B = target_B.view(3,-1,c,h,w)
        target_B = torch.sum(target_B,0)
        image_gen = self.target(target_B, [feature_list_a,feature_list_b,feature_list_c], [flow_fields_a,flow_fields_b,flow_fields_c], [masks_a,masks_b,masks_c], target_mask)
        # b,c,h,w = image_gen.size()
        # gen = image_gen*target_mask
        # gen = gen.view(3,-1,c,h,w)
        # gen = torch.sum(gen,0)
        # image_gen = image_gen*(1.0-target_backgrand_mask)# + source_backgrand*target_backgrand_mask
        return image_gen, [flow_fields_a,flow_fields_b,flow_fields_c], [masks_a,masks_b,masks_c]

    def forward_hook_function(self, source, source_B, target_B):
        feature_list = self.source(source)
        flow_fields, masks = self.flow_net(source, source_B, target_B)
        hook_target, hook_source, hook_attn, hook_mask = self.target.forward_hook_function(target_B, feature_list, flow_fields, masks)        
        return hook_target, hook_source, hook_attn, hook_mask



class PoseSourceNet(BaseNetwork):
    def __init__(self, input_nc=3, ngf=64, img_f=1024, layers=6, norm='batch',
                activation='ReLU', use_spect=True, use_coord=False):  
        super(PoseSourceNet, self).__init__()
        self.layers = layers
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        # encoder part CONV_BLOCKS
        self.block0 = EncoderBlock(input_nc, ngf, norm_layer,
                                 nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ngf)
            block = EncoderBlock(ngf*mult_prev, ngf*mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)        


    def forward(self, source):
        feature_list=[source]
        out = self.block0(source)
        feature_list.append(out)
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) 
            feature_list.append(out)

        feature_list = list(reversed(feature_list))
        return feature_list


class PoseTargetNet(BaseNetwork):
    def __init__(self, batchSize, image_nc=3, structure_nc=18, output_nc=3, ngf=64, img_f=1024, layers=6, num_blocks=2, 
                norm='batch', activation='ReLU', attn_layer=[1,2], extractor_kz={'1':5,'2':5}, use_spect=True, use_coord=False):  
        super(PoseTargetNet, self).__init__()
        self.batchSize = batchSize
        self.layers = layers
        self.attn_layer = attn_layer

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)


        self.block0 = EncoderBlock(structure_nc, ngf, norm_layer,
                                 nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ngf)
            block = EncoderBlock(ngf*mult_prev, ngf*mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)         


        # decoder part
        mult = min(2 ** (layers-1), img_f//ngf)
        for i in range(layers):
            mult_prev = mult
            mult = min(2 ** (layers-i-2), img_f//ngf) if i != layers-1 else 1
            if num_blocks == 1:
                up = nn.Sequential(ResBlockDecoder(ngf*mult_prev, ngf*mult, None, norm_layer, 
                                         nonlinearity, use_spect, use_coord))
            else:
                up = nn.Sequential(ResBlocks(num_blocks-1, ngf*mult_prev, None, None, norm_layer, 
                                             nonlinearity, False, use_spect, use_coord),
                                   ResBlockDecoder(ngf*mult_prev, ngf*mult, None, norm_layer, 
                                             nonlinearity, use_spect, use_coord))
            setattr(self, 'decoder' + str(i), up)

            if layers-i in attn_layer:
                attn = ExtractorAttn(ngf*mult_prev, extractor_kz[str(layers-i)], nonlinearity, softmax=True)
                setattr(self, 'attn' + str(i), attn)

        self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spect, use_coord)

    def mask_maker(self,mask,target_mask):
        _,_,h,w = mask.size()
        my_mask = torch.nn.functional.interpolate(target_mask, (h,w))
        return my_mask*mask

    def forward(self, target_B, source_feature, flow_fields, masks, target_mask):
        out = self.block0(target_B)
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) 

        counter=0
        for i in range(self.layers):
            if self.layers-i in self.attn_layer:
                model = getattr(self, 'attn' + str(i))

                my_mask_a = self.mask_maker(masks[0][counter],target_mask[:self.batchSize])
                my_mask_b = self.mask_maker(masks[1][counter],target_mask[self.batchSize:2*self.batchSize])
                my_mask_c = self.mask_maker(masks[2][counter],target_mask[2*self.batchSize:])
                my_mask = my_mask_a+my_mask_b+my_mask_c
                _,_,h,w = flow_fields[0][counter].size()
                target_mask = torch.nn.functional.interpolate(target_mask, (h,w))
                out_attn = model([source_feature[0][i],source_feature[1][i],source_feature[2][i]], out, [flow_fields[0][counter],flow_fields[1][counter],flow_fields[2][counter]],[target_mask[:self.batchSize],target_mask[self.batchSize:2*self.batchSize],target_mask[2*self.batchSize:]])        
                out = out*(1-my_mask) + out_attn*my_mask
                counter += 1

            model = getattr(self, 'decoder' + str(i))
            out = model(out)

        out_image = self.outconv(out)
        return out_image

    def forward_hook_function(self, target_B, source_feature, flow_fields, masks):
        hook_target=[]
        hook_source=[]      
        hook_attn=[]      
        hook_mask=[]      
        out = self.block0(target_B)
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) 

        counter=0
        for i in range(self.layers):
            if self.layers-i in self.attn_layer:
                model = getattr(self, 'attn' + str(i))

                attn_param, out_attn = model.hook_attn_param(source_feature[i], out, flow_fields[counter])        
                out = out*(1-masks[counter]) + out_attn*masks[counter]

                hook_target.append(out)
                hook_source.append(source_feature[i])
                hook_attn.append(attn_param)
                hook_mask.append(masks[counter])
                counter += 1

            model = getattr(self, 'decoder' + str(i))
            out = model(out)

        out_image = self.outconv(out)
        return hook_target, hook_source, hook_attn, hook_mask    


class PoseFlowNet(nn.Module):
    """docstring for FlowNet"""
    def __init__(self, image_nc, structure_nc, ngf=64, img_f=1024, encoder_layer=5, attn_layer=[1], norm='batch',
                activation='ReLU', use_spect=True, use_coord=False):
        super(PoseFlowNet, self).__init__()

        self.encoder_layer = encoder_layer
        self.decoder_layer = encoder_layer - min(attn_layer)
        self.attn_layer = attn_layer
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        input_nc = 2*structure_nc + image_nc

        self.block0 = EncoderBlock(input_nc, ngf, norm_layer,
                                 nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(encoder_layer-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ngf)
            block = EncoderBlock(ngf*mult_prev, ngf*mult,  norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)         
        
        for i in range(self.decoder_layer):
            mult_prev = mult
            mult = min(2 ** (encoder_layer-i-2), img_f//ngf) if i != encoder_layer-1 else 1
            up = ResBlockDecoder(ngf*mult_prev, ngf*mult, ngf*mult, norm_layer, 
                                    nonlinearity, use_spect, use_coord)
            setattr(self, 'decoder' + str(i), up)
            
            jumpconv = Jump(ngf*mult, ngf*mult, 3, None, nonlinearity, use_spect, use_coord)
            setattr(self, 'jump' + str(i), jumpconv)

            if encoder_layer-i-1 in attn_layer:
                flow_out = nn.Conv2d(ngf*mult, 2, kernel_size=3,stride=1,padding=1,bias=True)
                setattr(self, 'output' + str(i), flow_out)

                flow_mask = nn.Sequential(nn.Conv2d(ngf*mult, 1, kernel_size=3,stride=1,padding=1,bias=True),
                                          nn.Sigmoid())
                setattr(self, 'mask' + str(i), flow_mask)


    def forward(self, source, source_B, target_B):
        flow_fields=[]
        masks=[]
        outs={}
        inputs = torch.cat((source, source_B, target_B), 1) 
        out = self.block0(inputs)
        result=[out]
        for i in range(self.encoder_layer-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            result.append(out) 
        for i in range(self.decoder_layer):
            model = getattr(self, 'decoder' + str(i))
            out = model(out)

            model = getattr(self, 'jump' + str(i))
            jump = model(result[self.encoder_layer-i-2])
            out = out+jump

            if self.encoder_layer-i-1 in self.attn_layer:
                flow_field, mask = self.attn_output(out, i)
                flow_fields.append(flow_field)
                masks.append(mask)
                outs[i] = out

        return flow_fields, masks, outs

    def attn_output(self, out, i):
        model = getattr(self, 'output' + str(i))
        flow = model(out)
        model = getattr(self, 'mask' + str(i))
        mask = model(out)
        return flow, mask  

class PoseFlowNetGenerator(BaseNetwork):
    def __init__(self, image_nc=3, structure_nc=18, output_nc=3, ngf=64,  img_f=1024, layers=6, norm='batch',
                activation='ReLU', encoder_layer=5, attn_layer=[1,2], use_spect=True, use_coord=False):  
        super(PoseFlowNetGenerator, self).__init__()

        self.layers = layers
        self.encoder_layer = encoder_layer
        self.decoder_layer = encoder_layer - min(attn_layer)
        self.attn_layer = attn_layer

        self.flow_net = PoseFlowNet(image_nc, structure_nc, ngf, img_f, 
                        encoder_layer, attn_layer=attn_layer,
                        norm=norm, activation=activation, 
                        use_spect=use_spect, use_coord= use_coord)
        mult = 1
        for i in range(encoder_layer-1):
            mult = min(2 ** (i + 1), img_f//ngf)
        for i in range(self.decoder_layer):
            if self.encoder_layer-i-1 in self.attn_layer:
                layout = nn.Sequential(nn.Conv2d(3*ngf*mult, 4, kernel_size=3,stride=1,padding=1,bias=True),
                                          nn.Softmax2d())
                setattr(self, 'layout' + str(i), layout)

    def forward(self, source, source_B, target_B):
        source_a, source_b, source_c = torch.chunk(source)
        source_B_a, source_B_b, source_B_c = torch.chunk(source_B)
        target_B_a, target_B_b, target_B_c = torch.chunk(target_B)

        flow_fields_a, masks_a, outs_a = self.flow_net(source_a, source_B_a, target_B_a)
        flow_fields_b, masks_b, outs_b = self.flow_net(source_b, source_B_b, target_B_b)
        flow_fields_c, masks_c, outs_c = self.flow_net(source_c, source_B_c, target_B_c)

        layouts = []
        for i in range(self.decoder_layer):
            if self.encoder_layer-i-1 in self.attn_layer:
                model = getattr(self,'layout'+str(i))
                tmp_out = torch.cat([outs_a[i],outs_b[i],outs_c[i]],1)
                layouts.append(model(tmp_out))
        return [flow_fields_a, flow_fields_b, flow_fields_c], [masks_a, masks_b, masks_c], layouts

