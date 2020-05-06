import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
import numpy as np
from itertools import permutations,combinations
from torch.nn.modules.module import Module
import math
import logging as log
from utils import get_base_model
from model_helpers import PyramidFeatures, Fusion, ClassificationModel, RegressionModel, ObjectDetectionHeads, DecoderNetwork
from losses import compute_ts_road_map, compute_ats_bounding_boxes
from utils import block, dblock,dice_loss

OUT_BLOCK4_DIMENSION_DICT = {"resnet18": 512, "resnet34":512, "resnet50":2048, "resnet101":2048,
                              "resnet152":2048}

def get_adv_model(name, args):

    return {"generator":ViewGANModels(args),"discriminator":Discriminator(args)}

class Discriminator(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.stage = None
        self.disc_finetune_params = []

        self.type = self.args.disc_type

        if "patch" in self.type:
            channels = [16,32,64,128,1]
        else:
            channels = [64,128,256,256,512,512,512]

        network_layers = []
        initc = 1
        for i,channel in enumerate(channels):
            if i == len(channels)-1 and "patch" in self.type:
                network_layers.append(block(initc,channel,3,1,1,activation="sigmoid",norm=False))
            elif i==0:
                network_layers.append(block(initc,channel,4,2,1,activation="leakyrelu",norm=False))
            else:
                network_layers.append(block(initc,channel,4,2,1,activation="leakyrelu"))

            initc = channel
            
        self.network = nn.Sequential(*network_layers)

        self.disc_finetune_params += list(self.network.parameters())

        if "patch" not in self.type:
            self.final = dblock(2048,1,activation="sigmoid")
            self.disc_finetune_params += list(self.final.parameters())
        
        # print(self.network)
    def config_stage(self, stage):
    
        self.stage = "finetune"

        disc_param_groups = [
            {
                "params": self.disc_finetune_params,
                "max_lr": self.args.finetune_learning_rate,
                "weight_decay": self.args.finetune_weight_decay,
            }
        ]


        optimizer_D = optim.Adam(disc_param_groups, lr=self.args.finetune_learning_rate, betas= [0.5,0.99])

        return optimizer_D

        
    def forward(self, dinput):
        
        output = self.network(dinput)
        b,c,h,w = output.shape

        if "patch" in self.type:
            out = output
        else:
            # print(output.view(b,-1).shape)
            out = self.final(output.view(b,-1))

        return out
        


class GAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.stage = None
        self.shared_params = []
        self.gen_finetune_params = []

    def forward(self, batch_input, task=None):
 
        raise NotImplementedError

    def config_stage(self, stage):

        self.stage = "finetune"
        
        gen_param_groups = [
            {
                "params": self.gen_finetune_params,
                "max_lr": self.args.finetune_learning_rate,
                "weight_decay": self.args.finetune_weight_decay,
            }
        ]
        if self.args.transfer_paradigm == "tunable":
            gen_param_groups.append(
                {
                    "params": self.shared_params,
                    "max_lr": self.args.finetune_learning_rate,
                    "weight_decay": self.args.finetune_weight_decay,
                }
            )

        optimizer_G = optim.Adam(gen_param_groups, lr=self.args.finetune_learning_rate, betas= [0.5,0.99])

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer=optimizer,
        #     mode = "min",
        #     factor=0.1, 
        #     patience=10, 
        #     verbose=False, 
        #     threshold=0.0001, 
        #     threshold_mode='rel', 
        #     cooldown=0, 
        #     min_lr=0, 
        #     eps=1e-08
        # )

        return optimizer_G

class ViewGANModels(GAN):
    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self.d_model = OUT_BLOCK4_DIMENSION_DICT[self.args.network_base]

        full_resnet = get_base_model(self.args.network_base)

        self.init_layers = nn.Sequential(
            full_resnet.conv1,
            full_resnet.bn1,
            full_resnet.relu,
            full_resnet.maxpool,
        )

        self.block1 = full_resnet.layer1
        self.block2 = full_resnet.layer2
        self.block3 = full_resnet.layer3
        self.block4 = full_resnet.layer4

        self.image_network = nn.Sequential(
            full_resnet.conv1,
            full_resnet.bn1,
            full_resnet.relu,
            full_resnet.maxpool,
            self.block1,
            self.block2,
            self.block3,
            self.block4,
        )

        self.obj_detection_head = self.args.obj_det_head

        self.gen_roadmap = self.args.gen_road_map
        self.detect_objects = self.args.detect_objects

        self.fusion = self.args.view_fusion_strategy
        
        self.obj_detection_head = "retinanet"

        if self.d_model > 1024:
            self.max_f = 1024
        else:
            self.max_f = self.d_model

        # print(self.args.finetune_obj)
        self.model_type = self.args.finetune_obj.split("_")[0]

        self.input_dim = 256

        self.blobs_strategy = self.args.blobs_strategy
        
        self.conditional_generator = "cond" in self.args.finetune_obj

        # print(self.fusion)
        self.dense_fuse = "dense" in self.fusion
        self.conv_fuse = "conv" in self.fusion
        self.dense_before_fuse = "dbf" in self.fusion
        self.dense_after_fuse = "daf" in self.fusion

        self.frefine_layers = 0
        self.brefine_layers = 0

        if self.conv_fuse or self.dense_fuse:
            self.frefine_layers = 1
            self.brefine_layers = 1

        self.drefine_layers = 0
        self.dcrefine_layers = 0
        if self.dense_before_fuse:
            self.drefine_layers = 1
        if self.dense_after_fuse or self.conditional_generator:
            self.dcrefine_layers = 1

            # print("dfuse",self.dense_fuse, "cfuse", self.conv_fuse, "dproj", self.dense_project)
        self.fuse = Fusion(args, self.d_model, frefine_layers=self.frefine_layers, brefine_layers=self.brefine_layers, drefine_layers=self.drefine_layers, dense_fusion=self.dense_fuse, conv_fusion=self.conv_fuse, dcrefine_layers=self.dcrefine_layers)                           
        # print(self.fuse)
                                              
        out_dim = 1
        
        if self.dense_fuse:
    
            init_layer_dim = 32
            init_channel_dim = 64
            
            self.decoder_network = DecoderNetwork(init_layer_dim, init_channel_dim, self.max_f, self.d_model, add_initial_upsample_conv=True)

            
        else:
            
            if self.drefine_layers > 0 or self.dcrefine_layers > 0:
                
                if self.dcrefine_layers > 0:
                    init_layer_dim = 32
                    init_channel_dim = 64
                    self.decoder_network = DecoderNetwork(init_layer_dim, init_channel_dim, self.max_f, self.d_model, add_initial_upsample_conv=True)

                else:
                    init_layer_dim = 32
                    init_channel_dim = 128
                    self.decoder_network = DecoderNetwork(init_layer_dim, init_channel_dim, self.max_f, self.d_model, add_initial_upsample_conv=True)
                # print("add_convs_before_decoding", self.add_convs_before_decoding)
                
            else:
                init_layer_dim = 8
                init_channel_dim = 128
                self.decoder_network = DecoderNetwork(init_layer_dim, init_channel_dim, self.max_f, self.d_model, add_convs_before_decoding=True)

            
            # print(self.decoder_network)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))     

        self.generator = nn.Sequential(self.image_network, self.fuse, self.decoder_network)

        # self.discriminator = Discriminator(self.args.disc_type)

        self.loss_type = "bce"
        if self.loss_type == "mse":
            self.criterion = torch.nn.MSELoss()
        elif self.loss_type == "bce":
            self.criterion = torch.nn.BCELoss()
        
        # if "retinanet" in self.obj_detection_head and self.detect_objects:
        #     if "decoder" in self.blobs_strategy:
        #         self.obj_detection_model = ObjectDetectionHeads(args,self.image_network, self.decoder_network)
        #     else:
        #         self.obj_detection_model = ObjectDetectionHeads(args,self.image_network)
        
        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)

        self.sigmoid = nn.Sigmoid()
        self.shared_params = list(self.image_network.parameters())

        if args.imagessl_load_ckpt is not "none":
            self.gen_finetune_params += list(self.fuse.parameters())
        else:
            self.shared_params += list(self.fuse.parameters())

        self.gen_finetune_params += list(self.decoder_network.parameters())

        # self.disc_finetune_params += list(self.discriminator.parameters())

    def forward(self, batch_input, task="none"):
        batch_output = {}
        
        # index = batch_input["idx"]
        self.stage = "finetune"

        views = batch_input["image"]
        device = views.device
        bs = views.size(0)
        self.batch_size = bs

        
        # road_map = batch_input["road"]
        
        gen_latent_features = self.image_network(views.flatten(0,1))

        _,c,h,w = gen_latent_features.shape
        views = gen_latent_features.view(bs,6,c,h,w)

        batch_output["loss"] = 0

        # print("views", views.shape)
        if "cond" in self.args.finetune_obj:
            z = torch.randn(bs, self.args.latent_dim).to(device)
            fusion = self.fuse(views,z)
        else:
            fusion = self.fuse(views)
        # print("fusion", fusion.shape)

        # print("here")
        # if self.dense_fuse:

        #     fusion = self.reshape(fusion).view(-1,32,16,16)

        # print("reshape", fusion.shape)

        gen_image = self.decoder_network(fusion)#fusion)

        # real_disc_inp = batch_input["road"]
        # fake_disc_inp = gen_image.detach()

        # if "patch" in self.args.disc_type:
        #     b,c,h,w = real_disc_inp.shape
        #     # real_disc_inp = real_disc_inp.view(b,-1)
        #     # fake_disc_inp = fake_disc_inp.view(b,-1)
        #     zeros = torch.zeros(bs,1,16,16).to(device)
        #     ones = torch.ones(bs,1,16,16).to(device)

        # else:
        #     zeros = torch.zeros(bs,1).to(device)
        #     ones = torch.ones(bs,1).to(device)

        # real_disc_op = self.discriminator(real_disc_inp)
        # batch_output["real_dloss"] = self.criterion(real_disc_op,ones)

        # fake_disc_op = self.discriminator(fake_disc_inp)
        # batch_output["fake_dloss"] = self.criterion(fake_disc_op,zeros)
        # batch_output["Dloss"] = batch_output["real_dloss"] + batch_output["fake_dloss"]

        batch_output["road_map"] = torch.sigmoid(gen_image)
        batch_output["ts_road_map"] = compute_ts_road_map(batch_output["road_map"],batch_input["road"])
        batch_output["ts"] = batch_output["ts_road_map"]
        # batch_output["GDiscloss"] = self.criterion(fake_disc_op,ones)
        if self.args.road_map_loss == "dice":
            batch_output["GSupLoss"] = dice_loss(batch_input["road"], batch_output["road_map"])
        else:
            batch_output["GSupLoss"] = self.criterion(batch_output["road_map"], batch_input["road"])
        # else:
        #     batch_output["GSupLoss"] = self.criterion(batch_output["road_map"], batch_input["road"])

        # batch_output["GSupLoss"] = self.criterion(batch_output["road_map"],batch_input["road"])
        # batch_output["Gloss"] = batch_output["GDiscloss"] + batch_output["GSupLoss"]

        # batch_output["loss"] = batch_output["Dloss"] + batch_output["Gloss"]


        # if self.training:
        #     batch_output["recon_loss"] = self.criterion(mapped_image, road_map)
        #     batch_output["road_map"] = torch.sigmoid(mapped_image)
        #     batch_output["ts_road_map"] = compute_ts_road_map(batch_output["road_map"],road_map)
        #     batch_output["loss"] += batch_output["recon_loss"]
        # else:
        #     return torch.sigmoid(mapped_image)

        # if self.detect_objects:
            
        #     if "decoder" in self.blobs_strategy:
        #         if "var" in self.model_type:
        #             batch_output = self.obj_detection_model(z,batch_input,batch_output)
        #         else:
        #             batch_output = self.obj_detection_model(fusion,batch_input,batch_output,fusion)
        #     else:
        #         batch_output = self.obj_detection_model(batch_input["image"],batch_input,batch_output)

        return batch_output
        

        # if self.detect_objects:
        #     self.finetune_params += list(self.obj_detection_model.params.parameters())

