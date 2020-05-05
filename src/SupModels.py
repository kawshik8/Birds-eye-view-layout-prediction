# This file implements models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
import numpy as np
from torchvision.ops import nms
from itertools import permutations,combinations
from torch.nn.modules.module import Module
import losses
from losses import compute_ts_road_map, compute_ats_bounding_boxes
import math
import logging as log
from GANmodels import get_adv_model
from utils import get_base_model
from model_helpers import PyramidFeatures, Fusion, ClassificationModel, RegressionModel, ObjectDetectionHeads, DecoderNetwork
from utils import dblock

OUT_BLOCK4_DIMENSION_DICT = {"resnet18": 512, "resnet34":512, "resnet50":2048, "resnet101":2048,
                              "resnet152":2048}

def get_sup_model(name, args):
    if "adv" in args.finetune_obj:
        return get_adv_model(name,args)
    else:
        return ViewGenModels(args)


class ViewModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.stage = None
        self.shared_params = []
        self.finetune_params = []

    def forward(self, batch_input, task=None):
 
        raise NotImplementedError

    def config_stage(self, stage):

        self.stage = "finetune"
        
        param_groups = [
            {
                "params": self.finetune_params,
                "max_lr": self.args.finetune_learning_rate,
                "weight_decay": self.args.finetune_weight_decay,
            }
        ]
        if self.args.transfer_paradigm == "tunable":
            param_groups.append(
                {
                    "params": self.shared_params,
                    "max_lr": self.args.finetune_learning_rate,
                    "weight_decay": self.args.finetune_weight_decay,
                }
            )
        optimizer = optim.SGD(param_groups, lr=self.args.finetune_learning_rate, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode = "min",
            factor=0.1, 
            patience=10, 
            verbose=False, 
            threshold=0.0001, 
            threshold_mode='rel', 
            cooldown=0, 
            min_lr=0, 
            eps=1e-08
        )

        return optimizer, scheduler

class ViewGenModels(ViewModel):
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
        if self.dense_after_fuse:
            self.dcrefine_layers = 1

        if self.gen_roadmap or (self.detect_objects and "decoder" in self.blobs_strategy):
            # print("dfuse",self.dense_fuse, "cfuse", self.conv_fuse, "dproj", self.dense_project)
            self.fuse = Fusion(args, self.d_model, frefine_layers=self.frefine_layers, brefine_layers=self.brefine_layers, drefine_layers=self.drefine_layers, dense_fusion=self.dense_fuse, conv_fusion=self.conv_fuse, dcrefine_layers=self.dcrefine_layers)                           
            # print(self.fuse)
                                              
        out_dim = 1
        
        
        
        if self.gen_roadmap:

            if self.model_type == "det":                   
                                    
                if self.dense_fuse:

                    init_layer_dim = 32
                    init_channel_dim = 64
                    
                    self.decoder_network = DecoderNetwork(init_layer_dim, init_channel_dim, self.max_f, self.d_model, add_initial_upsample_conv=True)

                    self.decoding = nn.Sequential(self.decoder_network)
                    
                else:
                    
                    if self.drefine_layers > 0 or self.dcrefine_layers > 0:
                        # print("changed add_convs_before_decoding")
                        
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

                   

                    self.decoding = nn.Sequential(self.decoder_network)

            else:
                self.max_f = 64
                self.latent_dim = self.args.latent_dim
                init_layer_dim = 32
                init_channel_dim = self.max_f

                self.decoder_network = DecoderNetwork(init_layer_dim, init_channel_dim, self.max_f, self.d_model, add_initial_upsample_conv=True)

                if self.conv_fuse:
                    self.avg_pool_refine = dblock(self.latent_dim, self.latent_dim)

                self.z_project = nn.Linear(self.d_model, 2*self.latent_dim)
                self.z_refine = dblock(self.latent_dim, self.latent_dim)
                self.z_reshape = dblock(self.latent_dim,16*16*32)
                self.decoding = nn.Sequential(self.decoder_network, self.z_refine, self.refine, self.z_reshape, self.z_project)

            # print(self.decoder_network)
            self.loss_type = self.args.road_map_loss

            if self.loss_type == "mse":
                self.criterion = torch.nn.MSELoss()
            elif self.loss_type == "bce":
                self.criterion = torch.nn.BCEWithLogitsLoss()

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))     
        
        if "retinanet" in self.obj_detection_head and self.detect_objects:
            if "decoder" in self.blobs_strategy:
                self.obj_detection_model = ObjectDetectionHeads(args,self.image_network, self.decoder_network)
            else:
                self.obj_detection_model = ObjectDetectionHeads(args,self.image_network)
        

        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)

        self.sigmoid = nn.Sigmoid()
        self.shared_params = list(self.image_network.parameters())

        if self.gen_roadmap or (self.detect_objects and "decoder" in self.blobs_strategy):
            if args.imagessl_load_ckpt:
                self.finetune_params += list(self.fuse.parameters())
            else:
                self.shared_params += list(self.fuse.parameters())

        if self.gen_roadmap:
            self.finetune_params += list(self.decoding.parameters())

        if self.detect_objects:
            self.finetune_params += list(self.obj_detection_model.params.parameters())


    def reparameterize(self, mu, logvar):

        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std).to(self.args.device)
            return mu + eps*std
        else:
            return mu

    def forward(self, batch_input, task=None):
        batch_output = {}
        
        # index = batch_input["idx"]
        self.stage = "finetune"

        views = batch_input["image"]
        device = views.device
        bs = views.size(0)
        self.batch_size = bs

        
        # road_map = batch_input["road"]
        
        final_features = self.image_network(views.flatten(0,1))

        _,c,h,w = final_features.shape
        views = final_features.view(bs,6,c,h,w)

        batch_output["loss"] = 0

        # print("views", views.shape)

        if self.gen_roadmap or (self.detect_objects and "decoder" in self.blobs_strategy):
            fusion = self.fuse(views)
            # print("fusion", fusion.shape)

        if self.gen_roadmap:
            if "det" in self.model_type:
                # print("here")
                # if self.dense_fuse:

                #     fusion = self.reshape(fusion).view(-1,32,16,16)

                # print("reshape", fusion.shape)

                mapped_image = self.decoder_network(fusion)#fusion)
                
                # if self.training:
                batch_output["recon_loss"] = self.criterion(mapped_image, mapped_image)
                batch_output["road_map"] = torch.sigmoid(mapped_image)
                batch_output["ts_road_map"] = compute_ts_road_map(batch_output["road_map"],mapped_image)
                batch_output["loss"] += batch_output["recon_loss"]
                # else:
                #     return torch.sigmoid(mapped_image)

            else:
                
                if self.conv_fuse:
                    fusion = self.avg_pool_refine(self.avg_pool(fusion).view(-1,self.d_model))

                mu_logvar = self.z_project(fusion).view(bs,2,self.latent_dim)

                mu = mu_logvar[:,0,:]
                logvar = mu_logvar[:,1,:]

                z = self.reparameterize(mu,logvar)
                
                z = self.z_refine(z)

                z = self.z_reshape(z).view(bs,32,16,16)                
  
                generated_image = self.decoder_network(z)

                reconstruction_loss = self.criterion(generated_image, generated_image)
                kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                batch_output["road_map"] = torch.sigmoid(generated_image)
                batch_output["recon_loss"] = reconstruction_loss
                batch_output["KLD_loss"] = kl_divergence_loss
                batch_output["ts_road_map"] = compute_ts_road_map(batch_output["road_map"],generated_image)
                batch_output["loss"] += batch_output["recon_loss"] + batch_output["KLD_loss"]

        if self.detect_objects:
            
            if "decoder" in self.blobs_strategy:
                if "var" in self.model_type:
                    batch_output = self.obj_detection_model(z,batch_input,batch_output)
                else:
                    batch_output = self.obj_detection_model(fusion,batch_input,batch_output,fusion)
            else:
                batch_output = self.obj_detection_model(batch_input["image"],batch_input,batch_output)

        return batch_output
