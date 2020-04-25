# This file implements models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models.resnet as resnet
import itertools
import numpy as np
from itertools import permutations,combinations
from torch.nn.modules.module import Module
from utils import Anchors, BBoxTransform, ClipBoxes, block, Resblock
import losses
import math
from SupModels import get_sup_model
import logging as log



OUT_BLOCK4_DIMENSION_DICT = {"resnet18": 512, "resnet34":512, "resnet50":2048, "resnet101":2048,
                              "resnet152":2048}

def get_base_model(name):
    if name=="resnet18":
        return resnet.resnet18()
    if name=="resnet34":
        return resnet.resnet34()
    if name=="resnet50":
        return resnet.resnet50()
    if name=="resnet101":
        return resnet.resnet101()
    if name=="resnet152":
        return resnet.resnet152()


def get_model(name, args):
    # print(args.image_pretrain_obj, args.view_pretrain_obj)
    if name == "image_ssl":
        return ImageSSLModels(args)
    elif name == "view_ssl":
        return ViewSSLModels(args)
    else:
        return get_sup_model(name, args)
    

def get_part_model(model,layer):
    if layer == "res1":
        return model[:4]
    elif layer == "res2":
        return model[:5]
    elif layer == "res3":
        return model[:6]
    elif layer == "res4":
        return model[:7]
    elif layer == "res5":
        return model

class SSLModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.stage = None
        self.shared_params = []
        self.pretrain_params = []

    def forward(self, batch_input, task=None):
        """
        inputs:
            batch_input: dict[str, tensor]: inputs in one minibatch
                "idx": long (bs), index of the image instance
                "image": float (bs, num_patches, channels, height, width), pixels from raw and
                transformed image
                "query": bool (bs, num_patches), which patches are queried, only in pretrain
                "label": long (bs), class label of the image, only in fine-tune
                (if cfgs.dup_pos > 0, each image instance in minibatch will have (1 + dup_pos)
                transformed versions.)
            task: task object
        outputs:
            batch_output: dict[str, tensor]: outputs in one minibatch
                "loss": float (1), full loss of a batch
                "loss_*": float (1), one term of the loss (if more than one SSL tasks are used)
                "acc": float (1), jigsaw puzzle accuracy, only when pretrain
                "cls_acc": float (1), classification accuracy, only when finetune
                "predict": float (bs), class prediction, only when finetune
        """
        raise NotImplementedError

    def config_stage(self, stage):
        """
        switch between pretrain and finetune stages
        inputs:
            stage: str, "pretrain" or "finetune"
        outputs:
            optimizer: ...
            scheduler: ...
        """
        self.stage = "pretrain"
        param_groups = [
            {
                "params": self.shared_params + self.pretrain_params,
                "max_lr": self.args.pretrain_learning_rate,
                "weight_decay": self.args.pretrain_weight_decay,
            }
        ]
        #print(self.args.pretrain_learning_rate)
        optimizer = optim.AdamW(param_groups)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            anneal_strategy="cos",
            total_steps=self.args.pretrain_total_iters,
            pct_start=self.args.warmup_iters / self.args.pretrain_total_iters,
            cycle_momentum=False,
            max_lr=self.args.pretrain_learning_rate,
        )

        return optimizer, scheduler


def masked_select(inp, mask):
    return inp.flatten(0, len(mask.size()) - 1)[mask.flatten().nonzero()[:, 0]]

class ImageSSLModels(SSLModel):
    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self.num_patches = args.num_patches
        self.d_model = OUT_BLOCK4_DIMENSION_DICT[self.args.network_base]

        full_resnet = get_base_model(self.args.network_base)
        self.patch_network = nn.Sequential(
            full_resnet.conv1,
            full_resnet.bn1,
            full_resnet.relu,
            full_resnet.maxpool,
            full_resnet.layer1,
            full_resnet.layer2,
            full_resnet.layer3,
            full_resnet.layer4,
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.project_dim = self.args.project_dim

        self.reduce = nn.Linear(self.num_patches*self.d_model, self.d_model)
        self.project1 = nn.Linear(self.d_model, self.project_dim)
        self.project2 = nn.Linear(self.d_model, self.project_dim)

        self.pretrain_network = nn.Sequential(
            self.patch_network,
            self.avg_pool,
        )

        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)

        self.use_memory_bank = True

        if self.use_memory_bank:
            self.register_buffer("memory_bank", torch.randn(self.args.vocab_size, self.project_dim))
            self.all_indices = np.arange(self.args.vocab_size)

        self.negatives = self.args.num_negatives
        self.lambda_wt = self.args.lambda_pirl
        self.beta_wt = self.args.beta_ema

        self.sigmoid = nn.Sigmoid()

        self.shared_params = list(self.patch_network.parameters())
        self.pretrain_params = list(self.reduce.parameters()) + list(self.project1.parameters()) + list(self.project2.parameters())
        # #self.shared_params += list(self.attention_pooling.parameters())
        # self.pretrain_params = list(self.reduce.parameters()) + list(self.project1.parameters()) + list(self.project2.parameters())
        # self.finetune_params = list(self.cls_classifiers.parameters())

    def forward(self, batch_input, task=None):
        batch_output = {}
        
        index = batch_input["idx"]
        image = batch_input["image"]
        query = batch_input["query"]
        # print(inp.shape)

        device = image.device#batch_input["aug"].device
        bs = image.size(0)
        self.batch_size = bs

        patches = self.pretrain_network(query.flatten(0, 1)).view(
            bs, self.num_patches, -1
        )  # (bs, num_patches, d_model)
        
        flatten = patches.view(bs,-1)
        patches = self.project1(self.reduce(flatten))

        images = self.pretrain_network(image).view(
            bs, -1
        )

        images = self.project2(images)

        if "nce_loss" in self.args.image_pretrain_obj  or "infonce_loss" in self.args.image_pretrain_obj:

            # positives = F.cosine_similarity(images,patches,axis=-1)/0.07
            
            if self.use_memory_bank:
                positives_from_memory = self.memory_bank[index]

                batch_indices = np.array(index)

                neg_ind = list(set(self.all_indices) - set(batch_indices))

                neg_ind = neg_ind[:self.negatives]

                negatives_from_memory = self.memory_bank[neg_ind]

                total_instances = negatives_from_memory.shape[0] + 1
                
                positives = positives_from_memory.unsqueeze(1)
                negatives = negatives_from_memory.unsqueeze(0).repeat(bs,1,1)

                query1 = images.unsqueeze(1).repeat(1,total_instances,1)
                query2 = patches.unsqueeze(1).repeat(1,total_instances,1)
                
                key = torch.cat([positives,negatives],dim=1)

                scores1 = F.cosine_similarity(query1,key,axis=-1)/0.07
                scores2 = F.cosine_similarity(query2,key,axis=-1)/0.07

                jigsaw_labels = torch.zeros(bs,scores1.shape[1]).to(device)
                jigsaw_labels[:,0] = 1

                if "nce_loss" in self.args.image_pretrain_obj:
                    scores1 = torch.softmax(scores1,dim=-1)
                    scores2 = torch.softmax(scores2,dim=-1)

                    loss1 = F.binary_cross_entropy(scores1.float(), jigsaw_labels.float(),reduction='none').sum()/scores1.shape[0]
                    loss2 = F.binary_cross_entropy(scores2.float(), jigsaw_labels.float(),reduction='none').sum()/scores2.shape[0]

                    batch_output["loss"] = self.lambda_wt * loss1 + (1 - self.lambda_wt) * loss2
                    
                    batch_output["acc"] = ((scores1.max(dim=1)[1] == jigsaw_labels.max(dim=1)[1]).float().mean() + (scores2.max(dim=1)[1] == jigsaw_labels.max(dim=1)[1]).float().mean())/2

                else:

                    loss1 = F.cross_entropy(scores1.float(),jigsaw_labels.max(dim=1)[1].long())
                    loss2 = F.cross_entropy(scores2.float(),jigsaw_labels.max(dim=1)[1].long())

                    batch_output["loss"] = self.lambda_wt * loss1 + (1 - self.lambda_wt) * loss2
                
                    batch_output["acc"] = ((scores1.max(dim=1)[1] == jigsaw_labels.max(dim=1)[1]).float().mean() + (scores2.max(dim=1)[1] == jigsaw_labels.max(dim=1)[1]).float().mean())/2


                self.memory_bank_t = self.memory_bank.clone().detach()
                self.memory_bank_t[batch_indices] = self.beta_wt * self.memory_bank_t[batch_indices] + (1-self.beta_wt) * images
                self.memory_bank = self.memory_bank_t.clone().detach()

            else:
                neg_ind = torch.cat(
                    [torch.cat([torch.arange(self.batch_size)[0:i],torch.arange(self.batch_size)[i+1:]]) for i in range(self.batch_size)]
                    ).view(self.batch_size,-1)
                
                neg_images = images[neg_ind]

                query = images.view(bs,1,-1).repeat(1,bs,1)

                key = torch.cat([patches.unsqueeze(1),neg_images],dim=1)

                scores = F.cosine_similarity(query,key,axis=-1)/0.07

                jigsaw_labels = torch.zeros(bs,scores.shape[1]).to(device)
                jigsaw_labels[:,0] = 1
                
                if "nce_loss" in self.args.image_pretrain_obj:
                    scores = torch.softmax(scores,dim=-1)
                    batch_output["loss"] = F.binary_cross_entropy(scores.float(), jigsaw_labels.float(),reduction='none').sum()/scores.shape[0]
                else:
                    batch_output["loss"] = F.cross_entropy(scores.float(),jigsaw_labels.max(dim=1)[1].long())
            
                batch_output["acc"] = (scores.max(dim=1)[1] == (torch.ones(bs,).long()*0)).float().mean()

        else:
            raise NotImplementedError

        return batch_output 

class ViewSSLModels(SSLModel):
    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self.num_patches = args.num_patches
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
            self.init_layers,
            self.block1,
            self.block2,
            self.block3,
            self.block4,
        )

        if self.d_model > 1024:
            self.max_f = 1024
        else:
            self.max_f = self.d_model

        if self.args.view_pretrain_obj != "none":
            self.model_type = self.args.view_pretrain_obj.split("_")[0]
        else:
            self.model_type = "var"

        self.input_dim = 256

        self.latent_dim = self.args.latent_dim
        self.mask_ninps = 1

        self.reduce = nn.Conv2d(6 * self.d_model, self.d_model, kernel_size = 1, stride = 1)

        if self.model_type == "det":

            decoder_network_layers = []

            decoder_network_layers.append(
                block(int(self.d_model), int(self.max_f), 2, 2, 0)
            )

            init_layer_dim = int(self.input_dim // 16)
            init_channel_dim = self.max_f

            while init_layer_dim < self.input_dim:
                decoder_network_layers.append( 
                    block(int(init_channel_dim), int(init_channel_dim // 2), 2, 2, 0),
                )
                init_layer_dim *= 2
                init_channel_dim = init_channel_dim / 2
            
            if self.stage != "pretrain":
                out_dim = 1
            else:
                out_dim = 3

            decoder_network_layers.append(
                block(int(init_channel_dim), out_dim, 3, 1, 1, "sigmoid", False, False),
            )

            self.decoder_network =  nn.Sequential(*decoder_network_layers)

            self.decoding = nn.Sequential(self.decoder_network)

            self.loss_type = "bce"

        else:
            
            decoder_network_layers = []
            decoder_network_layers.append(
                block(int(self.latent_dim), int(self.max_f), 4, 1, 0, "leakyrelu"),
            )
            
            init_layer_dim = 4
            init_channel_dim = self.max_f
            while init_layer_dim < self.input_dim:
                decoder_network_layers.append( 
                    block(int(init_channel_dim), int(init_channel_dim // 2), 4, 2, 1, "leakyrelu"),
                )
                init_layer_dim *= 2
                init_channel_dim = init_channel_dim / 2
            
            if self.stage=="pretrain":
                decoder_network_layers.append(
                    block(int(init_channel_dim), 3, 3, 1, 1, "tanh", False, False),
                )
            else:
                decoder_network_layers.append(
                    block(int(init_channel_dim), 1, 3, 1, 1, "sigmoid", False, False),
                )

            self.decoder_network = nn.Sequential(*decoder_network_layers)

            self.z_project = nn.Linear(self.d_model, 2*self.latent_dim)
            # self.reduce = nn.Linear((6-self.mask_ninps) * self.d_model, self.d_model)
            self.decoding = nn.Sequential(self.decoder_network, self.z_project)

            self.loss_type = "mse"

        if self.loss_type == "mse":
            self.criterion = torch.nn.MSELoss()
        elif self.loss_type == "bce":
            self.criterion = torch.nn.BCELoss()

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.n_synthesize = self.args.synthesizer_nlayers

        if self.n_synthesize > 0:
            synthesizer_layers = []
            for i in range(self.n_synthesize):
                synthesizer_layers.append(Resblock(self.d_model,self.d_model, 3, 1, 1))
            
            self.synthesizer = nn.Sequential(*synthesizer_layers)


        self.fusion = self.args.view_fusion_strategy

        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)

        self.det_fusion_strategy = "concat_fuse"
        assert self.det_fusion_strategy in ["concat_fuse", "mean"]        

        self.sigmoid = nn.Sigmoid()
        self.shared_params = list(self.image_network.parameters())
        self.pretrain_params = list(self.reduce.parameters()) + list(self.decoding.parameters()) + list(self.synthesizer.parameters())

    def reparameterize(self, mu, logvar):

        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    

    def forward(self, batch_input, task=None):
        batch_output = {}
        
        index = batch_input["idx"]

        device = index.device#batch_input["aug"].device
        bs = index.size(0)
        self.batch_size = bs
        self.stage = "pretrain"

        if "masked" in self.args.view_pretrain_obj or "denoising" in self.args.view_pretrain_obj:
            mask = torch.cat([torch.arange(6)+1 for i in range(bs)]).view(bs,6)
            mask_indices = mask.clone()
            
            query_views = batch_input["image"]
            if "masked" in self.args.view_pretrain_obj:
                key_views = torch.zeros(bs,self.mask_ninps,3,self.input_dim,self.input_dim)
            else:
                key_views = batch_input["image"]

            for index in range(bs):
                mask = torch.arange(6) + 1
                mask[np.random.choice(6,self.mask_ninps)] = 0
                mask_indices = torch.arange(6) + 1

                neg_mask = mask_indices[(mask==0).nonzero()].view(-1)-1
                # print(pos_mask)
                # print(neg_mask)
                if "masked" in self.args.view_pretrain_obj:
                    # print(query_views[index,neg_mask].shape)
                    key_views[index] = query_views[index,neg_mask]

                query_views[index,neg_mask] = 0
                # key_view[index] = views[index,neg_mask]


        else:
            query_views = batch_input["image"]
            key_views = batch_input["image"]

        views = self.image_network(query_views.flatten(0,1))
        _,c,h,w = views.shape
        views = views.view(bs,6,c,h,w)

        if self.det_fusion_strategy == "concat_fuse":
            fusion = self.reduce(views.flatten(1,2))

        elif self.det_fusion_strategy == "mean":
            fusion = views.mean(dim=1)

        if self.n_synthesize > 0:
            fusion = self.synthesizer(fusion)

        if "det" in self.model_type:
            if "masked" in self.args.view_pretrain_obj:

                mapped_image = torch.zeros(bs,self.mask_ninps,3,self.input_dim, self.input_dim)
                for i in range(self.mask_ninps):
                    mapped_image[i] = self.decoder_network(fusion)

                batch_output["loss"] = self.criterion(mapped_image, key_views)
                
                batch_output["acc"] = (mapped_image == key_views).float().mean()

            elif "denoising" in self.args.view_pretrain_obj:

                mapped_image = torch.zeros(batch_input["input"].shape)

                for i in range(6):
                    mapped_image[i] = self.decoder_network(fusion)

                batch_output["loss"] = self.criterion(mapped_image, key_views)
                
                batch_output["acc"] = (mapped_image == key_views).float().mean()

        else:

            fusion = self.avg_pool(fusion).view(bs,self.d_model)

            mu_logvar = self.z_project(fusion).view(bs,2,-1)

            mu = mu_logvar[:,0]
            logvar = mu_logvar[:,1]

            z = self.reparameterize(mu,logvar).view(bs,self.latent_dim,1,1)

            generated_image = self.decoder_network(z)
            # print(generated_image.shape, mu.shape, logvar.shape)

            reconstruction_loss = self.criterion(generated_image, key_views)
            kl_divergence_loss = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

            batch_output["loss"] = reconstruction_loss + kl_divergence_loss
            batch_output["recon_loss"] = reconstruction_loss
            batch_output["KLD_loss"] = kl_divergence_loss
            batch_output["acc"] = (generated_image == key_views).float().mean()

        return batch_output


    





