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
    if name == "selfie":
        return SelfieModel(args)
    # elif name == "selfie1":
    #     return SelfieModel_revised(args)
    elif name == "baseline":
        return ImageSSLModels(args)
    else:
        raise NotImplementedError

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

def get_resize_dim(layer):
    if layer == "res1":
        return 12
    elif layer == "res2":
        return 6
    elif layer == "res3":
        return 4
    elif layer == "res4":
        return 3
    elif layer == "res5":
        return 2

def flatten_dim(layer):
    #task,layer = tasklayer.split("_")
    
    if layer == "res1" or layer == "res2" or layer == "res4":
        return 9216
    else:
        return 8192

class JigsawModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.stage = None
        self.shared_params = []
        self.pretrain_params = []
        self.finetune_params = []

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
                "jigsaw_acc": float (1), jigsaw puzzle accuracy, only when pretrain
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
        self.stage = stage
        if stage == "pretrain":
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
        elif stage == "finetune":
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
            optimizer = optim.AdamW(param_groups)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                anneal_strategy="cos",
                total_steps=self.args.finetune_total_iters,
                pct_start=self.args.warmup_iters / self.args.finetune_total_iters,
                cycle_momentum=False,
                max_lr=self.args.finetune_learning_rate,
            )

        return optimizer, scheduler


def masked_select(inp, mask):
    return inp.flatten(0, len(mask.size()) - 1)[mask.flatten().nonzero()[:, 0]]

class ImageSSLModels(JigsawModel):
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

        self.project_dim = 128

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

        self.cls_classifiers = nn.ModuleDict()

        from tasks import task_num_class

        self.linear = False
        for taskname in args.finetune_tasks:
            name = taskname.split("_")[0]
            #tasklayer = "_".join([taskname.split("_")[0],taskname.split("_")[2]])
            #layer = taskname.split("_")[2]
            
            if len(taskname.split("_")) > 2 and taskname.split("_")[2]!="none":
                self.linear = True
                tasklayer = "_".join([taskname.split("_")[0],taskname.split("_")[2]])
                layer = taskname.split("_")[2]
                for layer in (taskname.split("_")[2:]):
                    print(tasklayer,flatten_dim(layer))
                    self.cls_classifiers[tasklayer] = nn.Linear(flatten_dim(layer), task_num_class(name))
                    self.resize_dim[layer] = get_resize_dim(layer)
            else:
                self.cls_classifiers[taskname.split("_")[0]] = nn.Linear(self.f_model, task_num_class(name))

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.sigmoid = nn.Sigmoid()
        self.shared_params = list(self.patch_network.parameters())
        #self.shared_params += list(self.attention_pooling.parameters())
        self.pretrain_params = list(self.reduce.parameters()) + list(self.project1.parameters()) + list(self.project2.parameters())
        self.finetune_params = list(self.cls_classifiers.parameters())

    def forward(self, batch_input, task=None):
        batch_output = {}
        
        index = batch_input["idx"]
        image = batch_input["image"]
        query = batch_input["query"]
        # print(inp.shape)

        device = image.device#batch_input["aug"].device
        bs = image.size(0)
        self.batch_size = bs

        if self.stage == "pretrain":

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
                        
                        batch_output["jigsaw_acc"] = ((scores1.max(dim=1)[1] == jigsaw_labels.max(dim=1)[1]).float().mean() + (scores2.max(dim=1)[1] == jigsaw_labels.max(dim=1)[1]).float().mean())/2

                    else:

                        loss1 = F.cross_entropy(scores1.float(),jigsaw_labels.max(dim=1)[1].long())
                        loss2 = F.cross_entropy(scores2.float(),jigsaw_labels.max(dim=1)[1].long())

                        batch_output["loss"] = self.lambda_wt * loss1 + (1 - self.lambda_wt) * loss2
                 
                        batch_output["jigsaw_acc"] = ((scores1.max(dim=1)[1] == jigsaw_labels.max(dim=1)[1]).float().mean() + (scores2.max(dim=1)[1] == jigsaw_labels.max(dim=1)[1]).float().mean())/2


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
                    
                    if "nce_loss" in elf.args.image_pretrain_obj:
                        scores = torch.softmax(scores,dim=-1)
                        batch_output["loss"] = F.binary_cross_entropy(scores.float(), jigsaw_labels.float(),reduction='none').sum()/scores.shape[0]
                    else:
                        batch_output["loss"] = F.cross_entropy(scores.float(),jigsaw_labels.max(dim=1)[1].long())
                
                    batch_output["jigsaw_acc"] = (scores.max(dim=1)[1] == (torch.ones(bs,).long()*0)).float().mean()

                # neg_patches = patches[neg_ind]
                # negatives = torch.cat([neg_images,neg_pat])
            
            elif self.args.image_pretrain_obj == "multilabel_loss":
                jigsaw_pred = torch.mm(
                final, final.transpose(0,1)
                )/(self.d_model**(1/2.0))  # (bs, bs)
                
                #jigsaw_pred = self.sigmoid(similarity) # (bs , bs)

                jigsaw_label = torch.zeros(size=(bs,bs),dtype=torch.float).to(device)
                for i in range(bs):
                    
                    indices = torch.arange(int((i/self.dup_pos))*self.dup_pos,int(((i/self.dup_pos))+1)*self.dup_pos).type(torch.long).to(device)
                    #### Creates an array of size self.dup_pos_patches 
                    jigsaw_label[i] = jigsaw_label[i].scatter_(dim=0, index=indices, value=1.)
                    #### Makes the indices of jigsaw_labels (array of zeros) 1 based on the labels in indices

                batch_output["loss"] = F.binary_cross_entropy_with_logits(jigsaw_pred, jigsaw_label)#F.cross_entropy(jigsaw_pred, jigsaw_label)
                ones = torch.ones(jigsaw_pred.shape).to(device)
                zeros = torch.zeros(jigsaw_pred.shape).to(device)
                jigsaw_pred = torch.where(jigsaw_pred>0.5,ones,zeros)
                batch_output["jigsaw_acc"] = ((jigsaw_pred) == jigsaw_label).float().mean()

            else:
                raise NotImplementedError

        elif self.stage == "finetune":
            if self.linear:
                name = task.name
                taskname = name.split("_")[0]
                tasklayer = "_".join([name.split("_")[0],name.split("_")[2]])
                layer = name.split("_")[2]
                features = get_part_model(self.patch_network,layer)(batch_input["image"])
                #print(layer,features.shape,self.resize_dim[layer])
                resize = F.interpolate(features,size=(self.resize_dim[layer],self.resize_dim[layer])).view(bs,-1)
                dropout = self.dropout(resize)
                
                #print(layer,dropout.shape)
                cls_pred = self.cls_classifiers[tasklayer](dropout)
                batch_output["loss"] = F.cross_entropy(cls_pred, batch_input["label"])
                batch_output["predict"] = cls_pred.max(dim=1)[1]
                batch_output["cls_acc"] = (
                    (batch_output["predict"] == batch_input["label"]).float().mean()
                )
            else:
                features = self.patch_network(batch_input["image"])
                features = self.finetune_conv_layer(features)
                features_pool = self.avg_pool(features).view(bs,-1)
                dropout = self.dropout(features_pool)
                #print(features_pool.shape)
                cls_pred = self.cls_classifiers[task.name.split("_")[0]](dropout)
                batch_output["loss"] = F.cross_entropy(cls_pred, batch_input["label"])
                batch_output["predict"] = cls_pred.max(dim=1)[1]
                batch_output["cls_acc"] = (
                    (batch_output["predict"] == batch_input["label"]).float().mean()
                )

        return batch_output 


class ViewSSLModels(JigsawModel):
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

        self.project_dim = 128

        self.reduce = nn.Linear(self.num_patches*self.d_model, self.d_model)
        self.project = nn.Linear(self.d_model, self.project_dim)

        self.fusion = self.args.view_fusion_strategy

        self.pretrain_network = nn.Sequential(
            self.patch_network,
            self.avg_pool,
        )

        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)

        # self.use_memory_bank = True

        # if self.use_memory_bank:
        #     self.register_buffer("memory_bank", torch.randn(self.args.vocab_size, self.project_dim))
        #     self.all_indices = np.arange(self.args.vocab_size)

        # self.negatives = self.args.num_negatives
        # self.lambda_wt = self.args.lambda_pirl
        # self.beta_wt = self.args.beta_ema

        self.cls_classifiers = nn.ModuleDict()

        from tasks import task_num_class

        self.linear = False
        for taskname in args.finetune_tasks:
            name = taskname.split("_")[0]
            #tasklayer = "_".join([taskname.split("_")[0],taskname.split("_")[2]])
            #layer = taskname.split("_")[2]
            
            if len(taskname.split("_")) > 2 and taskname.split("_")[2]!="none":
                self.linear = True
                tasklayer = "_".join([taskname.split("_")[0],taskname.split("_")[2]])
                layer = taskname.split("_")[2]
                for layer in (taskname.split("_")[2:]):
                    print(tasklayer,flatten_dim(layer))
                    self.cls_classifiers[tasklayer] = nn.Linear(flatten_dim(layer), task_num_class(name))
                    self.resize_dim[layer] = get_resize_dim(layer)
            else:
                self.cls_classifiers[taskname.split("_")[0]] = nn.Linear(self.f_model, task_num_class(name))

        self.mask_ninps = 1
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.sigmoid = nn.Sigmoid()
        self.shared_params = list(self.patch_network.parameters())
        #self.shared_params += list(self.attention_pooling.parameters())
        self.pretrain_params = list(self.reduce.parameters()) + list(self.project1.parameters()) + list(self.project2.parameters())
        self.finetune_params = list(self.cls_classifiers.parameters())

    def forward(self, batch_input, task=None):
        batch_output = {}
        
        index = batch_input["idx"]
        if "masked_autoencoder" in self.args.image_pretrain_obj:
            mask = torch.ones(6)
            mask[np.random.choice(6,self.mask_ninps)] = 0
            views = batch_input["image"]

            query_views = torch.masked_select

        # mask_view = 
        # query = batch_input["query"]
        # print(inp.shape)

        device = image.device#batch_input["aug"].device
        bs = image.size(0)
        self.batch_size = bs

        if self.stage == "pretrain":

            views = self.project(self.pretrain_network(views.flatten(0,1))).view(
                bs, 6, -1
            )

            if "masked_autoencoder" in self.args.image_pretrain_obj:

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
                        
                        batch_output["jigsaw_acc"] = ((scores1.max(dim=1)[1] == jigsaw_labels.max(dim=1)[1]).float().mean() + (scores2.max(dim=1)[1] == jigsaw_labels.max(dim=1)[1]).float().mean())/2

                    else:

                        loss1 = F.cross_entropy(scores1.float(),jigsaw_labels.max(dim=1)[1].long())
                        loss2 = F.cross_entropy(scores2.float(),jigsaw_labels.max(dim=1)[1].long())

                        batch_output["loss"] = self.lambda_wt * loss1 + (1 - self.lambda_wt) * loss2
                 
                        batch_output["jigsaw_acc"] = ((scores1.max(dim=1)[1] == jigsaw_labels.max(dim=1)[1]).float().mean() + (scores2.max(dim=1)[1] == jigsaw_labels.max(dim=1)[1]).float().mean())/2


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
                
                    batch_output["jigsaw_acc"] = (scores.max(dim=1)[1] == (torch.ones(bs,).long()*0)).float().mean()

                # neg_patches = patches[neg_ind]
                # negatives = torch.cat([neg_images,neg_pat])
            
            elif self.args.pretrain_obj == "multilabel_loss":
                jigsaw_pred = torch.mm(
                final, final.transpose(0,1)
                )/(self.d_model**(1/2.0))  # (bs, bs)
                
                #jigsaw_pred = self.sigmoid(similarity) # (bs , bs)

                jigsaw_label = torch.zeros(size=(bs,bs),dtype=torch.float).to(device)
                for i in range(bs):
                    
                    indices = torch.arange(int((i/self.dup_pos))*self.dup_pos,int(((i/self.dup_pos))+1)*self.dup_pos).type(torch.long).to(device)
                    #### Creates an array of size self.dup_pos_patches 
                    jigsaw_label[i] = jigsaw_label[i].scatter_(dim=0, index=indices, value=1.)
                    #### Makes the indices of jigsaw_labels (array of zeros) 1 based on the labels in indices

                batch_output["loss"] = F.binary_cross_entropy_with_logits(jigsaw_pred, jigsaw_label)#F.cross_entropy(jigsaw_pred, jigsaw_label)
                ones = torch.ones(jigsaw_pred.shape).to(device)
                zeros = torch.zeros(jigsaw_pred.shape).to(device)
                jigsaw_pred = torch.where(jigsaw_pred>0.5,ones,zeros)
                batch_output["jigsaw_acc"] = ((jigsaw_pred) == jigsaw_label).float().mean()

            else:
                raise NotImplementedError

        elif self.stage == "finetune":
            if self.linear:
                name = task.name
                taskname = name.split("_")[0]
                tasklayer = "_".join([name.split("_")[0],name.split("_")[2]])
                layer = name.split("_")[2]
                features = get_part_model(self.patch_network,layer)(batch_input["image"])
                #print(layer,features.shape,self.resize_dim[layer])
                resize = F.interpolate(features,size=(self.resize_dim[layer],self.resize_dim[layer])).view(bs,-1)
                dropout = self.dropout(resize)
                
                #print(layer,dropout.shape)
                cls_pred = self.cls_classifiers[tasklayer](dropout)
                batch_output["loss"] = F.cross_entropy(cls_pred, batch_input["label"])
                batch_output["predict"] = cls_pred.max(dim=1)[1]
                batch_output["cls_acc"] = (
                    (batch_output["predict"] == batch_input["label"]).float().mean()
                )
            else:
                features = self.patch_network(batch_input["image"])
                features = self.finetune_conv_layer(features)
                features_pool = self.avg_pool(features).view(bs,-1)
                dropout = self.dropout(features_pool)
                #print(features_pool.shape)
                cls_pred = self.cls_classifiers[task.name.split("_")[0]](dropout)
                batch_output["loss"] = F.cross_entropy(cls_pred, batch_input["label"])
                batch_output["predict"] = cls_pred.max(dim=1)[1]
                batch_output["cls_acc"] = (
                    (batch_output["predict"] == batch_input["label"]).float().mean()
                )
        return batch_output 






