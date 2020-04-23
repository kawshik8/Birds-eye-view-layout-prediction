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
    print(args.image_pretrain_obj, args.view_pretrain_obj)
    if name == "selfie":
        return SelfieModel(args)
    # elif name == "selfie1":
    #     return SelfieModel_revised(args)
    
    elif args.image_pretrain_obj != "none":
        return ImageSSLModels(args)
    elif args.view_pretrain_obj != "none":
        return ViewSSLModels(args)
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

class block(nn.Module):
    
    def __init__(self, input_channels, output_channels, kernel, strides, pad, activation="relu", norm = True, use_transpose=True):
        super().__init__()

        if use_transpose:
            self.conv = nn.ConvTranspose2d(in_channels = input_channels, out_channels = output_channels, kernel_size = kernel, stride = strides, padding = pad)
        else:
            self.conv = nn.Conv2d(in_channels = input_channels, out_channels = output_channels, kernel_size = kernel, stride = strides, padding = pad)

        self.use_norm = norm

        if norm:
            self.bn = nn.BatchNorm2d(output_channels)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU(0.2,inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x):

        x = self.conv(x)

        if self.use_norm:
            x = self.bn(x)

        x = self.activation(x)

        return x

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
                    
                    if "nce_loss" in elf.args.image_pretrain_obj:
                        scores = torch.softmax(scores,dim=-1)
                        batch_output["loss"] = F.binary_cross_entropy(scores.float(), jigsaw_labels.float(),reduction='none').sum()/scores.shape[0]
                    else:
                        batch_output["loss"] = F.cross_entropy(scores.float(),jigsaw_labels.max(dim=1)[1].long())
                
                    batch_output["acc"] = (scores.max(dim=1)[1] == (torch.ones(bs,).long()*0)).float().mean()

                # neg_patches = patches[neg_ind]
                # negatives = torch.cat([neg_images,neg_pat])
            
            # elif self.args.image_pretrain_obj == "multilabel_loss":
            #     jigsaw_pred = torch.mm(
            #     final, final.transpose(0,1)
            #     )/(self.d_model**(1/2.0))  # (bs, bs)
                
            #     #jigsaw_pred = self.sigmoid(similarity) # (bs , bs)

            #     jigsaw_label = torch.zeros(size=(bs,bs),dtype=torch.float).to(device)
            #     for i in range(bs):
                    
            #         indices = torch.arange(int((i/self.dup_pos))*self.dup_pos,int(((i/self.dup_pos))+1)*self.dup_pos).type(torch.long).to(device)
            #         #### Creates an array of size self.dup_pos_patches 
            #         jigsaw_label[i] = jigsaw_label[i].scatter_(dim=0, index=indices, value=1.)
            #         #### Makes the indices of jigsaw_labels (array of zeros) 1 based on the labels in indices

            #     batch_output["loss"] = F.binary_cross_entropy_with_logits(jigsaw_pred, jigsaw_label)#F.cross_entropy(jigsaw_pred, jigsaw_label)
            #     ones = torch.ones(jigsaw_pred.shape).to(device)
            #     zeros = torch.zeros(jigsaw_pred.shape).to(device)
            #     jigsaw_pred = torch.where(jigsaw_pred>0.5,ones,zeros)
            #     batch_output["acc"] = ((jigsaw_pred) == jigsaw_label).float().mean()

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
        self.image_network = nn.Sequential(
            full_resnet.conv1,
            full_resnet.bn1,
            full_resnet.relu,
            full_resnet.maxpool,
            full_resnet.layer1,
            full_resnet.layer2,
            full_resnet.layer3,
            full_resnet.layer4,
        )

        if self.d_model > 1024:
            self.max_f = 1024
        else:
            self.max_f = self.d_model

        self.model_type = self.args.view_pretrain_obj.split("_")[0]
        self.input_dim = 256

        self.project_dim = 128
        self.mask_ninps = 1

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
            
            decoder_network_layers.append(
                block(int(init_channel_dim), 3, 3, 1, 1, "sigmoid", False, False),
            )

            self.decoder_network =  nn.Sequential(*decoder_network_layers)

            self.reduce = nn.Conv2d((6-self.mask_ninps) * self.d_model, self.d_model, kernel_size = 1, stride = 1)
            self.decoding = nn.Sequential(self.decoder_network, self.reduce)

            self.loss_type = "mse"

        else:
            
            decoder_network_layers = []
            decoder_network_layers.append(
                block(int(self.project_dim), int(self.max_f), 4, 1, 0, "leakyrelu"),
            )
            
            init_layer_dim = 4
            init_channel_dim = self.max_f
            while init_layer_dim < self.input_dim:
                decoder_network_layers.append( 
                    block(int(init_channel_dim), int(init_channel_dim // 2), 4, 2, 1, "leakyrelu"),
                )
                init_layer_dim *= 2
                init_channel_dim = init_channel_dim / 2
            
            decoder_network_layers.append(
                block(int(init_channel_dim), 3, 3, 1, 1, "tanh", False, False),
            )

            self.decoder_network = nn.Sequential(*decoder_network_layers)

            self.z_project = nn.Linear(self.d_model, 2*self.project_dim)
            self.reduce = nn.Linear((6-self.mask_ninps) * self.d_model, self.d_model)
            self.decoding = nn.Sequential(self.decoder_network, self.z_project, self.reduce)

            self.loss_type = "mse"

        if self.loss_type == "mse":
            self.criterion = torch.nn.MSELoss()
        elif self.loss_type == "bce":
            self.criterion = torch.nn.BCELoss()

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fusion = self.args.view_fusion_strategy

        self.pretrain_network = nn.Sequential(
            self.image_network,
            self.avg_pool,
            self.decoding,
        )

        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)

        # self.use_memory_bank = True

        # if self.use_memory_bank:
        #     self.register_buffer("memory_bank", torch.randn(self.args.vocab_size, self.project_dim))
        #     self.all_indices = np.arange(self.args.vocab_size)

        # self.negatives = self.args.num_negatives
        # self.lambda_wt = self.args.lambda_pirl
        # self.beta_wt = self.args.beta_ema
        self.det_fusion_strategy = "concat_fuse"
        assert self.det_fusion_strategy in ["concat_fuse", "mean"]

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

        

        self.sigmoid = nn.Sigmoid()
        self.shared_params = list(self.pretrain_network.parameters())
        #self.shared_params += list(self.attention_pooling.parameters())
        # self.pretrain_params = list(self.reduce.parameters()) + list(self.project1.parameters()) + list(self.project2.parameters())
        self.finetune_params = list(self.cls_classifiers.parameters())

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

        if "masked_autoencoder" in self.args.view_pretrain_obj:
            mask = torch.cat([torch.arange(6)+1 for i in range(bs)]).view(bs,6)
            mask_indices = mask.clone()
            
            views = batch_input["image"]

            query_views = torch.zeros(bs, 6 - self.mask_ninps, 3, self.input_dim, self.input_dim)
            key_view = torch.zeros(bs, self.mask_ninps, 3, self.input_dim, self.input_dim)
            for index in range(bs):
                mask = torch.arange(6) + 1
                mask[np.random.choice(6,self.mask_ninps)] = 0
                mask_indices = torch.arange(6) + 1

                pos_mask = mask_indices[(mask!=0).nonzero()].view(-1)-1
                neg_mask = mask_indices[(mask==0).nonzero()].view(-1)-1
                # print(pos_mask)
                # print(neg_mask)
                query_views[index] = views[index,pos_mask]
                key_view[index] = views[index,neg_mask]

            key_view = key_view.flatten(0,1)
            print(key_view.shape)
            print(query_views.shape)

        else:
            query_views = batch_input["image"]

        views = self.image_network(query_views.flatten(0,1))
        _,c,h,w = views.shape
        views = views.view(bs,6 - self.mask_ninps,c,h,w)
        print(views.shape)

        if "det" in self.model_type:

            if self.det_fusion_strategy == "concat_fuse":
                print(views.flatten(1,2).shape)
                fusion = self.reduce(views.flatten(1,2))

            elif self.det_fusion_strategy == "mean":
                fusion = views.mean(dim=1)
            print(fusion.shape)

            mapped_image = self.decoder_network(fusion)
            print(mapped_image.shape, key_view.shape)

            batch_output["loss"] = self.criterion(mapped_image, key_view)
            
            batch_output["acc"] = (mapped_image == key_view).float().mean()

        else:

            print("VAE")

            views = self.avg_pool(views.flatten(0,1)).view(bs, 6 - self.mask_ninps,-1) 

            print(views.shape)

            if self.det_fusion_strategy == "concat_fuse":
                fusion = self.reduce(views.flatten(1,2))

            elif self.det_fusion_strategy == "mean":
                fusion = views.mean(dim=1)

            mu_logvar = self.z_project(fusion).view(bs,2,-1)

            mu = mu_logvar[:,0]
            logvar = mu_logvar[:,1]

            z = self.reparameterize(mu,logvar).view(bs,self.project_dim,1,1)

            generated_image = self.decoder_network(z)
            print(generated_image.shape, mu.shape, logvar.shape)

            reconstruction_loss = self.criterion(generated_image, key_view)
            kl_divergence_loss = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

            batch_output["loss"] = reconstruction_loss + kl_divergence_loss
            batch_output["recon_loss"] = reconstruction_loss
            batch_output["KLD_loss"] = kl_divergence_loss
            batch_output["acc"] = (generated_image == key_view).float().mean()

        return batch_output

       





