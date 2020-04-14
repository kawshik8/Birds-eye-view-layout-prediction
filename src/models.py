# This file implements models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models.resnet as resnet
import itertools
from itertools import permutations,combinations
from torch.nn.modules.module import Module

def get_model(name, args):
    if name == "selfie":
        return SelfieModel(args)
    # elif name == "selfie1":
    #     return SelfieModel_revised(args)
    elif name == "baseline":
        return BaselineModel(args)
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

class BaselineModel(JigsawModel):
    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self.num_patches = args.num_patches
        self.d_model = 1024

        full_resnet = resnet.resnet50()
        self.patch_network = nn.Sequential(
            full_resnet.conv1,
            full_resnet.bn1,
            full_resnet.relu,
            full_resnet.maxpool,
            full_resnet.layer1,
            full_resnet.layer2,
            full_resnet.layer3,
        )

        self.res_block4 = full_resnet.layer4

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.reduce = nn.Linear(self.num_patches*self.d_model, self.d_model)
        self.project = nn.Linear(self.d_model, 128)

        self.pretrain_network = nn.Sequential(
            self.patch_network,
            self.avg_pool,
        )

        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)

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
        self.pretrain_params = list(self.reduce.parameters())
        self.finetune_params = list(self.cls_classifiers.parameters())
        if self.linear is False:
            #self.finetune_params = list(self.cls_classifiers.parameters())
            self.finetune_params += list(self.finetune_conv_layer.parameters())
        else:
            #self.finetune_params = list(self.linear_classifiers.parameters())
            self.finetune_params += list(self.dropout.parameters())

    def forward(self, batch_input, task=None):
        batch_output = {}
        
        inp = batch_input["image"]

        device = inp.device#batch_input["aug"].device
        bs = inp.size(0)

        if self.stage == "pretrain":

            patches = self.pretrain_network(inp.flatten(0, 1)).view(
                bs, self.num_patches, -1
            )  # (bs, num_patches, d_model)
           
            flatten = patches.view(bs,-1)
            final = self.project(self.reduce(flatten))

            if self.pretrain_obj == "nce_loss" or self.pretrain_obj == "crossentropy_loss":
                final = final.view(self.batch_size,self.dup_pos+1,self.d_model)

                c = list(combinations(torch.arange(self.dup_pos+1), 2))
                pos_ind = torch.Tensor([list(i) for i in c]).long()
                neg_ind = torch.cat(
                    [torch.cat([torch.arange(self.batch_size)[0:i],torch.arange(self.batch_size)[i+1:]]) for i in range(self.batch_size)]
                    ).view(self.batch_size,-1).unsqueeze(1).repeat(1,pos_ind.shape[0],1)
                negatives = final[neg_ind].view(self.batch_size,pos_ind.shape[0],-1,self.d_model)
                positives = final[:,pos_ind]
                final = torch.cat([positives,negatives],dim = 2)
                
                query = final[:,:,0:1,:].view(-1,1,self.d_model)
                key = final[:,:,1:,:].view(query.shape[0],-1,self.d_model)

                jigsaw_pred = F.cosine_similarity(query,key,axis=-1)/0.07
                if self.pretrain_obj == "nce_loss":
                    jigsaw_pred = F.softmax(jigsaw_pred,1)

                jigsaw_labels = torch.zeros(jigsaw_pred.shape[1]).long().unsqueeze(0).repeat(jigsaw_pred.shape[0],1).to(device)
                jigsaw_labels[:,0] = 1

                randperm = torch.cat([torch.randperm(jigsaw_pred.shape[1]) for i in range(jigsaw_pred.shape[0])]).view_as(jigsaw_labels)
                jigsaw_pred = jigsaw_pred[:,randperm][0]
                jigsaw_labels = jigsaw_labels[:,randperm][0]


                if self.pretrain_obj == "nce_loss":
                    batch_output["loss"] = F.binary_cross_entropy(jigsaw_pred.float(), jigsaw_labels.float(),reduction='none').sum()/jigsaw_pred.shape[0]
                else:
                    batch_output["loss"] = F.cross_entropy(jigsaw_pred.float(),jigsaw_labels.max(dim=1)[1].long())

                batch_output["jigsaw_acc"] = (jigsaw_pred.max(dim=1)[1] == jigsaw_labels.max(dim=1)[1]).float().mean()#F.nll_loss(F.log_softmax(jigsaw_pred,1), jigsaw_labels.max(dim=1)[1])#(jigsaw_pred.max(dim=1)[1] == jigsaw_labels.max(dim=1)[1]).float().mean()
            
            elif self.pretrain_obj == "multilabel_loss":
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

class SelfieModel(JigsawModel):
    def __init__(self, args, task = None):
        super().__init__(args)

        self.args = args
        self.num_patches = args.num_patches
        self.num_queries = args.num_queries
        self.num_context = self.num_patches - self.num_queries
        self.pretrain_obj = args.pretrain_obj
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        full_resnet = resnet.resnet50()
        self.patch_network = nn.Sequential(
            full_resnet.conv1,
            full_resnet.bn1,
            full_resnet.relu,
            full_resnet.maxpool,
            full_resnet.layer1,
            full_resnet.layer2,
            full_resnet.layer3,
        )

        self.pretrain_network = nn.Sequential(
            self.patch_network,
            self.avg_pool
        )

        self.finetune_conv_layer = full_resnet.layer4

        self.d_model = 1024
        self.f_model = 2048
        self.f1 = 256
        self.f2 = 1024
        self.f3 = 512

        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)

        self.attention_pool_u0 = nn.Parameter(torch.rand(size = (self.d_model,), dtype = torch.float, requires_grad=True))
        #print(self.attention_pool_u0.shape)
        transformer_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=32, dropout=0.1, dim_feedforward=640, activation='gelu')
        layer_norm = nn.LayerNorm(normalized_shape=self.d_model)
        self.attention_pooling = nn.TransformerEncoder(
            encoder_layer=transformer_layer, num_layers=3, norm=layer_norm
        )
        self.position_embedding = nn.Embedding(self.num_patches, self.d_model)
        self.cls_classifiers = nn.ModuleDict()
        self.resize_dim = {}
        #self.linear_classifiers = nn.ModuleDict()

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

        self.shared_params = list(self.patch_network.parameters())
        self.pretrain_params = list(self.position_embedding.parameters())
        self.pretrain_params += list(self.attention_pooling.parameters())
        self.pretrain_params += [self.attention_pool_u0]
        self.finetune_params = list(self.cls_classifiers.parameters())

        if self.linear is False:
            #self.finetune_params = list(self.cls_classifiers.parameters())
            self.finetune_params += list(self.finetune_conv_layer.parameters())
        else:
            #self.finetune_params = list(self.linear_classifiers.parameters())
            self.finetune_params += list(self.dropout.parameters())

    def forward(self, batch_input, task=None):
        batch_output = {}
        
        device = batch_input["image"].device
        bs = batch_input["image"].size(0)
        #print(self.attention_pool_u0.view(1,self.d_model).repeat(bs,1).shape)

        if self.stage == "pretrain":

            u0 = self.attention_pool_u0.view(1,self.d_model).repeat(bs,1).to(device)
            u0 = u0.to(device)
            patches = self.pretrain_network(batch_input["image"].flatten(0, 1)).view(
                bs, self.num_patches, -1
            )
            query_patch = masked_select(patches, batch_input["query"]).view(
                bs, self.num_queries, self.d_model
            )  # (bs, num_queries, d_model)
            visible_patch = (
                masked_select(patches, ~batch_input["query"])
                .view(bs, 1, self.num_context, self.d_model)
                .repeat(1, self.num_queries, 1, 1)
                .flatten(0, 1)
            )  # (bs * num_queries, num_context, d_model)
            pos_embeddings = self.position_embedding(
                torch.nonzero(batch_input["query"])[:, 1]
            ).view(
                bs, self.num_queries, self.d_model
            ) # (bs, num_queries, d_model)
         #   print(self.attention_pool_u0.shape,query_patch.shape)
            u0 = u0.view(bs,1,1,self.d_model).repeat(1,self.num_queries,1,1).flatten(0,1) # (bs * num_queries, 1, d_model)
            global_vector = self.attention_pooling(
                torch.cat([u0, visible_patch], dim=1).transpose(0,1)
            ).transpose(0,1)[:, 0, :].view_as(
                query_patch
            )  # (bs, num_queries, d_model)

            query_return = global_vector + pos_embeddings

            similarity = torch.bmm(
                query_patch, query_return.transpose(1, 2)
            )/(self.d_model**(1/2.0))  # (bs, num_queries, num_queries)
            #print(similarity[0])
            jigsaw_pred = F.log_softmax(similarity, 2).flatten(
                0, 1
            )  # (bs * num_queries, num_queries)
            jigsaw_label = (
                torch.arange(0, self.num_queries, device=device).repeat(bs).long()
            )  # (bs * num_queries)
            batch_output["loss"] = F.nll_loss(jigsaw_pred, jigsaw_label)
            batch_output["jigsaw_acc"] = (jigsaw_pred.max(dim=1)[1] == jigsaw_label).float().mean()

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

class AllPatchModel(JigsawModel):
    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self.dup_pos = args.dup_pos
        self.num_patches = args.num_patches
        self.num_queries = args.num_queries
        self.num_context = self.num_patches - self.num_queries
        self.d_model = 1024
        self.f_model = 2048

        full_resnet = resnet.resnet50()
        self.patch_network = nn.Sequential(
            full_resnet.conv1,
            full_resnet.bn1,
            full_resnet.relu,
            full_resnet.maxpool,
            full_resnet.layer1,
            full_resnet.layer2,
            full_resnet.layer3,
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.project = nn.Linear(self.d_model,128)

        self.pretrain_network = nn.Sequential(
            self.patch_network,
            self.avg_pool,
        )

        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)

        self.finetune_conv_layer = full_resnet.layer4

        self.attention_pool_u0 = nn.Parameter(torch.rand(size = (self.d_model,), dtype = torch.float, requires_grad=True))

        transformer_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=32, dropout=0.1, dim_feedforward=640, activation='gelu')
        layer_norm = nn.LayerNorm(normalized_shape=self.d_model)
        self.attention_pooling = nn.TransformerEncoder(
            encoder_layer=transformer_layer, num_layers=3, norm=layer_norm
        )
        self.position_embedding = nn.Embedding(self.num_patches, self.d_model)
        self.cls_classifiers = nn.ModuleDict()
        self.resize_dim = {}

        from tasks import task_num_class

        self.linear = False
        for taskname in args.finetune_tasks:
            name = taskname.split("_")[0]
            
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

        #self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.sigmoid = nn.Sigmoid()
        self.shared_params = list(self.patch_network.parameters())
        self.pretrain_params = list(self.attention_pooling.parameters())
        self.pretrain_params += [self.attention_pool_u0]
        self.pretrain_params += list(self.project.parameters())
        self.finetune_params = list(self.cls_classifiers.parameters())
        self.pretrain_obj = args.pretrain_obj
        self.batch_size = 0        

        if self.linear is False:
            #self.finetune_params = list(self.cls_classifiers.parameters())
            self.finetune_params += list(self.finetune_conv_layer.parameters())
        else:
            #self.finetune_params = list(self.linear_classifiers.parameters())
            self.finetune_params += list(self.dropout.parameters())

    def forward(self, batch_input, task=None):
        batch_output = {}
        
        inp = batch_input["image"]

        device = inp.device#batch_input["aug"].device
        bs = inp.size(0)
        self.batch_size = int(bs/(self.dup_pos+1))
        #print(inp.shape)
        # (bs, aug_patches, d_model)
        #final = self.avg_pool(attn_pool.transpose(1,2)).view(bs,self.d_model) # (bs, d_model)

        if self.stage == "pretrain":
            self.d_model = 1024
            u0 = self.attention_pool_u0.view(1,self.d_model).repeat(bs,1).to(device)
            u0 = u0.to(device)

            patches = self.pretrain_network(inp.flatten(0, 1)).view(
                bs, self.num_patches, -1
            )  # (bs, num_patches, d_model)
            u0 = u0.view(
                    bs,1,self.d_model
            ) # (bs, 1, d_model)
            final = self.project(self.attention_pooling(
                           torch.cat([u0, patches], dim=1).transpose(0,1)
                    )).transpose(0,1)[:,0,:] ### (bs,self.d_model)
            #final = self.project(final)
            self.d_model = 128
         #   print(bs,final.shape)

            if self.pretrain_obj == "nce_loss" or self.pretrain_obj == "crossentropy_loss":
                final = final.view(self.batch_size,self.dup_pos+1,self.d_model) ## (self.batch_size,dup_pos+1,self.d_model)

                c = list(combinations(torch.arange(self.dup_pos+1), 2)) ###((self.dup_pos+1)C2, 2)
                pos_ind = torch.Tensor([list(i) for i in c]).long()     ###((self.dup_pos+1)C2, 2)
                neg_ind = torch.cat(
                    [torch.cat([torch.arange(self.batch_size)[0:i],torch.arange(self.batch_size)[i+1:]]) for i in range(self.batch_size)]
                    ).view(self.batch_size,-1).unsqueeze(1).repeat(1,pos_ind.shape[0],1) ###(self.batch_size, (self.dup_pos+1)C2, self.batch_size-1)
                negatives = final[neg_ind].view(self.batch_size,pos_ind.shape[0],-1,self.d_model) ### (self.batch_size,(self.dup_pos+1)C2,(self.batch_size-1)*self.dup_pos,self.d_model)
                positives = final[:,pos_ind] ###(self.batch_size,(self.dup_pos+1)C2, 2, self.d_model)
                final = torch.cat([positives,negatives],dim = 2)###(self.batch_size,(self.dup_pos+1)C2, 2 + (self.batch_size-1)*self.dup_pos, self.d_model)
                
                query = final[:,:,0:1,:].view(-1,1,self.d_model) ###(self.batch_size*(self.dup_pos+1)C2, 1, self.d_model )
                key = final[:,:,1:,:].view(query.shape[0],-1,self.d_model)  ###(self.batch_size*(self.dup_pos+1)C2, 1 + (self.batch_size-1)*self.dup_pos, self.d_model )

                jigsaw_pred = F.cosine_similarity(query,key,axis=-1)/0.07 ###(self.batch_size*(self.dup_pos+1)C2, 1 + (self.batch_size-1)*self.dup_pos)
                if self.pretrain_obj == "nce_loss":
                    jigsaw_pred = F.softmax(jigsaw_pred,1)
                #jigsaw_pred = F.softmax(jigsaw_pred,1)

                jigsaw_labels = torch.zeros(jigsaw_pred.shape[1]).long().unsqueeze(0).repeat(jigsaw_pred.shape[0],1).to(device)
                jigsaw_labels[:,0] = 1

                # randperm = torch.randperm(jigsaw_pred.shape[1])#torch.cat([torch.randperm(jigsaw_pred.shape[1]) for i in range(jigsaw_pred.shape[0])]).view_as(jigsaw_labels)
                # jigsaw_pred = jigsaw_pred[:,randperm][0]
                # jigsaw_labels = jigsaw_labels[:,randperm][0]

            
                if self.pretrain_obj == "nce_loss":
                    batch_output["loss"] = F.binary_cross_entropy(jigsaw_pred.float(), jigsaw_labels.float(),reduction='none').sum()/jigsaw_pred.shape[0]
                else:
                    batch_output["loss"] = F.cross_entropy(jigsaw_pred.float(), jigsaw_labels.max(dim=1)[1].long())
                
                batch_output["jigsaw_acc"] = (jigsaw_pred.max(dim=1)[1] == jigsaw_labels.max(dim=1)[1]).float().mean()#F.nll_loss(F.log_softmax(jigsaw_pred,1), jigsaw_labels.max(dim=1)[1])#(jigsaw_pred.max(dim=1)[1] == jigsaw_labels.max(dim=1)[1]).float().mean()
            
            elif self.pretrain_obj == "multilabel_loss":
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

#def get_f(taskname):
   # if taskname == "stl10"



