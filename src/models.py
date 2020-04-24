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
from utils import Anchors, BBoxTransform, ClipBoxes
import losses


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
    if args.image_pretrain_obj != "none":
        return ImageSSLModels(args)
    else:
        return ViewSSLModels(args)
    

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

class Resblock(nn.Module):
    
    def __init__(self, input_channels, output_channels, kernel, strides, pad, activation="relu", norm = True):
        super().__init__()

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

    def forward(self, x1):

        x = self.bn(x1)
        x = self.activation(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.conv(x)

        x+=x1
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

class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Softmax(dim = 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class ViewSSLModels(JigsawModel):
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

        self.obj_detection_head = "retinanet"
        self.num_classes = 9

        self.gen_roadmap = True
        self.detect_objects = False

        if "retinanet" in self.obj_detection_head and self.detect_objects:
            if "resnet18" in self.args.network_base or "resnet34" in self.args.network_base:
                fpn_sizes = [self.block2[-1].conv2.out_channels, self.block3[-1].conv2.out_channels,
                            self.block4[-1].conv2.out_channels]
            else:
                fpn_sizes = [self.block2[-1].conv3.out_channels, self.block3[-1].conv3.out_channels,
                            self.block4[-1].conv3.out_channels]
        
            self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

            self.regressionModel = RegressionModel(256)
            self.classificationModel = ClassificationModel(256, num_classes=self.num_classes)
            
            self.anchors = Anchors()

            self.regressBoxes = BBoxTransform()

            self.clipBoxes = ClipBoxes()

            import losses

            self.focalLoss = losses.FocalLoss()

            prior = 0.01

            self.classificationModel.output.weight.data.fill_(0)
            self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

            self.regressionModel.output.weight.data.fill_(0)
            self.regressionModel.output.bias.data.fill_(0)

            self.obj_detection_model = nn.Sequential(
                self.fpn,
                self.regressionModel,
                self.classificationModel,
            )


        self.n_blobs = 3

        self.channel_projections = nn.ModuleDict()
        self.obj_detection_head = "retinanet"

        if self.d_model > 1024:
            self.max_f = 1024
        else:
            self.max_f = self.d_model

        if self.args.view_pretrain_obj != "none":
            self.model_type = self.args.view_pretrain_obj.split("_")[0]
        else:
            self.model_type = "var"

        self.input_dim = 256

        self.project_dim = 128
        self.mask_ninps = 1

        if self.stage=="pretrain":
            self.reduce = nn.Conv2d((6-self.mask_ninps) * self.d_model, self.d_model, kernel_size = 1, stride = 1)
        else:
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

            self.decoding = nn.Sequential(self.decoder_network, self.reduce)

            self.loss_type = "bce"

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
            
            if self.stage=="pretrain":
                decoder_network_layers.append(
                    block(int(init_channel_dim), 3, 3, 1, 1, "tanh", False, False),
                )
            else:
                decoder_network_layers.append(
                    block(int(init_channel_dim), 1, 3, 1, 1, "sigmoid", False, False),
                )

            self.decoder_network = nn.Sequential(*decoder_network_layers)

            self.z_project = nn.Linear(self.d_model, 2*self.project_dim)
            # self.reduce = nn.Linear((6-self.mask_ninps) * self.d_model, self.d_model)
            self.decoding = nn.Sequential(self.decoder_network, self.z_project, self.reduce)

            self.loss_type = "mse"

        if self.loss_type == "mse":
            self.criterion = torch.nn.MSELoss()
        elif self.loss_type == "bce":
            self.criterion = torch.nn.BCELoss()

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.n_synthesize = 1

        if self.n_synthesize > 0:
            synthesizer_layers = []
            for i in range(self.n_synthesize):
                synthesizer_layers.append(Resblock(self.d_model,self.d_model, 3, 1, 1))
            
            self.synthesizer = nn.Sequential(*synthesizer_layers)

            self.pretrain_network = nn.Sequential(
            self.image_network,
            self.synthesizer,
            self.decoding,
            )

            if self.detect_objects:
                self.finetune_network = nn.Sequential(
                    self.image_network,
                    self.synthesizer,
                    self.decoding,
                    self.obj_detection_model,
                    )

            else:
                self.finetune_network = nn.Sequential(
                self.image_network,
                self.synthesizer,
                self.decoding,
                )
        else:
            self.pretrain_network = nn.Sequential(
            self.image_network,
            self.decoding,
            )

            if self.detect_objects:
                self.finetune_network = nn.Sequential(
                    self.image_network,
                    self.decoding,
                    self.obj_detection_model,
                    )

            else:
                self.finetune_network = n.Sequential(
                self.image_network,
                self.decoding,
                )

        

        self.fusion = self.args.view_fusion_strategy

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
        

        self.sigmoid = nn.Sigmoid()
        self.pretrain_params = list(self.pretrain_network.parameters())
        #self.shared_params += list(self.attention_pooling.parameters())
        # self.pretrain_params = list(self.reduce.parameters()) + list(self.project1.parameters()) + list(self.project2.parameters())
        self.finetune_params = list(self.finetune_network.parameters())

    def reparameterize(self, mu, logvar):

        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def get_blobs(self, x):
        x = self.init_layers(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        layers = [x1,x2,x3,x4]

        return layers[-self.n_blobs:]
    

    def forward(self, batch_input, task=None):
        batch_output = {}
        
        index = batch_input["idx"]

        device = index.device#batch_input["aug"].device
        bs = index.size(0)
        self.batch_size = bs

        if self.stage == "pretrain":

            if "masked" in self.args.view_pretrain_obj or "denoising" in self.args.view_pretrain_obj:
                mask = torch.cat([torch.arange(6)+1 for i in range(bs)]).view(bs,6)
                mask_indices = mask.clone()
                
                query_views = batch_input["image"]
                if "masked" in self.args.view_pretrain_obj:
                    key_views = torch.zeros(bs,self.mask_inps,3,self.input_dim,self.input_dim)
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
                        key_views[index] = query_views[index,neg_mask]

                    query_views[index,neg_mask] = 0
                    # key_view[index] = views[index,neg_mask]

                print(query_views.shape)


            else:
                query_views = batch_input["image"]
                key_views = batch_input["image"]

            views = self.image_network(query_views.flatten(0,1))
            _,c,h,w = views.shape
            views = views.view(bs,6,c,h,w)
            print(views.shape)

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

                    print(mapped_image.shape, key_views.shape)

                    batch_output["loss"] = self.criterion(mapped_image, key_views)
                    
                    batch_output["acc"] = (mapped_image == key_visews).float().mean()

                elif "denoising" in self.args.view_pretrain_obj:

                    mapped_image = torch.zeros(batch_input["input"].shape)

                    for i in range(6):
                        mapped_image[i] = self.decoder_network(fusion)

                    print(mapped_image.shape, key_views.shape)

                    batch_output["loss"] = self.criterion(mapped_image, key_views)
                    
                    batch_output["acc"] = (mapped_image == key_views).float().mean()

            else:

                fusion = self.avg_pool(fusion).view(bs,self.d_model)

                mu_logvar = self.z_project(fusion).view(bs,2,-1)

                mu = mu_logvar[:,0]
                logvar = mu_logvar[:,1]

                z = self.reparameterize(mu,logvar).view(bs,self.project_dim,1,1)

                generated_image = self.decoder_network(z)
                # print(generated_image.shape, mu.shape, logvar.shape)

                reconstruction_loss = self.criterion(generated_image, key_view)
                kl_divergence_loss = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

                batch_output["loss"] = reconstruction_loss + kl_divergence_loss
                batch_output["recon_loss"] = reconstruction_loss
                batch_output["KLD_loss"] = kl_divergence_loss
                batch_output["acc"] = (generated_image == key_view).float().mean()

            return batch_output

        else:
            views = batch_input["image"]
            road_map = batch_input["road"]
            annotations = batch_input["bbox"]

            layers = self.get_blobs(views.flatten(0,1))

            final_features = layers[-1]
            print(final_features.shape)
            _,c,h,w = final_features.shape
            views = final_features.view(bs,6,c,h,w)

            if self.det_fusion_strategy == "concat_fuse":
                fusion = self.reduce(views.flatten(1,2))

            elif self.det_fusion_strategy == "mean":
                fusion = views.mean(dim=1)

            if self.n_synthesize > 0:
                fusion = self.synthesizer(fusion)

            batch_output["loss"] = 0
            
            if self.gen_roadmap:

                if "det" in self.model_type:

                    mapped_image = self.decoder_network(fusion)
                    print(mapped_image.shape,road_map.shape)

                    batch_output["recon_loss"] = self.criterion(mapped_image, road_map)
                    
                    batch_output["acc"] = (mapped_image == road_map).float().mean()
                    batch_output["loss"] += batch_output["recon_loss"]

                else:

                    fusion = self.avg_pool(fusion).view(bs,self.d_model)

                    mu_logvar = self.z_project(fusion).view(bs,2,-1)

                    mu = mu_logvar[:,0]
                    logvar = mu_logvar[:,1]

                    z = self.reparameterize(mu,logvar).view(bs,self.project_dim,1,1)

                    generated_image = self.decoder_network(z)
                    print(generated_image.shape, mu.shape, logvar.shape)

                    reconstruction_loss = self.criterion(generated_image, road_map)
                    kl_divergence_loss = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

                    batch_output["recon_loss"] = reconstruction_loss
                    batch_output["KLD_loss"] = kl_divergence_loss
                    batch_output["acc"] = (generated_image == road_map).float().mean()
                    batch_output["loss"] += batch_output["recon_loss"] + batch_output["KLD_loss"]

            if self.detect_objects:

                features = self.fpn(layers)

                regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

                classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

                anchors = self.anchors(img_batch)

                if self.training:
                    batch_output["classification_loss"], batch_output["detection_loss"] = self.focalLoss(classification, regression, anchors, annotations)
                    batch_output["loss"] += batch_output["classification_loss"] + batch_output["detection_loss"]
                else:
                    transformed_anchors = self.regressBoxes(anchors, regression)
                    transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

                    scores = torch.max(classification, dim=2, keepdim=True)[0]

                    scores_over_thresh = (scores > 0.05)[0, :, 0]

                    if scores_over_thresh.sum() == 0:
                        # no boxes to NMS, just return
                        return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

                    classification = classification[:, scores_over_thresh, :]
                    transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
                    scores = scores[:, scores_over_thresh, :]

                    anchors_nms_idx = nms(transformed_anchors[0,:,:], scores[0,:,0], 0.5)

                    nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

                    # return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]
            return batch_output


        





