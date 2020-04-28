# This file implements models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models.resnet as resnet
import itertools
import numpy as np
from torchvision.ops import nms
from itertools import permutations,combinations
from torch.nn.modules.module import Module
from utils import Anchors, BBoxTransform, ClipBoxes, block, Resblock
import losses
import math
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


def get_sup_model(name, args):
    return ViewGenModels(args)


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256, fusion_strategy=None):
        super(PyramidFeatures, self).__init__()

        self.fusion_strategy = fusion_strategy
        if self.fusion_strategy:
            self.fuse1 = nn.Conv2d(6*C3_size,C3_size, kernel_size = 1, stride = 1, padding = 0)
            self.fuse2 = nn.Conv2d(6*C4_size,C4_size, kernel_size = 1, stride = 1, padding = 0)
            self.fuse3 = nn.Conv2d(6*C5_size,C5_size, kernel_size = 1, stride = 1, padding = 0)

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

        # print("C3,C4,C5", C3.shape,C4.shape,C5.shape)

        if self.fusion_strategy is not None:
            
            b,c,h,w = C3.shape
            C3 = self.fuse1(C3.view(-1,6,c,h,w).flatten(1,2))
            b,c,h,w = C4.shape
            C4 = self.fuse2(C4.view(-1,6,c,h,w).flatten(1,2))
            b,c,h,w = C5.shape
            C5 = self.fuse3(C5.view(-1,6,c,h,w).flatten(1,2))

        # print("c3,c4,c5", C3.shape,C4.shape,C5.shape)

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
    def __init__(self, num_features_in, fused, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.fused = not fused 
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

        if not self.fused:
            b,c,w,h = out.shape
            out = out.view(-1, 6, c, w, h).flatten(1,2)

        # print("finalr", out.shape, "inputr", x.shape)
        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        # print("final1", out.shape)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, fused, num_anchors=9, num_classes=9, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.fused = not fused

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

        # b,c,h,w = out.shape
        # out = self.output_act(out.view(b, self.num_classes, self.num_anchors, h, w)).flatten(1,2)

        if not self.fused:
            b,c,w,h = out.shape
            out = out.view(-1, 6, c, w, h).flatten(1,2)

        # out is B x C x W x H, with C = n_classes + n_anchors
        # print("finalc", out.shape, "inputc", x.shape)

        out1 = out.permute(0, 2, 3, 1)

        # print("final", out1.shape)

        batch_size, width, height, channels = out1.shape

        if not self.fused:
            out2 = out1.view(batch_size, width, height, self.num_anchors*6, self.num_classes)
        else:
            out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        # print(out2.contiguous().view(batch_size, -1, self.num_classes).shape)
        return out2.contiguous().view(batch_size, -1, self.num_classes)

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

        self.obj_detection_head = self.args.obj_det_head
        self.num_classes = 9

        self.gen_roadmap = True
        self.detect_objects = self.args.detect_objects

        self.fusion = self.args.view_fusion_strategy

        self.n_blobs = 3
        

        self.channel_projections = nn.ModuleDict()
        self.obj_detection_head = "retinanet"

        if self.d_model > 1024:
            self.max_f = 1024
        else:
            self.max_f = self.d_model

        # print(self.args.finetune_obj)
        self.model_type = self.args.finetune_obj.split("_")[0]

        self.input_dim = 256
   
        self.reduce = nn.Conv2d(6 * self.d_model, self.d_model, kernel_size = 1, stride = 1)

        out_dim = 1

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
                block(int(init_channel_dim), out_dim, 3, 1, 1, "sigmoid", False, False),
            )

            self.decoder_network =  nn.Sequential(*decoder_network_layers)

            self.decoding = nn.Sequential(self.decoder_network, self.reduce)

            self.loss_type = "bce"

        else:
            
            self.latent_dim = self.args.latent_dim
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
            
            
            decoder_network_layers.append(
                block(int(init_channel_dim), 1, 3, 1, 1, "sigmoid", False, False),
            )

            self.decoder_network = nn.Sequential(*decoder_network_layers)

            self.z_project = nn.Linear(self.d_model, 2*self.latent_dim)
            # self.reduce = nn.Linear((6-self.mask_ninps) * self.d_model, self.d_model)
            self.decoding = nn.Sequential(self.decoder_network, self.z_project, self.reduce)

            self.loss_type = "mse"

        self.loss_type = self.args.road_map_loss
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

        else:
            self.pretrain_network = nn.Sequential(
            self.image_network,
            self.decoding,
            )

        self.blobs_strategy = self.args.blobs_strategy
        
        if "retinanet" in self.obj_detection_head and self.detect_objects:

            if "encoder" in self.blobs_strategy:
                if "resnet18" in self.args.network_base or "resnet34" in self.args.network_base:
                    fpn_sizes = [self.block2[-1].conv2.out_channels, self.block3[-1].conv2.out_channels,
                                self.block4[-1].conv2.out_channels]
                else:
                    fpn_sizes = [self.block2[-1].conv3.out_channels, self.block3[-1].conv3.out_channels,
                                self.block4[-1].conv3.out_channels]

            elif "decoder" in self.blobs_strategy:
                if "var" in self.model_type:
                    fpn_sizes = [self.decoder_network[3].conv.out_channels, self.decoder_network[2].conv.out_channels, self.decoder_network[1].conv.out_channels]
                else:
                    fpn_sizes = [self.decoder_network[1].conv.out_channels, self.decoder_network[0].conv.out_channels, self.synthesizer[-1].conv.out_channels ]


            if "encoder" in self.blobs_strategy and "fused" in self.blobs_strategy:
                self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fusion_strategy = "fused")
            else:
                self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
    
            self.dynamic_strategy = ("fused" not in self.blobs_strategy and "encoder" in self.blobs_strategy)
            # print("dynamic strat", self.dynamic_strategy)
            self.regressionModel = RegressionModel(256, self.dynamic_strategy)
            self.classificationModel = ClassificationModel(256, self.dynamic_strategy)
            
            self.anchors = Anchors()

            self.regressBoxes = BBoxTransform()

            self.clipBoxes = ClipBoxes()

            import losses

            self.focalLoss = losses.FocalLoss(self.dynamic_strategy)

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
        

        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)

        # self.use_memory_bank = True

        # if self.use_memory_bank:
        #     self.register_buffer("memory_bank", torch.randn(self.args.vocab_size, self.latent_dim))
        #     self.all_indices = np.arange(self.args.vocab_size)

        # self.negatives = self.args.num_negatives
        # self.lambda_wt = self.args.lambda_pirl
        # self.beta_wt = self.args.beta_ema

        

        self.sigmoid = nn.Sigmoid()
        self.shared_params = list(self.image_network.parameters())
        self.pretrain_params = list(self.reduce.parameters()) + list(self.decoding.parameters()) + list(self.synthesizer.parameters())
        #self.shared_params += list(self.attention_pooling.parameters())
        # self.pretrain_params = list(self.reduce.parameters()) + list(self.project1.parameters()) + list(self.project2.parameters())
        self.finetune_params = list(self.obj_detection_model.parameters())

    def reparameterize(self, mu, logvar):

        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def get_blobs_from_encoder(self, x):
    
        x = self.init_layers(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        layers = [x1,x2,x3,x4]

        return layers[-self.n_blobs:]

    def get_blobs_from_decoder(self, x, index):

        x1 = self.decoder_network[:1](x)
        x2 = self.decoder_network[:2](x)
        x3 = self.decoder_network[:3](x)
        x4 = self.decoder_network[:4](x)

        layers = np.array([x1,x2,x3,x4])

        return list(layers[index])
    

    def forward(self, batch_input, task=None):
        batch_output = {}
        
        index = batch_input["idx"]
        self.stage = "finetune"

        device = index.device#batch_input["aug"].device
        bs = index.size(0)
        self.batch_size = bs

        views = batch_input["image"]
        road_map = batch_input["road"]
        bbox = batch_input["bbox"]
        classes = batch_input["classes"]
        # print(bbox.shape,classes.shape)
        annotations = torch.cat([bbox.type(torch.FloatTensor),classes.type(torch.FloatTensor)],dim=2)
        # print(annotations.shape)

        final_features = self.image_network(views.flatten(0,1))

        # print(final_features.shape)
        _,c,h,w = final_features.shape
        views = final_features.view(bs,6,c,h,w)

        batch_output["loss"] = 0

        if self.fusion == "concat":
            fusion = self.reduce(views.flatten(1,2))

        else:#if self.fusion == "mean":
            fusion = views.mean(dim=1)

        if self.n_synthesize > 0:
            fusion = self.synthesizer(fusion)
        
        # print("fusion:", fusion.shape)

        if "det" in self.model_type:
            # print("here")
            mapped_image = self.decoder_network(fusion)
            # print(mapped_image.shape, road_map.shape)

            batch_output["recon_loss"] = self.criterion(mapped_image, road_map)
            batch_output["road_map"] = mapped_image
            batch_output["acc"] = (mapped_image == road_map).float().mean()
            batch_output["loss"] += batch_output["recon_loss"]

        else:

            pool = self.avg_pool(fusion).view(bs,self.d_model)

            mu_logvar = self.z_project(pool).view(bs,2,-1)

            mu = mu_logvar[:,0]
            logvar = mu_logvar[:,1]

            z = self.reparameterize(mu,logvar).view(bs,self.latent_dim,1,1)

            generated_image = self.decoder_network(z)
            # print(generated_image.shape, mu.shape, logvar.shape)

            reconstruction_loss = self.criterion(generated_image, road_map)
            kl_divergence_loss = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

            batch_output["road_map"] = generated_image
            batch_output["recon_loss"] = reconstruction_loss
            batch_output["KLD_loss"] = kl_divergence_loss
            batch_output["acc"] = (generated_image == road_map).float().mean()
            batch_output["loss"] += batch_output["recon_loss"] + batch_output["KLD_loss"]

        if self.detect_objects:

            # print(fusion.shape)
            if "decoder" in self.blobs_strategy:
                if "var" in self.model_type:
                    layers = self.get_blobs_from_decoder(z,[1,2,3])
                    # print(len(layers))
                    # print(layers[0].shape, layers[1].shape, layers[2].shape)
                    features = self.fpn([layers[2],layers[1],layers[0]])
                else:
                    layers = self.get_blobs_from_decoder(fusion,[0,1])
                    features = self.fpn([layers[1],layers[0],fusion])
            else:
                layers = self.get_blobs_from_encoder(batch_input["image"].flatten(0,1))
                # print(layers[0].shape, layers[1].shape, layers[2].shape)
                features = self.fpn(layers)
            # print(len(features))
            # print(features[0].shape,features[1].shape, features[2].shape)
            # print([(self.regressionModel(feature).shape,feature.shape) for feature in features])
            regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
            # print("regression:", regression.shape)

            classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
            # print("classification:",classification.shape)

            anchors = self.anchors(batch_input["image"].flatten(0,1))

            # if self.training:
                # print(classification.shape, regression.shape, anchors.shape, annotations.shape)
            
            batch_output["classification_loss"], batch_output["detection_loss"] = self.focalLoss(classification.to(device), regression.to(device), anchors.to(device), annotations.to(device))
            batch_output["loss"] += batch_output["classification_loss"][0] + batch_output["detection_loss"][0]
                # print(batch_output["loss"].shape)

            if self.training:
                batch_output["classes"] = classification
                batch_output["boxes"] = regression
                
            else:

                # print(anchors.shape,regression.shape)
                transformed_anchors = self.regressBoxes(anchors, regression)
                # print(transformed_anchors.shape)
                transformed_anchors = self.clipBoxes(transformed_anchors, batch_input["image"].flatten(0,1))

                scores = torch.max(classification, dim=2, keepdim=True)[0]

                scores_over_thresh = (scores > 0.05)[0, :, 0]

                if scores_over_thresh.sum() == 0:
                    # batch_output["classification_loss"], batch_output["detection_loss"] = self.focalLoss(classification, regression, anchors, annotations)
                    # batch_output["loss"] += batch_output["classification_loss"][0] + batch_output["detection_loss"][0]

                    return batch_output
                    # no boxes to NMS, just return

                classification = classification[:, scores_over_thresh, :]
                transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
                scores = scores[:, scores_over_thresh, :]

                anchors_nms_idx = nms(transformed_anchors[0,:,:], scores[0,:,0], 0.5)

                # print(classification.shape, classification[0, anchors_nms_idx, :].shape)
                nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)
                
                print(nms_scores.shape,nms_class.shape)
                
                batch_output["classes"] = classification
                batch_output["boxes"] = regression

                # return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]

        return batch_output
