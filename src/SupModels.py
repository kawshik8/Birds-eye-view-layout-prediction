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
from utils import Anchors, BBoxTransform, ClipBoxes, block, Resblock, dblock
import losses
from losses import compute_ts_road_map, compute_ats_bounding_boxes
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
            self.fuse1 = block(6*C3_size,C3_size, kernel = 3, strides = 1, pad = 1)
            self.fuse2 = block(6*C4_size,C4_size, kernel = 3, strides = 1, pad = 1)
            self.fuse3 = block(6*C5_size,C5_size, kernel = 3, strides = 1, pad = 1)

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
        self.output_act = nn.Sigmoid()

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
    
class ObjectDetectionHeads(nn.Module):
    def __init__(self, args, image_network, decoder_network=None):
        super().__init__()
        
        
        self.args = args
        self.blobs_strategy = self.args.blobs_strategy
        self.model_type = self.args.finetune_obj.split("_")[0]

        self.num_classes = 9
        self.n_blobs = 3

        # print(image_network)
        self.image_network = image_network
        self.init_layers = self.image_network[0]
        self.block1 = self.image_network[1]
        self.block2 = self.image_network[2]
        self.block3 = self.image_network[3]
        self.block4 = self.image_network[4]
        
        self.decoder_network = decoder_network
        
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
            self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fusion_strategy = "concat_fuse")
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

        self.params = nn.Sequential(
            self.fpn,
            self.regressionModel,
            self.classificationModel,
        )

        
        
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
    
    def forward(self, inputs, batch_input, batch_output, fusion=None):
        
        device = batch_input["idx"].device
        bbox = batch_input["bbox"]
        classes = batch_input["classes"]
        # print(bbox.shape,classes.shape)
        annotations = torch.cat([bbox.type(torch.FloatTensor),classes.type(torch.FloatTensor)],dim=2)

        if "decoder" in self.blobs_strategy:
            if "var" in self.model_type:
                layers = self.get_blobs_from_decoder(inputs,[1,2,3])
                # print(len(layers))
                # print(layers[0].shape, layers[1].shape, layers[2].shape)
                features = self.fpn([layers[2],layers[1],layers[0]])
            else:
                layers = self.get_blobs_from_decoder(inputs,[0,1])
                features = self.fpn([layers[1],layers[0],fusion])
        else:
            layers = self.get_blobs_from_encoder(inputs.flatten(0,1))
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

        batch_output["classification_loss"], batch_output["detection_loss"],batch_output["ts_boxes"] = self.focalLoss(classification.to(device), regression.to(device), anchors.to(device), annotations.to(device))
        batch_output["loss"] += batch_output["classification_loss"][0] + batch_output["detection_loss"][0]

        # batch_output["ts_obj_det"] = compute_ats_bounding_boxes()
            # print(batch_output["loss"].shape)

        if self.training:
            batch_output["classes"] = classification
            batch_output["boxes"] = regression

        else:

            if self.dynamic_strategy:
                anchors = anchors.unsqueeze(1).repeat(1,6,1,1).flatten(1,2)

            # print(anchors.shape,regression.shape)
            transformed_anchors = self.regressBoxes(anchors, regression)
            # print(transformed_anchors.shape)
            transformed_anchors = self.clipBoxes(transformed_anchors, batch_input["image"].flatten(0,1))

            scores = torch.max(classification, dim=2, keepdim=True)[0]
            # print()
            # print(scores.shape, (scores > 0.05).shape)

            scores_over_thresh = (scores > 0.05)[:, :, 0]

            if scores_over_thresh.sum() == 0:
                # batch_output["classification_loss"], batch_output["detection_loss"] = self.focalLoss(classification, regression, anchors, annotations)
                # batch_output["loss"] += batch_output["classification_loss"][0] + batch_output["detection_loss"][0]
                # print("no boxes to NMS, just return")
                return batch_output
                # no boxes to NMS, just return

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(transformed_anchors[0,:,:], scores[0,:,0], 0.5)

            # print(classification.shape, classification[0, anchors_nms_idx, :].shape)
            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            # print(nms_scores.shape,nms_class.shape)

            batch_output["classes"] = classification
            batch_output["boxes"] = regression
            
        return batch_output

class Fusion(nn.Module):
    def __init__(self, args, d_model, frefine_layers, brefine_layers, drefine_layers, dcrefine_layers, dense_fusion=False, conv_fusion=False, flatten=False):
        super().__init__()

        self.args = args
        self.d_model = d_model
        self.dim = d_model
        self.fusion_strategy = self.args.view_fusion_strategy
        self.dense_fusion = dense_fusion
        self.conv_fusion = conv_fusion

        # self.params = []
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.drefine_nlayers = drefine_layers
        self.brefine_nlayers = brefine_layers
        self.frefine_nlayers = frefine_layers
        self.dcrefine_nlayers = dcrefine_layers

        # print(self.dense_fusion, self.conv_fusion)

        if self.drefine_nlayers>0:
            
            drefine_layers = []
            for i in range(self.drefine_nlayers):
                drefine_layers.append(dblock(self.dim,self.dim))
                drefine_layers.append(nn.Dropout(0.5))

            self.dense_project_layers = nn.Sequential(*drefine_layers)
            # self.params += list(self.dense_project_layers.parameters())


        if self.dense_fusion:
            if self.frefine_nlayers > 0:
                frefine_layers = []
                for i in range(self.frefine_nlayers):
                    frefine_layers.append(dblock(6*self.dim,6*self.dim))
                    frefine_layers.append(nn.Dropout(0.5))

                self.refine_before_fuse = nn.Sequential(*frefine_layers)

            if "concat" in self.fusion_strategy:
                reduce_layers = []
                reduce_layers.append(dblock(int(6*self.dim), int(self.dim)))
                reduce_layers.append(nn.Dropout(0.5))
                self.reduce_views = nn.Sequential(*reduce_layers)

            if self.brefine_nlayers > 0:
                brefine_layers = []
                for i in range(self.brefine_nlayers):
                    brefine_layers.append(dblock(self.dim,self.dim))
                    brefine_layers.append(nn.Dropout(0.5))

                self.refine_after_fuse = nn.Sequential(*brefine_layers)


        if self.conv_fusion:  
            if self.drefine_nlayers > 0:
                self.reshape = dblock(self.dim, 16*16*32)
                self.conv_refine = nn.Sequential(block(32, 64, 3, 1, 1),Resblock(64, 64, 3, 1, 1))
                self.dim = 64

            if self.frefine_nlayers > 0:
                frefine_layers = []
                for i in range(self.frefine_nlayers):
                    frefine_layers.append(Resblock(int(6*self.dim), int(6*self.dim), 3, 1, 1))
                
                self.refine_before_fuse = nn.Sequential(*frefine_layers)

            if "concat" in self.fusion_strategy:
                self.reduce_views = block(int(6*self.dim), int(self.dim), 3, 1, 1)

            if self.brefine_nlayers > 0:
                brefine_layers = []
                for i in range(self.brefine_nlayers):
                    brefine_layers.append(Resblock(int(self.dim), int(self.dim), 3, 1, 1))

                self.refine_after_fuse = nn.Sequential(*brefine_layers)
            
        if self.dcrefine_nlayers>0:

            dcrefine_layers = []
            for i in range(self.dcrefine_nlayers):
                dcrefine_layers.append(dblock(self.dim,self.dim))
                dcrefine_layers.append(nn.Dropout(0.5))
            
            dcrefine_layers.append(dblock(self.dim, 16*16*32))
            self.refine = Resblock(32,32,3,1,1)

            self.dense_reduce_layers = nn.Sequential(*dcrefine_layers)
            

        else:
            if self.dense_fusion:

                self.reshape = dblock(self.dim, 16*16*32)
                self.refine = Resblock(32,32,3,1,1)
        
        # self.reduce = []

        # if self.frefine_nlayers > 0:
        #     self.params += list(self.refine_before_fuse.parameters())
        #     # self.reduce.append(self.refine_before_fuse)

        # # self.reduce.append(self.reduce_views)
        # if "concat" in self.fusion_strategy:
        #     self.params += list(self.reduce_views.parameters())

        # if self.brefine_nlayers > 0:
        #     self.params += list(self.refine_after_fuse.parameters())
            # self.reduce.append(self.refine_after_fuse)

        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)

        
    def forward(self, x):

        # print("inside fuse")
        # print("initial", x.shape)

        if self.drefine_nlayers>0:
            # print(self.avg_pool(x.flatten(0,1)).shape)
            x = self.avg_pool(x.flatten(0,1)).view(-1,self.d_model)
            x = self.dropout(x)
            # print("here", x.shape)
            x = self.dense_project_layers(x).view(-1,6,self.d_model)

            if self.conv_fusion:
                x = self.reshape(self.dropout(x.flatten(0,1))).view(-1,32,16,16)
                x = self.conv_refine(x)
                _,c,h,w = x.shape
                x = x.view(-1,6,c,h,w)

            x = x.flatten(1,2)
                
        else:
            if self.dense_fusion:
                x = self.avg_pool(x.flatten(0,1)).view(-1,6,self.dim).flatten(1,2)
                # print(x.shape)
                x = self.dropout(x)
            else:
                x = x.flatten(1,2)
                            
        # print("before fuse", x.shape)

        if self.frefine_nlayers > 0:
            x = self.refine_before_fuse(x)

        # print("before reduce", x.shape)

        
        if "concat" in self.fusion_strategy:
            x = self.reduce_views(x)
        else:
            # print(x.shape)
            if len(x.shape) > 3:
                _,c,h,w = x.shape
                x = x.view(-1,6,(c//6),h,w)
            else:
                _,d = x.shape
                x = x.view(-1,6,d//6)

            if "mean" in self.fusion_strategy:
                x = x.mean(dim=1)

            elif "sum" in self.fusion_strategy:
                x = x.sum(dim=1)

        if self.dense_fusion:
            x = self.dropout(x)

        # print("after reduce", x.shape)

        if self.brefine_nlayers > 0:
            x = self.refine_after_fuse(x)

        if self.dcrefine_nlayers > 0:
            if self.conv_fusion:
                x = self.avg_pool(x).view(-1,self.dim)

            x = self.refine(self.dense_reduce_layers(x).view(-1,32,16,16))

        elif self.dense_fusion:
            x = self.refine(self.reshape(x).view(-1,32,16,16))

        return x

class DecoderNetwork(nn.Module):
    def __init__(self, init_layer_dim, init_channel_dim, max_f, d_model, add_convs_before_decoding=False,add_initial_upsample_conv=False):
        super().__init__()
        
        decoder_network_layers = []

        self.max_f = max_f
        self.init_layer_dim = init_layer_dim
        self.init_channel_dim = init_channel_dim
        self.d_model = d_model
        self.input_dim = 256

        if add_initial_upsample_conv:
            decoder_network_layers.append(
                block(int(init_channel_dim//2), int(init_channel_dim), 3, 1, 1, upsample=True)
            )
                
            decoder_network_layers.append(
                block(int(init_channel_dim), int(init_channel_dim), 3, 1, 1, activation="identity")
            )

        if add_convs_before_decoding:
            # print("add_convs_before_decoding",add_convs_before_decoding)

            decoder_network_layers.append(
                block((self.d_model), int(self.init_channel_dim), 3, 1, 1)
            )
#                    
            
            decoder_network_layers.append(
                block(int(self.d_model//4), int(self.init_channel_dim), 3, 1, 1, activation='identity')
            )

        

        while self.init_layer_dim < self.input_dim:
            decoder_network_layers.append(
                block(int(self.init_channel_dim), int(self.init_channel_dim//2), 3, 1, 1, upsample=True)
            )
                
            decoder_network_layers.append(
                block(int(self.init_channel_dim//2), int(self.init_channel_dim//2), 3, 1, 1, activation="identity")
            )
            
            self.init_layer_dim *= 2
            self.init_channel_dim = self.init_channel_dim / 2
        
#                 decoder_network_layers.append(
#                     nn.ZeroPad2d((1,0,1,0))
#                 )
        decoder_network_layers.append(
            block(int(self.init_channel_dim), 1, 3, 1, 1, activation="identity", norm = False),
        )

        self.decoder_network = nn.Sequential(*decoder_network_layers)


    def forward(self,inputs):

        return(self.decoder_network(inputs))


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
#             print(self.fuse)
                                              
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

#             print(self.decoder_network)
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
        self.shared_params = list(self.image_network[:-1].parameters())
        self.finetune_params += list(self.image_network[-1].parameters())
        
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
            eps = torch.randn_like(std).to(device)
            return mu + eps*std
        else:
            return mu

    def forward(self, batch_input, task=None):
        batch_output = {}
        
        index = batch_input["idx"]
        self.stage = "finetune"

        device = index.device
        bs = index.size(0)
        self.batch_size = bs

        views = batch_input["image"]
        road_map = batch_input["road"]
        
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
                
                batch_output["recon_loss"] = self.criterion(mapped_image, road_map)
                batch_output["road_map"] = torch.sigmoid(mapped_image)
                batch_output["ts_road_map"] = compute_ts_road_map(batch_output["road_map"],road_map)
                batch_output["loss"] += batch_output["recon_loss"]

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

                reconstruction_loss = self.criterion(generated_image, road_map)
                kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                batch_output["road_map"] = torch.sigmoid(generated_image)
                batch_output["recon_loss"] = reconstruction_loss
                batch_output["KLD_loss"] = kl_divergence_loss
                batch_output["ts_road_map"] = compute_ts_road_map(batch_output["road_map"],road_map)
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
