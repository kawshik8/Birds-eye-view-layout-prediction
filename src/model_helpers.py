import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
import numpy as np
from torchvision.ops import nms
from itertools import permutations,combinations
from torch.nn.modules.module import Module
from utils import Anchors, BBoxTransform, ClipBoxes, block, Resblock, dblock
import math


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
        print("input shape",x.shape)
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

        print("final1", out.shape)

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
        self.block1 = self.image_network[-4]
        self.block2 = self.image_network[-3]
        self.block3 = self.image_network[-2]
        self.block4 = self.image_network[-1]
        
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
        print("regression:", regression.shape)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
        print("classification:",classification.shape)

        anchors = self.anchors(batch_input["image"].flatten(0,1))
        print(anchors.shape)

        # if self.training:
            # print(classification.shape, regression.shape, anchors.shape, annotations.shape)

        batch_output["classification_loss"], batch_output["detection_loss"],batch_output["ts_boxes"] = self.focalLoss(classification.to(device), regression.to(device), anchors.to(device), annotations.to(device))
        batch_output["loss"] += batch_output["classification_loss"][0] + batch_output["detection_loss"][0]

        # batch_output["ts_obj_det"] = compute_ats_bounding_boxes()
            # print(batch_output["loss"].shape)

        if self.training:
            batch_output["classes"] = classification
            batch_output["boxes"] = regression
            batch_output["ts"] += batch_output["ts_boxes"]

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
            batch_output["ts"] += batch_output["ts_boxes"]
            
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
                if i==0 and "cond" in self.args.finetune_obj:
                    dcrefine_layers.append(dblock(self.dim+self.args.latent_dim,self.dim))
                else:
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

        
    def forward(self, x, z=None):

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

            if "cond" in self.args.finetune_obj:
                x = torch.cat([x,z],axis=-1)
            #     z = torch.randn(x.shape[0],self.args.latent_dim)

            

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


