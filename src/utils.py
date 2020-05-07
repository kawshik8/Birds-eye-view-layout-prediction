import logging as log
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet

from pdb import set_trace as bp

EPSILON = 1e-8

def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

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

class dblock(nn.Module):
    
    def __init__(self, input_dim, output_dim, activation="relu", norm = True):
        super().__init__()

       
        self.dense = nn.Linear(input_dim, output_dim)

        self.use_norm = norm
        
        if norm:
            self.bn = nn.BatchNorm1d(output_dim)

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU(0.2,inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "identity":
            self.activation = nn.Identity()
            
    def forward(self, x):

        x = self.dense(x)

        if self.use_norm:
            # print(x.shape)
            x = self.bn(x)

        x = self.activation(x)
        
        return x

class block(nn.Module):
    
    def __init__(self, input_channels, output_channels, kernel, strides, pad, activation="relu", norm = True, use_transpose=False, upsample=False):
        super().__init__()

        if use_transpose:
            self.conv = nn.ConvTranspose2d(in_channels = input_channels, out_channels = output_channels, kernel_size = kernel, stride = strides, padding = pad)
        else:
            self.conv = nn.Conv2d(in_channels = input_channels, out_channels = output_channels, kernel_size = kernel, stride = strides, padding = pad)

        self.use_norm = norm
        self.upsample = upsample 
        
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
        elif activation == "identity":
            self.activation = nn.Identity()
            
    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)

        if self.use_norm:
            x = self.bn(x)

        x = self.activation(x)
        
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        return x

class Resblock(nn.Module):
    
    def __init__(self, input_channels, output_channels, kernel, strides, pad, activation="relu", norm = True):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = input_channels, out_channels = output_channels, kernel_size = kernel, stride = strides, padding = pad)
        self.conv2 = nn.Conv2d(in_channels = output_channels, out_channels = output_channels, kernel_size = kernel, stride = strides, padding = pad)

        self.use_norm = norm

        if norm:
            self.bn1 = nn.BatchNorm2d(output_channels)
            self.bn2 = nn.BatchNorm2d(output_channels)
            
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU(0.2,inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x1):

        x = self.conv1(x1)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        x+=x1
        x = self.activation(x)
        
        return x

class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None, angles=None):
        super(Anchors, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = np.array([0.2, 0.5, 1, 2])
        if scales is None:
            self.scales = np.array([0.02, 0.05, 0.08])
        if angles is None:
            self.angles = np.array([0, np.pi/12, np.pi/6, np.pi/4, np.pi/3,   np.pi*5/12, np.pi/2, 
                                     -np.pi/12, -np.pi/6, -np.pi/4, -np.pi/3, -np.pi*5/12, -np.pi/2,])

    def forward(self, image):
        
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        # print(image_shape)
        # print((image_shape + 2 ** 3 - 1))
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        # print("image shapes:", image_shapes)
        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 5)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            # print("pyramid lebel:",idx," base size:",self.sizes[idx])
            anchors         = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales, angles=self.angles)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)
        # print("all_anhors shape:", all_anchors.shape)

        if torch.cuda.is_available():
            return torch.from_numpy(all_anchors.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchors.astype(np.float32))

def generate_anchors(base_size=16, ratios=None, scales=None, angles=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales) * len(angles)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 5))

    # scale base_size
    # anchors[:, 2:4] = base_size * np.tile(scales, (2, len(ratios))).T
    anchors[:, 2:4] = base_size * np.tile(scales, (2, len(ratios)*len(angles))).T
    # print("scale base size:", anchors)

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    # anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    # anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales) * len(angles)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales) * len(angles))
    # print("correct_for_ratios:", anchors)
    # print(anchors)

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0:3:2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    anchors[:, 4] = np.tile(np.repeat(angles, len(ratios)), len(scales))
    # print(anchors)

    return anchors

def compute_shape(image_shape, pyramid_levels):
    """Compute shapes based on pyramid levels.
    :param image_shape:
    :param pyramid_levels:
    :return:
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    ratios=None,
    scales=None,
    strides=None,
    sizes=None,
    shapes_callback=None,
):

    image_shapes = compute_shape(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors         = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

    # print(all_anchors)
    return all_anchors


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    shifts = np.concatenate((shifts, np.zeros((K,1))), axis=1)
    all_anchors = (anchors.reshape((1, A, 5)) + shifts.reshape((1, K, 5)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 5))
    # print(all_anchors)

    return all_anchors

class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            if torch.cuda.is_available():
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0, 0]).astype(np.float32)).cuda()
            else:
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0, 0]).astype(np.float32))

        else:
            self.mean = mean
        if std is None:
            if torch.cuda.is_available():
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2, 1]).astype(np.float32)).cuda()
            else:
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2, 1]).astype(np.float32))
        else:
            self.std = std

    def forward(self, boxes, deltas):
        # print(boxes.shape, deltas.shape)


        widths  = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x   = boxes[:, :, 0] + 0.5 * widths
        ctr_y   = boxes[:, :, 1] + 0.5 * heights
        alpha   = boxes[:, :, 4]

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]
        dalpha = deltas[:, :, 4] * self.std[4] + self.mean[4]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w     = torch.exp(dw) * widths
        pred_h     = torch.exp(dh) * heights
        pred_alpha = torch.atan(dalpha) + alpha

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x4 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y4 = pred_ctr_y + 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h
        pred_boxes_x3 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y3 = pred_ctr_y - 0.5 * pred_h
        # bp()

        if torch.cuda.is_available():
            pred_boxes = torch.cat([pred_boxes_x1, pred_boxes_x2, pred_boxes_x3, pred_boxes_x4, 
                                    pred_boxes_y1, pred_boxes_y2, pred_boxes_y3, pred_boxes_y4, 
                                    torch.ones(pred_boxes_y4.shape).cuda(), torch.ones(pred_boxes_y4.shape).cuda(), torch.ones(pred_boxes_y4.shape).cuda(), 
                                    torch.ones(pred_boxes_y4.shape).cuda()]).view(pred_boxes_y4.shape[0],pred_boxes_y4.shape[1],3,4)#.transpose(2,3)
        else:
            pred_boxes = torch.cat([pred_boxes_x1, pred_boxes_x2, pred_boxes_x3, pred_boxes_x4, 
                                    pred_boxes_y1, pred_boxes_y2, pred_boxes_y3, pred_boxes_y4, 
                                    torch.ones(pred_boxes_y4.shape), torch.ones(pred_boxes_y4.shape), torch.ones(pred_boxes_y4.shape), 
                                    torch.ones(pred_boxes_y4.shape)]).view(pred_boxes_y4.shape[0],pred_boxes_y4.shape[1],3,4)#.transpose(2,3)

        pred_ctr_x = pred_ctr_x.unsqueeze(-1)
        pred_ctr_y = pred_ctr_y.unsqueeze(-1)
        pred_alpha = pred_alpha.unsqueeze(-1)
        # print(pred_ctr_x.shape)

        if torch.cuda.is_available():
            translation_matrix = torch.cat([torch.ones(pred_ctr_x.shape).cuda(), torch.zeros(pred_ctr_x.shape).cuda(), pred_ctr_x, 
                                            torch.zeros(pred_ctr_x.shape).cuda(), torch.ones(pred_ctr_x.shape).cuda(), pred_ctr_y, 
                                            torch.zeros(pred_ctr_x.shape).cuda(), torch.zeros(pred_ctr_x.shape).cuda(), torch.ones(pred_ctr_x.shape).cuda()],dim=-1).view(pred_ctr_x.shape[0],pred_ctr_x.shape[1],3,3)
            reverse_translation_matrix = torch.cat([torch.ones(pred_ctr_x.shape).cuda(), torch.zeros(pred_ctr_x.shape).cuda(), -pred_ctr_x, 
                                        torch.zeros(pred_ctr_x.shape).cuda(), torch.ones(pred_ctr_x.shape).cuda(), -pred_ctr_y, 
                                        torch.zeros(pred_ctr_x.shape).cuda(), torch.zeros(pred_ctr_x.shape).cuda(), torch.ones(pred_ctr_x.shape).cuda()],dim=-1).view(pred_ctr_x.shape[0],pred_ctr_x.shape[1],3,3)
            rotation_matrix = torch.cat([torch.cos(pred_alpha), -torch.sin(pred_alpha), torch.zeros(pred_alpha.shape).cuda(), 
                                         torch.sin(pred_alpha), torch.cos(pred_alpha), torch.zeros(pred_alpha.shape).cuda(), 
                                         torch.zeros(pred_alpha.shape).cuda(), torch.zeros(pred_alpha.shape).cuda(), torch.ones(pred_alpha.shape).cuda()],dim=-1).view(pred_alpha.shape[0],pred_alpha.shape[1],3,3)
        else:
            translation_matrix = torch.cat([torch.ones(pred_ctr_x.shape), torch.zeros(pred_ctr_x.shape), pred_ctr_x, 
                                            torch.zeros(pred_ctr_x.shape), torch.ones(pred_ctr_x.shape), pred_ctr_y, 
                                            torch.zeros(pred_ctr_x.shape), torch.zeros(pred_ctr_x.shape), torch.ones(pred_ctr_x.shape)],dim=-1).view(pred_ctr_x.shape[0],pred_ctr_x.shape[1],3,3)
            reverse_translation_matrix = torch.cat([torch.ones(pred_ctr_x.shape), torch.zeros(pred_ctr_x.shape), -pred_ctr_x, 
                                        torch.zeros(pred_ctr_x.shape), torch.ones(pred_ctr_x.shape), -pred_ctr_y, 
                                        torch.zeros(pred_ctr_x.shape), torch.zeros(pred_ctr_x.shape), torch.ones(pred_ctr_x.shape)],dim=-1).view(pred_ctr_x.shape[0],pred_ctr_x.shape[1],3,3)
            rotation_matrix = torch.cat([torch.cos(pred_alpha), -torch.sin(pred_alpha), torch.zeros(pred_alpha.shape), 
                                         torch.sin(pred_alpha), torch.cos(pred_alpha), torch.zeros(pred_alpha.shape), 
                                         torch.zeros(pred_alpha.shape),torch.zeros(pred_alpha.shape),torch.ones(pred_alpha.shape)],dim=-1).view(pred_alpha.shape[0],pred_alpha.shape[1],3,3)
        # print(translation_matrix,reverse_translation_matrix,rotation_matrix)
        # print(box.shape)
        # box = torch.cat([pred_boxes,torch.ones(pred_boxes.shape[0], pred_boxes.shape[1], pred_boxes.shape[-1]).type(torch.DoubleTensor).unsqueeze(2)],dim=2)
        # print(box)
        bbox_rotated = torch.matmul(translation_matrix, torch.matmul(rotation_matrix, torch.matmul(reverse_translation_matrix,pred_boxes)))[:,:,:2]

        pred_boxes = bbox_rotated.transpose(2,3)

        # print("fpred_boxes in get bbox_transform", pred_boxes.shape)


        # pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):

        # print(boxes.shape, img.shape)
        batch_size, num_channels, height, width = img.shape
        
        boxes[:, :, :, 0] = torch.clamp(boxes[:, :, :, 0], min=0, max=width)
        boxes[:, :, :, 1] = torch.clamp(boxes[:, :, :, 0], min=0, max=height)

        # boxes[:, :, 4] = torch.clamp(boxes[:, :, 3], max=np.pi, min=-np.pi)
      
        return boxes

def config_logging(log_file):
    log.basicConfig(
        level=log.INFO,
        format="%(asctime)s [%(threadName)-12.12s] %(message)s",
        handlers=[log.FileHandler(log_file), log.StreamHandler()],
    )


def load_model(load_ckpt, model):
    """
    Load a model, for training, evaluation or prediction
    """
    model_state = torch.load(load_ckpt)
    model.load_state_dict(model_state)
    log.info("Load parameters from %s" % load_ckpt)


def save_model(save_ckpt, model):
    """
    Save the parameters of the model to a checkpoint
    """
    torch.save(model.state_dict(), save_ckpt)
    log.info("Save parameters for %s" % save_ckpt)
