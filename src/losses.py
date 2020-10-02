import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely.geometry import Polygon

from pdb import set_trace as bp

def compute_iou(box1, box2):
    a = Polygon(torch.t(box1)).convex_hull
    b = Polygon(torch.t(box2)).convex_hull
    
    return a.intersection(b).area / a.union(b).area

def get_orig_dim(boxes,gt=False):
    # print(boxes.shape)

    x1 = boxes[:,:,0].unsqueeze(-1)
    y1 = boxes[:,:,1].unsqueeze(-1)

    x2 = boxes[:,:,2].unsqueeze(-1)
    y2 = boxes[:,:,3].unsqueeze(-1)

    w = x2 - x1
    h = y2 - y1

    cx = x1 + w/2
    cy = y1 + h/2

    # w = boxes[:,2].unsqueeze(-1)
    # h = boxes[:,3].unsqueeze(-1)

    # x1 = cx1 - (w/2)
    # y1 = cy1 - (h/2)

    final_bbox = torch.zeros(boxes.shape[0],boxes.shape[1],3,4)

    final_bbox[:,:,0] = torch.cat([x1,x1+w,x1,x1+w],dim=-1)
    final_bbox[:,:,1] = torch.cat([y1,y1,y1+h,y1+h],dim=-1)
    final_bbox[:,:,2] = torch.zeros(x1.shape[0],x1.shape[1],4)

    gamma = boxes[:,:,4]

    cx = cx.unsqueeze(-1)
    cy = cy.unsqueeze(-1)
    gamma = gamma.unsqueeze(-1)

    # bp()

    if torch.cuda.is_available():
        translation_matrix = torch.cat([torch.ones(cx.shape).cuda(), torch.zeros(cx.shape).cuda(), cx, 
                                        torch.zeros(cx.shape).cuda(),torch.ones(cx.shape).cuda(),  cy, 
                                        torch.zeros(cx.shape).cuda(),torch.zeros(cx.shape).cuda(),torch.ones(cx.shape).cuda()],dim=-1).view(cx.shape[0],cx.shape[1],3,3)
        reverse_translation_matrix = torch.cat([torch.ones(cx.shape).cuda(), torch.zeros(cx.shape).cuda(), -cx, 
                                                torch.zeros(cx.shape).cuda(), torch.ones(cx.shape).cuda(), -cy, 
                                                torch.zeros(cx.shape).cuda(), torch.zeros(cx.shape).cuda(), torch.ones(cx.shape).cuda()], dim=-1).view(cx.shape[0],cx.shape[1],3,3)
        rotation_matrix = torch.cat([torch.cos(gamma), -torch.sin(gamma), torch.zeros(gamma.shape).cuda(), 
                                     torch.sin(gamma),  torch.cos(gamma), torch.zeros(gamma.shape).cuda(),
                                     torch.zeros(gamma.shape).cuda(),torch.zeros(gamma.shape).cuda(),torch.ones(gamma.shape).cuda()],dim=-1).view(gamma.shape[0],gamma.shape[1],3,3)
    else:
        translation_matrix = torch.cat([torch.ones(cx.shape), torch.zeros(cx.shape), cx, 
                                        torch.zeros(cx.shape),torch.ones(cx.shape),  cy, 
                                        torch.zeros(cx.shape),torch.zeros(cx.shape),torch.ones(cx.shape)],dim=-1).view(cx.shape[0],cx.shape[1],3,3)
        reverse_translation_matrix = torch.cat([torch.ones(cx.shape), torch.zeros(cx.shape), -cx, 
                                                torch.zeros(cx.shape), torch.ones(cx.shape), -cy, 
                                                torch.zeros(cx.shape), torch.zeros(cx.shape), torch.ones(cx.shape)], dim=-1).view(cx.shape[0],cx.shape[1],3,3)
        rotation_matrix = torch.cat([torch.cos(gamma), -torch.sin(gamma), torch.zeros(gamma.shape), 
                                     torch.sin(gamma),  torch.cos(gamma), torch.zeros(gamma.shape),
                                     torch.zeros(gamma.shape),torch.zeros(gamma.shape),torch.ones(gamma.shape)],dim=-1).view(gamma.shape[0],gamma.shape[1],3,3)
    # print(translation_matrix,reverse_translation_matrix,rotation_matrix)
    # print(box.shape)
    #     # print(box)
    if torch.cuda.is_available():
        final_bbox = final_bbox.cuda()
    bbox_rotated = torch.matmul(translation_matrix, torch.matmul(rotation_matrix, torch.matmul(reverse_translation_matrix, final_bbox)))[:,:,:2]
    # rotation_matrix = torch.from_numpy(rotation_matrix)
    final_bbox = bbox_rotated.transpose(2,3)
    # print("final_bbox in get orig dim", final_bbox.shape)
    
    return final_bbox

def compute_ats_bounding_boxes(pred, gt):

    # print(pred.shape, gt.shape)
    pred = pred.transpose(2,3)#get_orig_dim(pred)
    gt = get_orig_dim(gt).transpose(2,3)#get_orig_dim(gt)
    # print("boxes1 and boxes2:", pred.shape, gt.shape)

    ts_scores = torch.zeros(pred.shape[0])

    for batch in range(pred.shape[0]):

        boxes1 = pred[batch]
        boxes2 = gt[batch]

        num_boxes1 = boxes1.size(0)
        num_boxes2 = boxes2.size(0)

        boxes1_max_x = boxes1[:, 0].max(dim=1)[0]
        boxes1_min_x = boxes1[:, 0].min(dim=1)[0]
        boxes1_max_y = boxes1[:, 1].max(dim=1)[0]
        boxes1_min_y = boxes1[:, 1].min(dim=1)[0]

        boxes2_max_x = boxes2[:, 0].max(dim=1)[0]
        boxes2_min_x = boxes2[:, 0].min(dim=1)[0]
        boxes2_max_y = boxes2[:, 1].max(dim=1)[0]
        boxes2_min_y = boxes2[:, 1].min(dim=1)[0]

        condition1_matrix = (boxes1_max_x.unsqueeze(1) > boxes2_min_x.unsqueeze(0))
        condition2_matrix = (boxes1_min_x.unsqueeze(1) < boxes2_max_x.unsqueeze(0))
        condition3_matrix = (boxes1_max_y.unsqueeze(1) > boxes2_min_y.unsqueeze(0))
        condition4_matrix = (boxes1_min_y.unsqueeze(1) < boxes2_max_y.unsqueeze(0))
        # print(condition_matrix.shape)
        condition_matrix = condition1_matrix * condition2_matrix * condition3_matrix * condition4_matrix

        iou_matrix = torch.zeros(num_boxes1, num_boxes2)
        for i in range(num_boxes1):
            for j in range(num_boxes2):
                if condition_matrix[i][j]:
                    # print("goes inside")
                    # bp()
                    iou_matrix[i][j] = compute_iou(boxes1[i], boxes2[j])

        iou_max = iou_matrix.max(dim=0)[0]
        # print(iou_max)

        iou_thresholds = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
        total_threat_score = 0
        total_weight = 0
        for threshold in iou_thresholds:
            tp = (iou_max > threshold).sum()
            # print(threshold, tp)
            threat_score = tp * 1.0 / (num_boxes1 + num_boxes2 - tp)
            total_threat_score += 1.0 / threshold * threat_score
            total_weight += 1.0 / threshold

        average_threat_score = total_threat_score / total_weight
        # print(average_threat_score.shape)
        # print(average_threat_score)
        # print(i)
        ts_scores[batch] = average_threat_score
    # print(average_threat_score)
    
    return ts_scores


def compute_ts_road_map(road_map1, road_map2):
    tp = (road_map1 * road_map2).sum()

    return tp * 1.0 / (road_map1.sum() + road_map2.sum() - tp)

def calc_ariou(a, b):
    # print("ariou input shapes:",a.shape,b.shape)

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua
    # print("IoU:",IoU.shape)
    # print(",b angle shapes:", a[:, 4].shape, b[:,4].shape)
    # print("cos:",torch.cos(torch.unsqueeze(a[:, 4],dim=1) - b[:, 4]).shape)
    # print(a[:])
    arIoU = IoU * torch.cos(abs(torch.unsqueeze(a[:, 4],dim=1) - b[:, 4])-(np.pi/2))

    return arIoU

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    # arIou = IoU * torch.cos(a[:, 4] - torch.cos(b[:,4]))

    return IoU

class FocalLoss(nn.Module):
    def __init__(self, fused):
        super().__init__()
        self.fused = not fused
        self.cls_loss = torch.nn.BCEWithLogitsLoss()
        self.iou_type = "ariou"

    def forward(self, classifications, regressions, anchors, annotations):
        #print("---- input regressions ----")
        #print (regressions)
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        ts_scores = []

        # print("insider focal loss")
        # print(anchors.shape)
        # print(classifications.shape,regressions.shape, annotations.shape)

        if not self.fused:
            anchors = anchors.unsqueeze(1).repeat(1,6,1,1).flatten(1,2)
        
        anchor = anchors[0, :, :]

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights
        anchor_angle   = anchor[:, 4]

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]

            bbox_annotation = bbox_annotation[bbox_annotation[:, 5] != -1]

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                    classification_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())
                    classification_losses.append(torch.tensor(0).float())

                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            #bp()

            if self.iou_type == "ariou":

                IoU = calc_ariou(anchors[0, :, 0:5], bbox_annotation[:, :5])
            else:
                IoU = calc_iou(anchors[0, :, 0:4], bbox_annotation[:, :4]) # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            # angle_diff = abs(torch.unsqueeze(anchors[0,:,4],dim=1) - bbox_annotation[:, 4])

            # angle_min, angle_argmin = torch.min(angle_diff,dim=1)


            # bp()

            #import pdb
            #pdb.set_trace()

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1

            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0
            # targets[torch.gt(angle_min, np.pi/6)] = 0

            positive_indices = torch.ge(IoU_max, 0.5)
            # negative_angle_indices = torch.gt(angle_min, np.pi/6)
            # positive_indices = positive_indices * negative_angle_indices

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            # assigned_annotations = assigned_annotations

            # print(positive_indices[positive_indices].shape)
            # print(assigned_annotations[positive_indices, 4])
            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
            # print(targets[positive_indices])

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            # print(classification.shape, targets.shape)
            # print(targets[0])
            ce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            # print(ce.shape)
            # ce = F.binary_cross_entropy_with_logits(classification, targets, reduction='none')# * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            #ce = F.cross_entropy(classification, targets.max(dim=1)[1], reduction='none')
            # print(ce.shape)

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * ce

            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            #bp()

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # bp()
            # compute the loss for regression
            # print("positive_indices.sum()",positive_indices.sum())
            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]
                anchor_angle_pi = anchor_angle[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights
                gt_angle   = assigned_annotations[:, 4]

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)
                targets_dangle = torch.tan(gt_angle - anchor_angle_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh, targets_dangle))
                targets = targets.t()

                if torch.cuda.is_available():
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2, 1]]).cuda()
                else:
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2, 1]])

                negative_indices = 1 + (~positive_indices)

                # print ("----positive_indices----")
                # print (positive_indices)
                # print ("----negative_indices----")
                # print (negative_indices)
                # bp()

                # ts_scores.append(compute_ats_bounding_boxes(regression[positive_indices, 0:4],targets[0:4]))
                # print(target.shape, regression[positive_indices, :].shape)
                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )

                regression_losses.append(regression_loss.mean().float())
            else:

                if torch.cuda.is_available():
                    # ts_scores.append(torch.tensor(0).float().cuda())
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    # ts_scores.append(torch.tensor(0).float())
                    regression_losses.append(torch.tensor(0).float())

        #bp()

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)#, torch.stack(ts_scores).mean(dim=0, keepdim=True)