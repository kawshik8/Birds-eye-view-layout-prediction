import os
from PIL import Image

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from helper import convert_map_to_lane_map, convert_map_to_road_map

unlabelled_scene_index = np.arange(106)
labelled_scene_index = np.arange(106,134)

NUM_SAMPLE_PER_SCENE = 126
NUM_IMAGE_PER_SAMPLE = 6
image_names = [
    'CAM_FRONT_LEFT.jpeg',
    'CAM_FRONT.jpeg',
    'CAM_FRONT_RIGHT.jpeg',
    'CAM_BACK_LEFT.jpeg',
    'CAM_BACK.jpeg',
    'CAM_BACK_RIGHT.jpeg',
    ]

transform = torchvision.transforms.ToTensor()

# The dataset class for unlabeled data.
class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, args, scene_index=unlabelled_scene_index, transform=transform):
        """
        Args:
            image_folder (string): the location of the image folder
            scene_index (list): a list of scene indices for the unlabeled data 
            first_dim ({'sample', 'image'}):
                'sample' will return [batch_size, NUM_IMAGE_PER_SAMPLE, 3, H, W]
                'image' will return [batch_size, 3, H, W] and the index of the camera [0 - 5]
                    CAM_FRONT_LEFT: 0
                    CAM_FRONT: 1
                    CAM_FRONT_RIGHT: 2
                    CAM_BACK_LEFT: 3
                    CAM_BACK.jpeg: 4
                    CAM_BACK_RIGHT: 5
            transform (Transform): The function to process the image
        """
        self.args = args
        self.image_folder = self.args.image_folder
        self.scene_index = scene_index
        self.transform = transform
        self.first_dim = self.args.sampling_type
        assert self.first_dim in ['sample', 'image']

    def __len__(self):
        if self.first_dim == 'sample':
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE
        elif self.first_dim == 'image':
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE
    
    def __getitem__(self, index):
        if self.first_dim == 'sample':
            scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
            sample_id = index % NUM_SAMPLE_PER_SCENE
            sample_path = os.path.join(self.image_folder, 'scene_'+str(scene_id), 'sample_'+str(sample_id)) 
            
            images = []
            queries = []
            for image_name in image_names:
                image_path = os.path.join(sample_path, image_name)
                image = Image.open(image_path)
                image.load()
                images.append(self.transform["image"](image))
                queries.append(self.transform["query"](image))

            # print(images[0].shape)
            images = torch.cat(images).view(6,3,256,256)
            queries = torch.cat(queries).view(6,3,256,256)
            # print(images.shape)
            
            return index, images, queries

        elif self.first_dim == 'image':
            scene_id = self.scene_index[index // (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)]
            sample_id = (index % (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)) // NUM_IMAGE_PER_SAMPLE
            image_name = image_names[index % NUM_IMAGE_PER_SAMPLE]

            image_path = os.path.join(self.image_folder, 'scene_'+str(scene_id), 'sample_'+str(sample_id), image_name) 
            
            image = Image.open(image_path)
            image.load()

            query = self.transform["query"](image)
            image = self.transform["image"](image)

            return index, image, query

# The dataset class for labeled data.
class LabeledDataset(torch.utils.data.Dataset):    
    def __init__(self, args, scene_index=labelled_scene_index, extra_info=True, transform = transform):
        """
        Args:
            image_folder (string): the location of the image folder
            annotation_file (string): the location of the annotations
            scene_index (list): a list of scene indices for the unlabeled data 
            transform (Transform): The function to process the image
            extra_info (Boolean): whether you want the extra information
        """

        self.args = args
        self.image_folder = self.args.image_folder
        annotation_file = os.path.join(self.image_folder,"annotation.csv")
        self.annotation_dataframe = pd.read_csv(annotation_file)
        self.scene_index = scene_index
        self.transform = transform
        self.extra_info = extra_info
    
    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE

    def __getitem__(self, index):
        scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
        sample_id = index % NUM_SAMPLE_PER_SCENE
        sample_path = os.path.join(self.image_folder, 'scene_'+str(scene_id), 'sample_'+str(sample_id)) 

        images = []
        for image_name in image_names:
            image_path = os.path.join(sample_path, image_name)
            image = Image.open(image_path)
            image.load()
            images.append(self.transform["image"](image))
        image_tensor = torch.stack(images)

        data_entries = self.annotation_dataframe[(self.annotation_dataframe['scene'] == scene_id) & (self.annotation_dataframe['sample'] == sample_id)]
        corners = data_entries[['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y','bl_y', 'br_y']].to_numpy()
        categories = data_entries.category_id.to_numpy()
        
        ego_path = os.path.join(sample_path, 'ego.png')
        ego_image = Image.open(ego_path)
        ego_image.load()
        ego_image = torchvision.transforms.functional.to_tensor(ego_image)
        road_image = convert_map_to_road_map(ego_image)
        road_image = self.transform["road"](road_image.type(torch.FloatTensor))

#         print(torch.as_tensor(corners).view(-1, 2, 4).transpose(1,2).flatten(1,2))
        bounding_box = torch.as_tensor(corners).view(-1, 2, 4)#.transpose(1,2)#.flatten(1,2)
        bounding_box[:,0] = (bounding_box[:,0] * 10) + 400
        bounding_box[:,1] = (-bounding_box[:,1] * 10) + 400
        bounding_box = (bounding_box * 256)/800
        bounding_box = bounding_box.transpose(1,2)
        # print(bounding_box[:, :, 0].shape)
       
        bbox = torch.zeros(bounding_box.shape[0],4)
        # print(bbox.shape, bounding_box.shape)

        # bbox = (bbox * 256)/800

        bbox_new = torch.zeros(bounding_box.shape[0], 5)
        # print(bbox.shape, bounding_box.shape)
        # bbox[:, 0] = bounding_box[:, :, 0].min(dim=1)[0]
        # bbox[:, 1] = bounding_box[:, :, 1].min(dim=1)[0]
        # bbox[:, 2] = bounding_box[:, :, 0].max(dim=1)[0]
        # bbox[:, 3] = bounding_box[:, :, 1].max(dim=1)[0]

        # Computre rotate angle from center point
        for i, box in enumerate(bounding_box):
            if box[0][0] <= box[2][0] and box[0][1] >= box[1][1]:
                br = box[0]
                bl = box[1]
                fr = box[2]
                fl = box[3]
            else:
                fl = box[0]
                fr = box[1]
                bl = box[2]
                br = box[3]                
           
            print("before:",box)
            centerpoint = (fl+br)/2
            if fl[0] > fr[0]: # negative angle
                theta = torch.atan((centerpoint[1]-fr[1])/(fr[0]-centerpoint[0]))
                a = bl-centerpoint
                b = fl-centerpoint
                tempangle = torch.acos(torch.dot(a,b)/(torch.norm(a, 2)*torch.norm(b, 2)))
                beta = (np.pi-tempangle)/2
                gamma = -(theta-beta)
                # print ("-----test----")
                # print (torch.norm(a, 2))
                # print (torch.norm(b, 2))
                # print (theta)
                # print (beta)
                # print (gamma)
            else: # positive angle
                theta = torch.atan((centerpoint[1]-br[1])/(centerpoint[0]-br[0]))
                a = fl-centerpoint
                b = bl-centerpoint
                tempangle = torch.acos(torch.dot(a,b)/(torch.norm(a, 2)*torch.norm(b, 2)))
                beta = (np.pi-tempangle)/2
                gamma = (theta-beta)

            print((gamma*180)/np.pi)
                
                #theta = np.arctan((fr[1] - br[1])/(fr[0]-br[0]))
            bbox_new[i, 4] = gamma
            
            translation_matrix = torch.tensor([[1,0,centerpoint[0]],[0,1,centerpoint[1]],[0,0,1]])
            reverse_translation_matrix = torch.tensor([[1,0,-centerpoint[0]],[0,1,-centerpoint[1]],[0,0,1]])
            rotation_matrix = torch.tensor([[torch.cos(-gamma.unsqueeze(0)), -torch.sin(-gamma.unsqueeze(0)), 0],[torch.sin(-gamma.unsqueeze(0)), torch.cos(-gamma.unsqueeze(0)), 0],[0,0,1]])
            # print(translation_matrix,reverse_translation_matrix,rotation_matrix)
            # print(box.shape)
            box = torch.cat([box.transpose(0,1),torch.ones(box.shape[0]).type(torch.DoubleTensor).unsqueeze(0)],dim=0)
            print(box)
            bbox_rotated = torch.matmul(translation_matrix, torch.matmul(rotation_matrix, torch.matmul(reverse_translation_matrix,box)))[:2]
            print(bbox_rotated)
            # print("\nrotation matrix shape:",rotation_matrix.shape)
            # rotation_matrix = torch.from_numpy(rotation_matrix)
            # bbox_rotated = torch.matmul(rotation_matrix, torch.transpose(box, 0, 1))
            print("\nbbox_rotated shape:",bbox_rotated.shape)
            print("\nrotated_bbox:", bbox_rotated)
            print("\nbbox new shape:",bbox_new.shape)
            if box[0][0] <= box[2][0] and box[0][1] >= box[1][1]:

                bbox_new[i, 0] = bbox_rotated[0, 1]
                bbox_new[i, 1] = bbox_rotated[1, 1]
                bbox_new[i, 2] = bbox_rotated[0, 2]
                bbox_new[i, 3] = bbox_rotated[1, 2]
            
            else:

                bbox_new[i, 0] = bbox_rotated[0, 0]
                bbox_new[i, 1] = bbox_rotated[1, 0]
                bbox_new[i, 2] = bbox_rotated[0, 3]
                bbox_new[i, 3] = bbox_rotated[1, 3]

            print("\nafter:",bbox_new[i])
            if len(bbox_rotated[bbox_rotated<0])>0:
                exit(0)

        # print(bbox[0])
        # print(scene_id, sample_id, bounding_box.shape)
        classes = torch.as_tensor(categories).view(-1, 1)


        if self.extra_info:
            actions = data_entries.action_id.to_numpy()
            # You can change the binary_lane to False to get a lane with 
            lane_image = convert_map_to_lane_map(ego_image, binary_lane=True)
            
            action = torch.as_tensor(actions)
            ego = self.transform["road"](ego_image)
            road = lane_image

            # print(scene_id, sample_id, bounding_box[0])
            # print(bounding_box.shape,classes.shape)
            # print(classes)
            # exit(0)
            return index,image_tensor, bbox_new, classes, action, ego, road_image

        else:
            return index,image_tensor, bbox_new, classes

        
