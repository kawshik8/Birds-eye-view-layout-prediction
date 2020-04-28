import sys
import os
sys.path.append('../')
sys.path.append('../src/')
sys.path.append('../../../data/code/')
from SSLmodels import get_model
from SupModels import *
from utils import *
from args import parser, process_args
import torchvision.transforms as transforms
from data_helper import *
from tasks import collater
import matplotlib.pyplot as plt
import cv2
from helper import collate_fn, draw_box

def compute_ts_road_map(road_map1, road_map2):
    
    tp = (road_map1 * road_map2).sum()

    return tp * 1.0 / (road_map1.sum() + road_map2.sum() - tp)


args = parser.parse_args()
process_args(args)
args.detect_objects = True
args.finetune_obj = "det_encoder"
args.road_map_loss = "bce"

model = get_model("sup", args)

print(args.road_map_loss,args.finetune_obj)

files = os.listdir("../../../sub_models1/results/")
print(files)

if args.detect_objects:
    file = [file for file in files if args.road_map_loss in file and args.finetune_obj.split("_")[0] in file and "obj" in file]
else:
    file = [file for file in files if args.road_map_loss in file and args.finetune_obj.split("_")[0] in file]

print(file)

model.load_state_dict(torch.load("../../../sub_models1/results/" + str(file[0]) + "/finetune_custom_sup_best.ckpt", map_location=torch.device('cpu')))
model.eval()
# print(model)

labeled_scene_index = np.arange(106, 108)

train_transform = eval_transform = {
                "image": transforms.Compose(
                    [
                        transforms.Resize((256,256), interpolation=2),
                        transforms.ToTensor(),
                    ]
                ),
                "road": transforms.Compose(
                    [
                        torchvision.transforms.ToPILImage(),
                        transforms.Resize((256,256), interpolation=2),
                        transforms.ToTensor(),
                    ]
                ),
            }

image_folder = '../data'
annotation_csv = '../data/annotation.csv'

labeled_trainset = LabeledDataset(
                                  args = args,
                                  scene_index=labeled_scene_index,
                                  transform=train_transform,
                                  extra_info=True,
                                 )
trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=1, shuffle=False, pin_memory=True, drop_last=False, num_workers=2, collate_fn=collater)


fig, ax = plt.subplots()

# ax.imshow(road_image[0], cmap='binary')
# plt.show()
color_list = ['b', 'g', 'orange', 'c', 'm', 'y', 'k', 'w', 'r']


for batch, inputs in enumerate(trainloader):
    if batch < 5:
        if os.path.isdir("./generated/" + str(file) + "/"):
            os.mkdir("./generated/" + str(file) + "/")
    
        index, image, bounding_box, classes, action, ego, road = inputs
        batch_input = {"image":image,"idx":index, "bbox":bounding_box, "classes":classes, "action":action, "ego":ego, "road":road}

        batch_output = model(batch_input)
        if args.detect_objects:
            boxes = batch_output["boxes"].detach().numpy()
            classes = batch_output["classes"].detach().numpy()
        
            print(boxes.shape)
            print(classes.shape)
        
        road = road.view(256,256)
#         road[road >= 0.5] = 1
#         road[road < 0.5] = 0

        
        road1 = batch_output["road_map"].view(256,256) 

        if args.detect_objects:
            fig, ax = plt.subplots()

            ax.imshow(road, cmap ='binary');

            # The ego car position
            ax.plot(128, 128, 'x', color="red")

            for i, bb in enumerate(batch_input['bbox'][0]):
                # You can check the implementation of the draw box to understand how it works 
    #             print(bb)
                bb = bb.unsqueeze(0)
    #             print(batch_input['classes'][0][i][0])
                bb = torch.cat([bb[:,2],bb[:,2],bb[:,0],bb[:,0],bb[:,1],bb[:,3],bb[:,1],bb[:,3]]).view(2,4)
                bb = ((((bb * 800)/256) - 400 )/10)*256/800
                draw_box(ax, bb, color=color_list[int(batch_input['classes'][0][i][0])])    


            plt.savefig("./generated/" + str(file) + "_gt_road_boxes_" + str(batch) + ".jpg")

#         ['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y','bl_y', 'br_y']
        

            fig, ax = plt.subplots()

            ax.imshow(road1.detach().numpy(), cmap ='binary');

            # The ego car position
            ax.plot(128, 128, 'x', color="red")

            for i, bb in enumerate(batch_output["boxes"][0]):
                # You can check the implementation of the draw box to understand how it works 
    #             print(bb.shape)
                bb = bb.unsqueeze(0).detach().clone()
                if(len(bb[bb!=0])>0):
                    print(bb)

                bb = torch.cat([bb[:,2],bb[:,2],bb[:,0],bb[:,0],bb[:,1],bb[:,3],bb[:,1],bb[:,3]]).view(2,4)
                bb = ((((bb * 800)/256) - 400 )/10)*256/800
    #             print(batch_output["classes"][0][i])
    #             print(int(batch_output["classes"][0][i].max(dim=0)[1]))
                draw_box(ax, bb, color=color_list[int(batch_output["classes"][0][i].max(dim=0)[1].detach().clone())])    


            plt.savefig("./generated/" + str(file) + "_gen_road_boxes_" + str(batch) + ".jpg")
        #         print("road",np.unique(road))
        
#         road1[road1 >= 0.5] = 1
#         road1[road1 < 0.5] = 0
        
#         print("road1",np.unique(road1))
        
        print(compute_ts_road_map(road,batch_output["road_map"]))
        road1 = road1.detach().numpy()
        road = road.detach().numpy()
        
#         plt.hist(road.flatten())
#         plt.savefig("./generated/gt_road_hist_" + str(batch) + ".jpg")
#         plt.show()

#         plt.hist(road1.flatten())
#         plt.savefig("./generated/gen_road_hist_" + str(batch) + ".jpg")
#         plt.show()
        
        cv2.imwrite("./generated/" + str(file) + "_gt_road_" + str(batch) + ".jpg",(road*255))
        cv2.imwrite("./generated/" + str(file) + "_gen_road_" + str(batch) + ".jpg",(road1*255))
#         print(os.listdir("./generated/"))
#         plt.imshow(road.view(256,256)*255)
#         batch_output["road_map"][batch_output["road_map"]>=0.5] = 1
#         batch_output["road_map"][batch_output["road_map"]<0.5] = 0
#         plt.imshow(batch_output["road_map"].view(256,256).detach().numpy()*255)
#         plt.show()


# print(model)

# model.load(