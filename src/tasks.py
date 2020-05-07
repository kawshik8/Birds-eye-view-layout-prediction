# This file implement dataset and tasks.
# dataset loads and preprocess data
# task manages data, create transformerd version, create dataloader, loss and evaluation metrics
import logging as log
import os
import torch
import json
import numpy
from PIL import Image
from torchvision import datasets, transforms
import random
import torch.nn.functional as F
from data_helper import *
import time 

def get_task(name, args):

    if name == "stl10_un":
        return STL10(name, args, pretrain=True)
    if name.startswith("stl10_fd"):
        return STL10(name, args, fold=int(name.replace("stl10_fd", "").split("_")[0]))
    elif name == "cifar10_un":
        return CIFAR10(name, args, pretrain=True)
    elif name.startswith("cifar10_lp"):
        return CIFAR10(name, args, label_pct=float(name.replace("cifar10_lp", "").split("_")[0]) / 100)
    elif name.startswith("custom_un"):
        return CUSTOM(name, args, pretrain=True)
    elif name.startswith("custom_sup"):
        return CUSTOM(name, args)#label_pct=float(name.replace("cifar100_lp", "").split("_")[0]) / 100)
    elif name == "cifar100_un":
        return CIFAR100(name, args, pretrain=True)
    elif name.startswith("cifar100_lp"):
        return CIFAR100(name, args, label_pct=float(name.replace("cifar100_lp", "").split("_")[0]) / 100)
    else:
        raise NotImplementedError


def task_num_class(name):
    if name.startswith("imagenet"):
        return 1000
    elif name.startswith("cifar100"):
        return 100
    else:
        return 10

def collater(data):
    # print(len(data))
    # print(len(data[0]))
    bs = len(data)

    image_shape = data[0][1].shape
    

    # index, image, bounding_box, classes, action, ego, road
    # index, image, query = inputs
    if len(data[0]) == 7:
        road_shape = data[0][-1].shape
        ego_shape = data[0][-2].shape
        indices = torch.zeros(bs,1)
        images = torch.zeros(bs, image_shape[0], image_shape[1], image_shape[2], image_shape[3])
        egos = torch.zeros(bs,ego_shape[0],ego_shape[1],ego_shape[2])
        roads = torch.zeros(bs,road_shape[0],road_shape[1],road_shape[2])

        shapes = []
        for item_set in data:
            # for item in item_set:
                shapes.append(item_set[2].shape[0])

        max_boxes = max(shapes)

        bboxes = torch.ones(bs,max_boxes,5)*-1
        classes = torch.ones(bs,max_boxes,1)*-1
        actions = torch.ones(bs,max_boxes)*-1

        items = [indices, images, bboxes, classes, actions, egos, roads]

        for i,data_items in enumerate(data):
            for j, item in enumerate(data_items):
                if j>=2 and j<=4:
                    shape = item.shape[0]
                    items[j][i][:shape] = item
                else:
                    items[j][i] = item

    
    else:
        query_shape = data[0][-1].shape
        indices = torch.zeros(bs,1)
        images = torch.zeros(bs, image_shape[0], image_shape[1], image_shape[2], image_shape[3])
        queries = torch.zeros(bs, query_shape[0], query_shape[1], query_shape[2], query_shape[3])

        items = [indices, images, queries]

        for i,data_items in enumerate(data):
            for j, item in data_items:
                items[j][i] = item

  
    return items


class RandomTranslateWithReflect:
    """
    Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = numpy.random.randint(
            -self.max_translation, self.max_translation + 1, size=2
        )
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop(
            (
                xpad - xtranslation,
                ypad - ytranslation,
                xpad + xsize - xtranslation,
                ypad + ysize - ytranslation,
            )
        )
        return new_image


class DupTransform:
    def __init__(self, num_dup, transform=lambda x: x):
        self.num_dup = num_dup
        self.transform = transform

    def __call__(self, inp):
        output = torch.stack([self.transform(inp) for _ in range(self.num_dup + 1)], dim=0)
        return output


class RandZero:
    def __init__(self, num_patches, num_queries):
        self.num_patches = num_patches
        self.num_queries = num_queries

    def __call__(self, query):
        mask = torch.randperm(self.num_patches) < self.num_queries
        return query * mask


class ToPatches:
    def __init__(self, num_patches, type, transform):
        self.num_div = int(numpy.sqrt(num_patches))
        self.type = type
        self.transform = transform

    def __call__(self, inp):
        if "random" in self.type:
            channel, height, width = inp.size()
            out = torch.zeros(self.num_div, channel, height, width)

            if "multi_view" in self.type:
                high = 0.5
            else:
                high = 0.75

            for i in range(self.num_div):
                size = int(random.uniform(0.25,high)*height)
                pixel = random.randrange(0,height-size)
                out[i] = F.interpolate(inp[:,pixel:pixel+size,pixel:pixel+size], size=(3,height,height))

        else:
            # print(inp.size())
            channel, height, width = inp.size()
            #print(inp.size())
            out = (
                inp.view(
                    channel, self.num_div, height // self.num_div, self.num_div, width // self.num_div
                )
                .transpose(2, 3)
                .flatten(1, 2)
                .transpose(0, 1)
            )
            # print(out.shape)
            out1 = torch.ones(self.num_div * self.num_div, channel, 64, 64)
            for i in range(out.size(0)):
                out1[i] = self.transform(out[i])
            

            # print(out.shape)
        return out1


class TransformDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, transform, tensors):
        self.transform = transform
        self.tensors = tensors
        for key in self.tensors:
            if key not in self.transform:
                self.transform[key] = lambda x: x

    def __getitem__(self, index):
        return {key: self.transform[key](tensor[index]) for key, tensor in self.tensors.items()}

    def __len__(self):
        return len(list(self.tensors.values())[0])


class Task(object):
    def __init__(self, name, args, pretrain=False):
        """
        inputs:
            name: str, dataset name
            args: args, global arguments
            pretrain: bool, load pretrain (self-supervised) data or finetune (supervised) data
        """
        self.name = name
        self.args = args
        self.pretrain = pretrain
        self.data_iterators = {}
        self.reset_scorers()
        self.path = os.path.join(args.data_dir, self.name.split("_")[0])
        # if pretrain:
        #     self.eval_metric = "acc"
        # else:
        if pretrain:
            self.eval_metric = "loss"
        else:
            self.eval_metric = "ts"
            #self.eval_metric = "loss"
            # if "adv" in args.finetune_obj:
            #     self.eval_metric = "Gloss"
        # if self.pretrain:
        #     self.eval_metric = "loss"
        # else:
        #     self.eval_metric = "ts"

    def _get_transforms(self):
        """
        outputs:
            train_transform: ...
            eval_transform: ...
        """
        raise NotImplementedError

    def _load_raw_data(self):
        """
        outputs:
            raw_data: dict[str, list[(image, label)]]: from split to list of data
                image: pil (*), raw image
                label: long (*), class label of the image, set to 0 when unavailable
        """
        raise NotImplementedError

    def _preprocess_data(self, data):
        output = {}
        for split, dataset in data.items():

            idx, image, label = zip(
                *[(idx, img, label) for idx, (img, label) in enumerate(dataset)]
            )
            output[split] = {
                "idx": torch.LongTensor(idx),
                "image": image,
                "query": image,
                "label": torch.LongTensor(label),
            }
            if self.pretrain:
                del output[split]["label"]
            else:
                del output[split]["query"]

        return output

    def make_data_split(self, train_data, pct=1.0, split_p=0.8):
        split_filename = os.path.join(self.path, "%s.json" % self.name)
        if os.path.exists(split_filename):
            with open(split_filename, "r") as f:
                split = json.loads(f.read())
        else:
            # start = time.time()
            full_size = len(train_data)
            train_size = int(full_size * pct * split_p)
            val_size = int(full_size * pct * (1-split_p)) + train_size
            full_idx = numpy.random.permutation(full_size).tolist()
            split = {"train": full_idx[:train_size], "val": full_idx[train_size:val_size]}
            # print(time.time()-start)
            with open(split_filename, "w") as f:
                f.write(json.dumps(split))
        val_data = [train_data[idx] for idx in split["val"]]
        train_data = [train_data[idx] for idx in split["train"]]
        return train_data, val_data

    def load_data(self):
        """
        load data, create data iterators. use cached data when available.
        """
        log.info("Loading %s data" % self.name)

        data = self._load_raw_data()
        # start = time.time()
        # data = self._preprocess_data(data)
        # print("preprocess", time.time()-start)
        # start = time.time()
        # train_transform, eval_transform = self._get_transforms()
        # if self.pretrain:
        #     data["train"] = TransformDataset(train_transform, data["train"])
        #     data["val"] = TransformDataset(eval_transform, data["val"])
        # else:
        #     data["train"] = TransformDataset(train_transform, data["train"])
        #     data["val"] = TransformDataset(eval_transform, data["val"])
        #     data["test"] = TransformDataset(eval_transform, data["test"])
        # print("transform", time.time()-start)

        if self.pretrain:
            for split, dataset in data.items():
                self.data_iterators[split] = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.args.batch_size,
                    shuffle=(split == "train"),
                    pin_memory=True,
                    drop_last=(split == "train"),
                    num_workers=self.args.num_workers,
                )
        else:
            for split, dataset in data.items():
                self.data_iterators[split] = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.args.batch_size,
                shuffle=(split == "train"),
                pin_memory=True,
                drop_last=(split == "train"),
                num_workers=self.args.num_workers,
                collate_fn = collater,
            )


    def reset_scorers(self):
        self.scorers = {"count" : 0} #"loss": [], "classification_loss": [], "detection_loss": [], "KLD_loss": [], "recon_loss":[], "acc": []}

        if self.pretrain:
            self.scorers.update({"loss": [], "acc": []})

            # TODO: Update this when new auxiliary losses are introduced
        else:
            if "adv" in self.args.finetune_obj:
                self.scorers.update({"loss": [], "ts": [], "GLoss": [], "GSupLoss": [], "GDiscLoss": [], "fake_DLoss": [], "real_DLoss":[], "ts_road_map":[]})#, "ts_boxes":[]})
            else:
                self.scorers.update({"loss": [], "ts": [], "classification_loss": [], "detection_loss": [], "KLD_loss": [], "recon_loss":[], "ts_road_map":[], "ts_boxes":[]})

                # print(self.scorers.keys())

                # print(self.args.detect_objects, self.args.gen_road_map)

                if not self.args.detect_objects:
                    # print("inside 1")
                    for i in ["classification_loss", "detection_loss", "ts_boxes"]:
                        del self.scorers[i]

                if not self.args.gen_road_map:
                    for i in ["KLD_loss", "recon_loss", "ts_road_map"]:
                        del self.scorers[i]

                if "KLD_loss" in self.scorers.keys() and "var" not in self.args.finetune_obj:
                    del self.scorers["KLD_loss"]
                
           

            # print(self.scorers.keys())
            
            

    def update_scorers(self, batch_input, batch_output):
        count = len(batch_input["idx"])
        self.scorers["count"] += count
        for key in self.scorers.keys():
            if key != "count":
                # if key == "ts":
                    # print(batch_output[key])
                    # print(self.scorers[key])
                self.scorers[key].append(batch_output[key].cpu().sum() * count)

    def report_scorers(self, reset=False):
        avg_scores = {
            key: sum(value) / self.scorers["count"]
            for key, value in self.scorers.items()
            if key != "count" and value != []
        }
        if reset:
            self.reset_scorers()
        return avg_scores

class CUSTOM(Task):
    def __init__(self, name, args, pretrain=False, label_pct=0.0, sample_type='sample'):
        super().__init__(name, args, pretrain)
        self.label_pct = label_pct
        self.instance_type = sample_type
        self.args = args

    def _get_transforms(self):
        flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.244, 0.262],)
        col_jitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        img_jitter = transforms.RandomApply([RandomTranslateWithReflect(4)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
#         rotation = torchvision.transforms.RandomRotation((0,90,180,270))
        rand_crop_image = transforms.RandomResizedCrop(size=(256, 256),scale=(0.6, 1.0))

        rand_crop_query = transforms.RandomResizedCrop(size=(255, 255),scale=(0.6, 1.0))
        if self.pretrain:
            if "pirl" in self.args.image_pretrain_obj:
                # print("PIRL transformations")
                train_transform = eval_transform = {
                    "image": transforms.Compose(
                            [
                                rand_crop_image,
                                col_jitter,
                                rnd_gray,
                                # transforms.Resize((256,256), interpolation=2),
                                transforms.ToTensor(),
#                                 transforms.Normalize((0, 0, 0), (1,1,1)),
                                # normalize,
                                # ToPatches(self.args.num_patches,self.args.view),
                            ]
                        ),
                    "query": transforms.Compose(
                            [
                                rand_crop_query,
                                # col_jitter,
                                # rnd_gray,
                                # transforms.Resize((256,256), interpolation=2),
                                transforms.ToTensor(),
#                                 transforms.Normalize((0, 0, 0), (1,1,1)),
				# normalize,
                                ToPatches(self.args.num_patches,self.args.view,transforms.Compose([torchvision.transforms.ToPILImage(),transforms.RandomCrop((64,64)),col_jitter,
                                rnd_gray,transforms.ToTensor()])),
                                
                            ]
                        ),
                }
            elif self.args.view_pretrain_obj != "none":
                train_transform = eval_transform = {
                    "image": transforms.Compose(
                            [
                                # rand_crop_image,
                                col_jitter,
                                rnd_gray,
                                
                                transforms.Resize((256,256), interpolation=2),
                                transforms.ToTensor(),
#                                 transforms.Normalize((0, 0, 0), (1,1,1)),
                                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                # normalize,
                                # ToPatches(self.args.num_patches,self.args.view),
                            ]
                        ),
                    "query": transforms.Compose(
                        [
                            transforms.Resize((256,256), interpolation=2),
                            transforms.ToTensor(),
                        ]
                    )
                }
            else:
                train_transform = eval_transform = {
                    "image": transforms.Compose(
                            [
                                # rand_crop_image,
                                col_jitter,
                                rnd_gray,
                                
                                transforms.Resize((256,256), interpolation=2),
                                transforms.ToTensor(),
#                                 transforms.Normalize((0, 0, 0), (1,1,1)),
                                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                # normalize,
                                # ToPatches(self.args.num_patches,self.args.view),
                            ]
                        ),
                    "query": transforms.Compose(
                            [
                                # rand_crop_image,
                                # col_jitter,
                                # rnd_gray,
                                transforms.Resize((256,256), interpolation=2),
                                transforms.ToTensor(),
#                                 transforms.Normalize((0, 0, 0), (1,1,1)),
                                # normalize,
                                # ToPatches(self.args.num_patches,self.args.view,transforms.Compose([torchvision.transforms.ToPILImage(),transforms.RandomCrop((64,64)),col_jitter,
                                # rnd_gray,transforms.ToTensor()])),
                                
                            ]
                        ),
                }

        else:
            train_transform = eval_transform = {
                "image": transforms.Compose(
                    [
#                         flip_lr,
#                         rotation,
                        transforms.Resize((256,256), interpolation=2),
                        transforms.ToTensor(),
#                         transforms.Normalize((0, 0, 0), (1,1,1)),
                    ]
                ),
                "road": transforms.Compose(
                    [
                        torchvision.transforms.ToPILImage(),
                        transforms.Resize((256,256), interpolation=2),
                        transforms.ToTensor(),
#                         transforms.Normalize((0, 0, 0), (1,1,1)),
                    ]
                ),
            }
        return train_transform, eval_transform

    def _load_raw_data(self):

        np.random.seed(8)
        train_transform, eval_transform = self._get_transforms()
        if self.pretrain:
            scene_index = np.random.permutation(np.arange(106))
            train_index = scene_index[:-20]
            val_index = scene_index[-20:]

            train = UnlabeledDataset(
                                    args = self.args,
                                    scene_index=train_index,
                                    transform = train_transform,
                                    )
            val = UnlabeledDataset(
                                    args = self.args,
                                    scene_index=val_index,
                                    transform = eval_transform,
                                )
                                
            # train, val = self.make_data_split(train, 1.0)
            self.args.vocab_size = len(train) + len(val)
            raw_data = {"train": train, "val": val}
        else:
            scene_index = np.random.permutation(np.arange(106, 134))
            train_index = scene_index[:-8]
            val_index = scene_index[-8:-4]
            test_index = scene_index[-4:]
            
            train = LabeledDataset(
                                  args= self.args,
                                  extra_info=True,
                                  scene_index=train_index,
                                  transform = train_transform,
                                 )
            val = LabeledDataset(
                                  args= self.args,
                                  extra_info=True,
                                  scene_index=val_index,
                                  transform = eval_transform,
                                 )
            test = LabeledDataset(
                                  args= self.args,
                                  extra_info=True,
                                  scene_index=test_index,
                                  transform = eval_transform,
                                 )
                                 
            # train, val = self.make_data_split(train, 1.0)
            # val, test = self.make_data_split(val, 1.0)
            raw_data = {"train": train, "val": val, "test": test}

        return raw_data


class CIFAR10(Task):
    def __init__(self, name, args, pretrain=False, label_pct=0.0):
        super().__init__(name, args, pretrain)
        self.label_pct = label_pct

    def _get_transforms(self):
        flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.244, 0.262],)
        col_jitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        img_jitter = transforms.RandomApply([RandomTranslateWithReflect(4)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        if self.pretrain:
            train_transform = eval_transform = {
                "idx": DupTransform(self.args.dup_pos),
                "image": DupTransform(
                    self.args.dup_pos,
                    transforms.Compose(
                        [
                            flip_lr,
                            img_jitter,
                            col_jitter,
                            rnd_gray,
                            transforms.ToTensor(),
                            normalize,
                            ToPatches(self.args.num_patches,self.args.view),
                        ]
                    ),
                ),
                "query": DupTransform(
                    self.args.dup_pos, RandZero(self.args.num_patches, self.args.num_queries)
                ),
            }
        else:
            train_transform = {
                "image": transforms.Compose(
                    [
                        flip_lr,
                        img_jitter,
                        col_jitter,
                        rnd_gray,
                        transforms.ToTensor(),
                        normalize,
                       # ToPatches(self.args.num_patches),
                    ]
                ),
            }
            eval_transform = {
                "image": transforms.Compose(
                    [
                        transforms.ToTensor(), 
                        normalize, 
                        #ToPatches(self.args.num_patches)
                    ]
                ),
            }
        return train_transform, eval_transform

    def _load_raw_data(self):
        cifar10_train = datasets.CIFAR10(root=self.path, train=True, download=True)
        if self.pretrain:
            cifar10_train, cifar10_val = self.make_data_split(cifar10_train, 1.0)
            raw_data = {"train": cifar10_train, "val": cifar10_val}
        else:
            cifar10_test = datasets.CIFAR10(root=self.path, train=False, download=True)
            cifar10_train, cifar10_val = self.make_data_split(cifar10_train, self.label_pct)
            raw_data = {"train": cifar10_train, "val": cifar10_val, "test": cifar10_test}
        # print(type(cifar10_train),len(cifar10_train),cifar10_train[0])#,type(cifar10_train[0]),len(cifar10_train)[0])
        return raw_data


class CIFAR100(CIFAR10):
    def __init__(self, name, args, pretrain=False, label_pct=0.0):
        super().__init__(name, args, pretrain, label_pct)

    def _load_raw_data(self):
        cifar100_train = datasets.CIFAR100(root=self.path, train=True, download=True)
        if self.pretrain:
            cifar100_train, cifar100_val = self.make_data_split(cifar100_train, 1.0)
            raw_data = {"train": cifar100_train, "val": cifar100_val}
        else:
            cifar100_test = datasets.CIFAR100(root=self.path, train=False, download=True)
            cifar100_train, cifar100_val = self.make_data_split(cifar100_train, self.label_pct)
            raw_data = {"train": cifar100_train, "val": cifar100_val, "test": cifar100_test}
        return raw_data


class STL10(Task):
    def __init__(self, name, args, pretrain=False, fold=0):
        super().__init__(name, args, pretrain)
        self.fold = fold

    def _get_transforms(self):
        flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        normalize = transforms.Normalize(mean=(0.43, 0.42, 0.39), std=(0.27, 0.26, 0.27))
        col_jitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        rand_crop = transforms.RandomResizedCrop(
            64, scale=(0.3, 1.0), ratio=(0.7, 1.4), interpolation=3
        )
        center_crop = transforms.Compose(
            [transforms.Resize(70, interpolation=3), transforms.CenterCrop(64)]
        )
        if self.pretrain:
            train_transform = eval_transform = {
                "idx": DupTransform(self.args.dup_pos),
                "image": DupTransform(
                    self.args.dup_pos,
                    transforms.Compose(
                        [
         #                   rand_crop,
                            col_jitter,
                            rnd_gray,
                            transforms.ToTensor(),
                            normalize,
                            ToPatches(self.args.num_patches,self.args.view),
                        ]
                    ),
                ),
                "query": DupTransform(
                    self.args.dup_pos, RandZero(self.args.num_patches, self.args.num_queries)
                ),
            }
        else:
            train_transform = {
                "image": transforms.Compose(
                    [
                        flip_lr,
        #                rand_crop,
                        col_jitter,
                        rnd_gray,
                        transforms.ToTensor(),
                        normalize,
                        #ToPatches(self.args.num_patches),
                    ]
                ),
            }
            eval_transform = {
                "image": transforms.Compose(
                    [
          #              center_crop,
                        transforms.ToTensor(),
                        normalize,
                        #ToPatches(self.args.num_patches),
                    ]
                ),
            }
        return train_transform, eval_transform

    def _load_raw_data(self):
        
        if self.pretrain:
            stl10_train = datasets.STL10(root=self.path, split="unlabeled", download=True)
            stl10_train, stl10_val = self.make_data_split(stl10_train)
            raw_data = {"train": stl10_train, "val": stl10_val}

        else:
            stl10_train = datasets.STL10(
                root=self.path, split="train", folds=self.fold, download=True
            )
            stl10_test = datasets.STL10(root=self.path, split="test", download=True)
            stl10_train, stl10_val = self.make_data_split(stl10_train)
            raw_data = {"train": stl10_train, "val": stl10_val, "test": stl10_test}
        return raw_data

