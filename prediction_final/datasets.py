import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from collections import Counter
import glob
import os

###############################################################################
#SRDataset

class SRDataset(Dataset):
    def __init__(self,path,with_test=True, max_size=300, final_size=64, padding=True):
        self.folders_names = os.listdir(path)
        self.images_path=[]
        self.labels = []
        self.test_images_path = []
        self.test_labels = []
        self.with_test=with_test
        self.max_h = max_size
        self.max_w = max_size
        self.final_size=final_size
        self.padding = padding
        for subfolder_name in self.folders_names:
            subfolder_images_names = glob.glob( os.path.join(path,subfolder_name , '*.jpg')   )
            if with_test:
                subfolder_images_names_train, subfolder_images_names_test =train_test_split(subfolder_images_names , test_size = 0.2)

                ## train
                self.images_path+= subfolder_images_names_train
                self.labels += [ int(subfolder_name[:3])  ]*len(subfolder_images_names_train)

                ## test
                self.test_images_path +=subfolder_images_names_test
                self.test_labels += [ int(subfolder_name[:3])  ]*len(subfolder_images_names_test)
            else:
                ## train
                self.images_path+= subfolder_images_names
                self.labels += [ int(subfolder_name[:3])  ]*len(subfolder_images_names)

    def __len__(self):
        return len(self.images_path)
    def __getitem__(self,index):
        label_ = torch.tensor(self.labels[index])
        im=torchvision.io.read_image(self.images_path[index], torchvision.io.ImageReadMode.GRAY)
        im=(1/255)*im
        im = im.repeat_interleave(3, dim=0)
        if self.padding:
            w,h = list(im.size())[1:]
            top = (self.max_h - h)//2
            bottom = self.max_h - top - h
            left = (self.max_w - w)//2
            right = self.max_w - left - w
            im=transforms.Compose([
                transforms.Pad((left, top, right, bottom), fill=1 ),
                transforms.Resize((self.final_size,self.final_size),interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                ])(im)
        else:
            im=transforms.Compose([
                transforms.Resize((self.final_size,self.final_size),interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                ])(im)
        return im,label_

    def get_labels(self):
        return self.labels

    def get_test_set(self):
        return self.test_images_path,self.test_labels
    def get_distribution(self):
        return dict(Counter(self.labels))
    def classes(self):
        indices = [[] for i in range(len(self.get_distribution().keys()))]
        for i in range(len(self.labels)):
            indices[self.labels[i]].append(i)
        return indices

class SREvalDataset(Dataset):
    def __init__(self,test_images_path,test_labels, max_size=300, final_size=64):
        self.test_images_path = test_images_path
        self.test_labels = test_labels
        self.max_h = max_size
        self.max_w = max_size
        self.final_size=final_size

    def __len__(self):
        return len(self.test_images_path)
    def __getitem__(self,index):
        label_ = torch.tensor(self.test_labels[index])
        im=torchvision.io.read_image(self.test_images_path[index], torchvision.io.ImageReadMode.GRAY)
        im=(1/255)*im
        im = im.repeat_interleave(3, dim=0)
        im=transforms.Compose([
            transforms.Resize((self.final_size,self.final_size),interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
            ])(im)
        return im, label_

    def get_labels(self):
        return self.test_labels
    def get_distribution(self):
        return dict(Counter(self.test_labels))
    def classes(self):
        indices = [[] for i in range(len(self.get_distribution().keys()))]
        for i in range(len(self.test_labels)):
            indices[self.test_labels[i]].append(i)
        return indices

###############################################################################
#weights handling

def get_weights(dataset_):
        weights_ = [0]*len(dataset_.get_labels())
        labels_distribution = dataset_.get_distribution()
        sum_val = sum(labels_distribution.values())
        for i in range(len(dataset_.get_labels())):
            weights_[i] = sum_val/labels_distribution[ dataset_.get_labels()[i]]
        return weights_

###############################################################################
#data loader

def get_dataloader(dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=4, drop_last=False, weights=False):
    if not weights:
        return torch.utils.data.dataloader.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory,
                num_workers=num_workers,
                drop_last=drop_last)
    else:
        ## get the labels distributions
        labels_distribution = dataset.get_distribution()

        ## weights for sampling
        weights =get_weights(dataset)
        sampler = WeightedRandomSampler(weights,len(weights))

        return torch.utils.data.dataloader.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler = sampler,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers = num_workers)


#############################################################################
class AugmentedDataset(Dataset):
    """"
    Dataset with data augmentation and/or undersampling
    """
    def __init__(self,path,with_test=True, final_size=64, minimum=4000, maximum=80000):
        self.folders_names = os.listdir(path)
        self.images_path=[]
        self.labels = []
        self.test_images_path = []
        self.test_labels = []
        self.with_test=with_test
        self.final_size=final_size
        find_test_size = lambda x: int(0.2*maximum) if (0.2*maximum)<=(x-maximum) else int(x-maximum)
        for subfolder_name in self.folders_names:
            subfolder_images_names = glob.glob( os.path.join(path,subfolder_name , '*.jpg')   )
            if with_test:
                if len(subfolder_images_names)>maximum:
                    test_size = find_test_size(len(subfolder_images_names))
                    subfolder_images_names_train, subfolder_images_names_test =train_test_split(subfolder_images_names , train_size=maximum, test_size=test_size)
                else:
                    subfolder_images_names_train, subfolder_images_names_test =train_test_split(subfolder_images_names, test_size = 0.2)

                ## train
                self.images_path+= subfolder_images_names_train
                self.labels += [ int(subfolder_name[:3])  ]*len(subfolder_images_names_train)

                ## test
                self.test_images_path +=subfolder_images_names_test
                self.test_labels += [ int(subfolder_name[:3])  ]*len(subfolder_images_names_test)
                if len(subfolder_images_names_train) < minimum:
                    ratio = int(minimum/len(subfolder_images_names_train))-1
                    self.images_path+= subfolder_images_names_train*ratio
                    self.labels += [ int(subfolder_name[:3])  ]*(len(subfolder_images_names_train)*ratio)
            else:
                ## train
                self.images_path+= subfolder_images_names
                self.labels += [ int(subfolder_name[:3])  ]*len(subfolder_images_names)

    def __len__(self):
        return len(self.images_path)
    def __getitem__(self,index):
        label_ = torch.tensor(self.labels[index])
        im=torchvision.io.read_image(self.images_path[index])
        im=(1/255)*im
        im = im.repeat_interleave(3, dim=0)
        im=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation((-150, 150), fill=1),
            #transforms.RandomPerspective(distortion_scale=0.2, fill=1),
            transforms.Resize((self.final_size,self.final_size),interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])(im)
        return im,label_

    def get_labels(self):
        return self.labels

    def get_test_set(self):
        return self.test_images_path,self.test_labels
    def get_distribution(self):
        return dict(Counter(self.labels))
    def classes(self):
        indices = [[] for i in range(len(self.get_distribution().keys()))]
        for i in range(len(self.labels)):
            indices[self.labels[i]].append(i)
        return indices


#######################################################################

class SRTestDataset(Dataset):
    def __init__(self,path, max_size=300, final_size=64):
        self.images_path=[]
        self.max_h = max_size
        self.max_w = max_size
        self.final_size=final_size
        subfolder_images_names = glob.glob( os.path.join(path, '*.jpg')   )
        ## train
        self.images_path+= subfolder_images_names

    def __len__(self):
        return len(self.images_path)
    def __getitem__(self,index):
        im_name = self.images_path[index].split("/")[-1]
        im=torchvision.io.read_image(self.images_path[index], torchvision.io.ImageReadMode.GRAY)
        im=(1/255)*im
        im = im.repeat_interleave(3, dim=0)
        im=transforms.Compose([
            transforms.Resize((self.final_size,self.final_size),interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])(im)
        return im,im_name
