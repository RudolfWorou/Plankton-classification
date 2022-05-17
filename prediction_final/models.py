from torch import nn, zeros, repeat_interleave
import torchvision.models as models
from tqdm import tqdm
import os
import torch

#####################################################################################
#Build functions

def conv_relu_maxp(in_channels, out_channels, ks):
    return [nn.Conv2d(in_channels, out_channels,
                      kernel_size=ks,
                      stride=1,
                      padding=int((ks-1)/2), bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)]

def dropout_linear_relu(dim_in, dim_out, p_drop):
	return [nn.Dropout(p_drop),
		nn.Linear(dim_in, dim_out),
		nn.ReLU(inplace=True)]

#####################################################################################
#Models

class basicCNN(nn.Module):
    def __init__(self, images_size=64, nb_class=86, interpolation='bilinear'):
        super().__init__()

        #first convolution
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=images_size//2, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        #max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        #Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=images_size//2, out_channels=images_size, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        #max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        #fully connected 1
        self.fc1 = nn.Linear(images_size*169, nb_class)

    def forward(self, x):
        #convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        #max pool 1
        out = self.maxpool1(out)

        #convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)

        #max pool 2
        out = self.maxpool2(out)

        #flatten
        out = out.view(out.size(0), -1)

        #Linear function (readout)
        out = self.fc1(out)
        return out

class TP1CNN(nn.Module):
    def __init__(self, images_size=64, nb_class=86):
        super().__init__()

        self.features = nn.Sequential(
                    *conv_relu_maxp(1, images_size//4, 5),
                    *conv_relu_maxp(images_size//4, images_size//2, 5),
                    *conv_relu_maxp(images_size//2, images_size, 5)
                    )
        probe_tensor = zeros((1, 1, images_size, images_size))
        out_features = self.features(probe_tensor).view(-1)

        self.classifier = nn.Sequential(
                    *dropout_linear_relu(out_features.shape[0], 128, 0.3),
                    *dropout_linear_relu(128, 256, 0.3),
                    nn.Linear(256, nb_class)
                    )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)


class CNNModel(nn.Module):
    def __init__(self, images_size=64, nb_class=86):
        super().__init__()

        self.features = nn.Sequential(
                    *conv_relu_maxp(1, images_size//4, 3),
                    *conv_relu_maxp(images_size//4, images_size//2, 3),
                    *conv_relu_maxp(images_size//2, images_size, 3),
                    *conv_relu_maxp(images_size, images_size*2, 5),
                    *conv_relu_maxp(images_size*2, images_size, 5),
                    )
        probe_tensor = zeros((1, 1, images_size, images_size))
        out_features = self.features(probe_tensor).view(-1)

        self.classifier = nn.Sequential(
                    *dropout_linear_relu(out_features.shape[0], 128, 0.3),
                    *dropout_linear_relu(128, 256, 0.3),
                    nn.Linear(256, nb_class)
                    )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y

class CNNModel_L2(nn.Module):
    def __init__(self, l2_reg, images_size=64, nb_class=86):
        super().__init__()
        self.l2_reg = l2_reg
        self.features = nn.Sequential(
                    *conv_relu_maxp(1, images_size//4, 3),
                    *conv_relu_maxp(images_size//4, images_size//2, 3),
                    *conv_relu_maxp(images_size//2, images_size, 3),
                    *conv_relu_maxp(images_size, images_size*2, 5),
                    *conv_relu_maxp(images_size*2, images_size, 5),
                    )
        probe_tensor = zeros((1, 1, images_size, images_size))
        out_features = self.features(probe_tensor).view(-1)

        self.classifier = nn.Sequential(
                    *dropout_linear_relu(out_features.shape[0], 128, 0.),
                    *dropout_linear_relu(128, 256, 0.),
                    nn.Linear(256, nb_class)
                    )

    def penalty(self):
        return self.l2_reg * (self.classifier[1].weight.norm(2)+ self.classifier[4].weight.norm(2)+self.classifier[6].weight.norm(2))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y

################################################################
#Fine tuning VGG19

class MyVGG19(nn.Module):
    def __init__(self, images_size=64, nb_classes=86, freeze=False):
        super(MyVGG19, self).__init__()

        self.model = models.vgg19(pretrained=True)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        probe_tensor = zeros((1, 3, images_size, images_size))
        out_features = self.model(probe_tensor).view(-1)

        self.classifier = nn.Sequential(
                    *dropout_linear_relu(out_features.shape[0], nb_classes*5, 0.2),
                    nn.Linear(nb_classes*5, nb_classes)
                    )

    def forward(self, x):
        x = self.model(x)
        return self.classifier(x)

 ################################################################
#Fine tuning ResNet50

class MyResNet50(nn.Module):
    def __init__(self, images_size=64, nb_classes=86, freeze=False, l2=0.):
        super(MyResNet50, self).__init__()
        self.l2 = l2
        self.model = models.resnet50(pretrained=True)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        probe_tensor = zeros((1, 3, images_size, images_size))
        out_features = self.model(probe_tensor).view(-1)

        self.fc = nn.Linear(out_features.shape[0], nb_classes)

    def forward(self, x):
        x = self.model(x)
        return self.fc(x)

    def penalty(self):
        return self.l2*self.fc.weight.norm(2)



 ################################################################
#Fine tuning SqueezeNet


class SqueezeNet(nn.Module):
    def __init__(self, images_size=64, nb_classes=86, freeze=True, l2=0.):
        super(SqueezeNet, self).__init__()
        self.l2 = l2
        self.model = models.squeezenet1_1(pretrained=True)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        probe_tensor = zeros((1, 3, images_size, images_size))
        out_features = self.model(probe_tensor).view(-1)

        self.fc = nn.Linear(out_features.shape[0], nb_classes)

    def forward(self, x):
        x = self.model(x)
        return self.fc(x)

    def penalty(self):
        return self.l2*self.fc.weight.norm(2)

 ################################################################
#Fine tuning ResNet101 and 152

class MyResNet101(nn.Module):
    def __init__(self, images_size=64, nb_classes=86, freeze=False, l2=0.):
        super(MyResNet101, self).__init__()
        self.l2 = l2
        self.model = models.resnet101(pretrained=True)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        probe_tensor = zeros((1, 3, images_size, images_size))
        out_features = self.model(probe_tensor).view(-1)
        self.fcregul= nn.Dropout(0.2)
        self.fc = nn.Linear(out_features.shape[0], nb_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.fcregul(x)
        return self.fc(x)

    def penalty(self):
        return self.l2*self.fc.weight.norm(2)


class MyResNet101V2(nn.Module):
    def __init__(self, images_size=64, nb_classes=86, freeze=False, l2=0.):
        super(MyResNet101V2, self).__init__()
        self.l2 = l2
        self.model = models.resnet101(pretrained=True)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        probe_tensor = zeros((1, 3, images_size, images_size))
        out_features = self.model(probe_tensor).view(-1)
        self.classifier = nn.Sequential(
                    *dropout_linear_relu(out_features.shape[0], nb_classes*10, 0.3),
                    *dropout_linear_relu( nb_classes*10, nb_classes*5, 0.3),
                    nn.Linear(nb_classes*5, nb_classes)
                    )

    def forward(self, x):
        x = self.model(x)
        return self.classifier(x)

    def penalty(self):
        return self.l2*self.classifier[1].weight.norm(2)



class MyResNet152(nn.Module):
    def __init__(self, images_size=64, nb_classes=86, freeze=False, l2=0.):
        super(MyResNet152, self).__init__()
        self.l2 = l2
        self.model = models.resnet152(pretrained=True)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        probe_tensor = zeros((1, 3, images_size, images_size))
        out_features = self.model(probe_tensor).view(-1)
        #self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(out_features.shape[0], nb_classes)

    def forward(self, x):
        x = self.model(x)
        #x = self.dropout(x)
        return self.fc(x)

    def penalty(self):
        return self.l2*self.fc.weight.norm(2)
