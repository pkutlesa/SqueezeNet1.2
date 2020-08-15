import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adadelta
from torch.autograd import Variable
import numpy as np


############################################################ MODELS ############################################################

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class FirePReLu(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.PReLU()
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.PReLU()
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.PReLU()

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version=1.1, num_classes=10):
        super(SqueezeNet, self).__init__()
        if version not in [1.1, 1.2]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.1 or 1.2 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.1:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5, stride=2),
                nn.ReLU(inplace=True)
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13, stride=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


class SqueezeNetPReLu(nn.Module):

    def __init__(self, version=1.1, num_classes=10):
        super(SqueezeNet, self).__init__()
        if version not in [1.1, 1.2]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.1 or 1.2 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.1:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.PReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                FirePReLu(64, 16, 64, 64),
                FirePReLu(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                FirePReLu(128, 32, 128, 128),
                FirePReLu(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                FirePReLu(256, 48, 192, 192),
                FirePReLu(384, 48, 192, 192),
                FirePReLu(384, 64, 256, 256),
                FirePReLu(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5, stride=2),
                nn.PReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                FirePReLu(64, 16, 64, 64),
                FirePReLu(128, 16, 64, 64),
                FirePReLu(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                FirePReLu(256, 32, 128, 128),
                FirePReLu(256, 48, 192, 192),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                FirePReLu(384, 64, 256, 256),
                FirePReLu(512, 64, 256, 256),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.PReLU(),
            nn.AvgPool2d(13, stride=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


########################################################### TRAINING ###########################################################

def save_models(epoch):
    torch.save(model.state_dict(), "cifar10model_{}.model".format(epoch))
    print("Checkpoint saved")


def test():
    model.eval()
    test_acc = 0.0
    for i, (images, labels) in enumerate(test_loader):
      
        if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

        # Predict classes using images from the test set
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        test_acc += torch.sum(prediction == labels.data)
        
    # Compute the average acc and loss over all 10000 test images
    test_acc = test_acc / 10000

    return test_acc

def train(num_epochs):
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            # Move images and labels to gpu if available
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            # Clear all accumulated gradients
            optimizer.zero_grad()
            # Predict classes using images from the test set
            outputs = model(images)
            
            # Compute the loss based on the predictions and actual labels
            loss = loss_fn(outputs,labels)
            
            # Backpropagate the loss
            loss.backward()

            # Adjust parameters according to the computed gradients
            optimizer.step()

            train_loss += loss.cpu().data * images.size(0)
            _, prediction = torch.max(outputs.data, 1)
            
            train_acc += torch.sum(prediction == labels.data)

        # Compute the average acc and loss over all 50000 training images
        train_acc = train_acc / 50000
        train_loss = train_loss / 50000

        # Evaluate on the test set
        test_acc = test()

        # Save the model if the test acc is greater than our current best
        if test_acc > best_acc:
            save_models(epoch)
            best_acc = test_acc

        # Print the metrics
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {}".format(epoch, train_acc, train_loss,test_acc))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


############################################################# MAIN #############################################################

# Define transformations for the training set, flip the images randomly, crop out and apply mean and std normalization
transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load the training set
train_set = CIFAR10(root="./data",train=True,transform=transformations,download=True)
# Create a loder for the training set
train_loader = DataLoader(train_set,batch_size=256,shuffle=True,num_workers=4)

# Load the test set, note that train is set to False
test_set = CIFAR10(root="./data", train=False, transform=transformations, download=True)
# Create a loder for the test set, note that both shuffle is set to false for the test loader
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)

# Create model, optimizer and loss function
# model = AlexNet(num_classes=10)
model = SqueezeNet(version=1.1, num_classes=10)
# model = SqueezeNet(version=1.2, num_classes=10)
# model = SqueezeNetPReLu(version=1.1, num_classes=10)
# model = SqueezeNetPReLu(version=1.2, num_classes=10)

if torch.cuda.is_available():
    model.cuda()

optimizer = Adadelta(model.parameters())
loss_fn = nn.CrossEntropyLoss()

# Print the number of model parameters
print(count_parameters(model))

# Train the model for 100 epochs
train(100)




