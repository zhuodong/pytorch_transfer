import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from tensorboardX import SummaryWriter
import time
import os
import cv2
import numpy as np
from PIL import Image



# 网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            # torch.nn.BatchNorm2d(16),
            # torch.nn.Dropout(0.5),
            # torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            # torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            # torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(9216, 128),
            # torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 4)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)

        return out, conv1_out


def load_data(datadir):
    data_transforms = {'train': transforms.Compose(
        [transforms.CenterCrop(100), transforms.Resize(100), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                       'valid': transforms.Compose(
                           [transforms.CenterCrop(100), transforms.Resize(100), transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    image_datasets = {x: datasets.ImageFolder(os.path.join(datadir, x), data_transforms[x]) for x in ['train', 'valid']}
    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=1,
                                                 shuffle=True,
                                                 num_workers=4
                                                 ) for x in ['train', 'valid']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    return dataloders, dataset_sizes


if __name__ == "__main__":

    model = Net().cuda()
    data_dir = './data/'
    transform = transforms.Compose([transforms.CenterCrop(100), transforms.Resize(100), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #https://www.jianshu.com/p/2fe73baa09b8?utm_source=oschina-app
    input_path = "./data/train/bingtiao/28.bmp"
    input_img = Image.open(input_path).convert('RGB')
    input_img.save('./img/cov_img/00.bmp')
    input_img = transform(input_img)
    input_img = Variable(input_img).unsqueeze(0)
    outputs, conv1_out = model(input_img.cuda())
    print("inputs", input_img.size())  # [1, 3, 100, 100]
    print("conv1_out", conv1_out.size())  # [1, 32, 50, 50]
    print("num_kernel",conv1_out.shape[1])
    for num in range(conv1_out.shape[1]):
        feature = conv1_out[:, num, :, :]  # feature.shape = [1, 16, 16]
        # print("feature",feature.size())
        feature = feature.view(feature.shape[1], feature.shape[2])  # [16, 16]
        # print("feature.size", feature.size())
        feature = (feature.cpu()).data.numpy()
        feature = 1.0 / (1 + np.exp(-1 * feature))  # use sigmod to [0,1]
        feature = np.round(feature * 255)  # [0,255]
        covpath = './img/cov_img/'+ str(num) + ".jpg"
        print("covpath", covpath)
        cv2.imwrite(covpath, feature)

    print("finish!")

