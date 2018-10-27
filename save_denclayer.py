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
import numpy as np



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
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 4)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)

        return out, res

if __name__ == "__main__":

    model = Net().cuda()
    
    transform = transforms.Compose([transforms.CenterCrop(100), transforms.Resize(100), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #https://www.jianshu.com/p/2fe73baa09b8?utm_source=oschina-app
    data_dir = "./train_valid_test(600_400)/train"
    labelfile = open('train_dense_label.txt', 'w') 
    densefile = open('train_dense.txt', 'w')  
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for filepath in filenames:
            input_path = os.path.join(dirpath, filepath)
            #print("input_path",input_path)
            #input_path = "./28.bmp"
            input_img = Image.open(input_path).convert('RGB')
	    #input_img.save('./img/cov_img/00.bmp'
            input_img = transform(input_img)
            input_img = Variable(input_img).unsqueeze(0)
            outputs, res = model(input_img.cuda())
	    #保存权链接层数据
            
            print("res",res.size())
            for ip in res:  
                densefile.write(str((ip.cpu()).detach().numpy()))  
                densefile.write('\n')  
               
            #print("input_path.split",input_path[::-1].split('/',3)[1])
            #print("filenames",filenames)
            for ip in res:  
                labelfile.write(input_path[::-1].split('/',3)[1])  
                labelfile.write('\n')  

    labelfile.close() 
    densefile.close() 
 
	 

    

