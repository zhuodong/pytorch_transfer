import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from tensorboardX import SummaryWriter
import time
import os

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for data in dataloders[phase]:
                inputs, labels = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                # forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.max(outputs, 1)[1]
                optimizer.zero_grad()
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]
            time_elapsed = time.time() - since
            print('{} Loss: {:.4f} Acc: {:.4f} time:{:.4f}'.format(phase, epoch_loss, epoch_acc, time_elapsed))
            if phase =='train':
                writer.add_scalar('train_loss', epoch_loss, epoch)
                writer.add_scalar('train_acc',  epoch_acc,epoch)
            if phase =='valid':
                writer.add_scalar('valid_loss', epoch_loss, epoch)
                writer.add_scalar('valid_acc',epoch_acc,epoch)
            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = (time.time() - since)
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    #torch.save(best_model_wts.state_dict(), 'params.pkl')
    torch.save(best_model_wts, 'vgg19_finalparams_(700_300).pkl')
    #保存模型
    writer.add_graph(model,(inputs,))
    return model

def load_data(datadir):
    data_transforms = {'train': transforms.Compose([transforms.CenterCrop(224),transforms.Resize(224), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
                       'valid': transforms.Compose([transforms.CenterCrop(224),transforms.Resize(224), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])}#transforms.CenterCrop(100)

    image_datasets = {x: datasets.ImageFolder(os.path.join(datadir, x), data_transforms[x]) for x in ['train', 'valid']}
    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=20,
                                                 shuffle=True,
                                                 num_workers=4) for x in ['train', 'valid']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    return dataloders,dataset_sizes



if __name__ == "__main__":
    #writer = SummaryWriter(log_dir='./log', comment='model')
    writer = SummaryWriter()
    
    """
    #选择网络结构进行训练，使用resnet18训练好的模型来初始化当前模型的参数
    model= models.resnet50(pretrained=True)#https://blog.csdn.net/jiangpeng59/article/details/79609392
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 4)
    model = model.cuda()
    """
    
    
    """
    #结果很差
    #冻结除最后一层(全连接层)之外的所有参数，然后小数据集训练时，只更新全连接层的参数。
    model_conv = torchvision.models.resnet34(pretrained=False) 
    for param in model_conv.parameters():         
        param.requires_grad = False # Parameters of newly constructed modules have requires_grad=True by  default 
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = torch.nn.Linear(num_ftrs, 4)
    model = model_conv.cuda()
    
    """
    
    model = models.vgg16(pretrained=True)
    #model = models.vgg11(pretrained=False) 
    model.classifier[6] = torch.nn.Linear(4096, 4)
    #print(model)    
    model = model.cuda()
    
    # your image data file
    data_dir = './train_valid_test(700_300) /'
    dataloders,dataset_sizes = load_data(data_dir)
    #优化器
    #optimizer = torch.optim.Adam(model.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
    #optimizer = torch.optim.ASGD(model.parameters(), lr=0.001, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)   
    criterion = torch.nn.CrossEntropyLoss()# define loss function
    # Observe that all parameters are being optimized
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    model = train_model(model=model,criterion=criterion,optimizer=optimizer,scheduler=lr_scheduler,num_epochs=50)
    
    writer.close()


   
