from torchvision import datasets, models, transforms
import torch
from PIL import Image
from torch.autograd import Variable
import numpy as np
import os
from sklearn.metrics import confusion_matrix

model= models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)
model.load_state_dict(torch.load('resnet50_finalparams_(700_300).pkl'))
model = model.cuda()
model.eval()
classes = ["b","f","s","y"]
matri =  np.zeros((4, 4), dtype=np.int)
transform = transforms.Compose([transforms.CenterCrop(224),transforms.Resize(224), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#input_path = "b879.bmp"
input_path = "./test/"
outputs_label,true_label = [],[]
for img_num in os.listdir(input_path):
	print("img_num",img_num)
	img = Image.open(input_path+img_num).convert('RGB')
	img = transform(img)
	img = Variable(img).unsqueeze(0)
	outputs = model(img.cuda())
	preds = torch.max(outputs, 1)[1]
	print("classes[preds]",classes[preds])
	print("img_num[0]",img_num[0])

	if classes[preds]=='b':#修改第一行
		if img_num[0]=='b':
			matri[0][0]=matri[0][0]+1
		if img_num[0]=='f':
			matri[0][1]=matri[0][1]+1
		if img_num[0]=='s':
			matri[0][2]=matri[0][2]+1
		if img_num[0]=='y':
			matri[0][3]=matri[0][3]+1

	if classes[preds]=='f':#修改第二行
		if img_num[0]=='b':
			matri[1][0]=matri[1][0]+1
		if img_num[0]=='f':
			matri[1][1]=matri[1][1]+1
		if img_num[0]=='s':
			matri[1][2]=matri[1][2]+1
		if img_num[0]=='y':
			matri[1][3]=matri[1][3]+1

	if classes[preds]=='s':#修改第三行
		if img_num[0]=='b':
			matri[2][0]=matri[2][0]+1
		if img_num[0]=='f':
			matri[2][1]=matri[2][1]+1
		if img_num[0]=='s':
			matri[2][2]=matri[2][2]+1
		if img_num[0]=='y':
			matri[2][3]=matri[2][3]+1

	if classes[preds]=='y':#修改第四行
		if img_num[0]=='b':
			matri[3][0]=matri[3][0]+1
		if img_num[0]=='f':
			matri[3][1]=matri[3][1]+1
		if img_num[0]=='s':
			matri[3][2]=matri[3][2]+1
		if img_num[0]=='y':
			matri[3][3]=matri[3][3]+1

print("matri",matri)











