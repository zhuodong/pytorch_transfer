#https://blog.csdn.net/shine19930820/article/details/78335550
from torchvision import datasets, models, transforms
import torch
from PIL import Image
from torch.autograd import Variable
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

model= models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)
model.load_state_dict(torch.load('./pkl/resnet50_finalparams_(700_300).pkl'))
model = model.cuda()
model.eval()
classes = ["b","f","s","y"]
matri =  np.zeros((4, 4), dtype=np.int)
transform = transforms.Compose([transforms.CenterCrop(224),transforms.Resize(224), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#input_path = "b879.bmp"
input_path = "./test/"
outputs_label,true_label ,output= [],[],[]
for img_num in os.listdir(input_path):
	#print("img_num",img_num)
	img = Image.open(input_path+img_num).convert('RGB')
	img = transform(img)
	img = Variable(img).unsqueeze(0)
	outputs = model(img.cuda())
	preds = torch.max(outputs, 1)[1]

	#x = (outputs.cpu()).detach().numpy()
	#print(np.exp(x)/np.sum(np.exp(x),axis=0))#softmax()
	#break

	if img_num[0]=='b':
             true_label.append(0)
	if img_num[0]=='f':
             true_label.append(1)
	if img_num[0]=='s':
             true_label.append(2)
	if img_num[0]=='y':
             true_label.append(3)
	outputs_label.append(preds.item())
	
#print("outputs_label",outputs_label)#预测标签
#print("true_label",true_label)#真实标签





confusion_matrix=confusion_matrix(outputs_label,true_label)
print("confusion_matrix",confusion_matrix)
print(classification_report(true_label, outputs_label))#直接输出各个类的precision recall f1-score support
"""
##绘制ROC曲线
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, auc
 # y_test：实际的标签, dataset_pred：预测的概率值。 
y_test = true_label
dataset_pred = outputs_label
fpr, tpr, thresholds = roc_curve(y_test, dataset_pred)
roc_auc = auc(fpr, tpr) 
#画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来  
plt.plot(fpr, tpr, lw=1, label='ROC(area = %0.2f)' % (roc_auc)) 
plt.xlabel("FPR (False Positive Rate)") 
plt.ylabel("TPR (True Positive Rate)") 
plt.title("Receiver Operating Characteristic, ROC(AUC = %0.2f)"% (roc_auc)) 
plt.show()


"""







