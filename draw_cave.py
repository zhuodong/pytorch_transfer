import csv
from matplotlib import pyplot as plt

trainfile = './CSVdata/Oct16_19-31-04_-train_acc.csv'
validfile = './CSVdata/Oct16_19-31-04_-valid_acc.csv'
savepath = './CSVpic/'

trainname = trainfile.split('/',2)[::-1][0]#获取.csv数据名称用于保存图片Oct16_19-31-04_-valid_acc.csv'
#保存图片命名格式Oct16_19-31-04_
savename = savepath + trainname[::-1].split('-', 1)[-1][::-1]+'.jpg'
#csvname[::-1].split('.', 1)[-1][::-1]逆序的截取的方式,截取‘.’之前的部分,逆序分割选取第一个再逆序

"""获取trainfile的内容"""
trainfile = open(trainfile,'r')
train_reader = csv.reader(trainfile)
header_row = next(train_reader)#第一行的内容，表头
epoch, train_acc = [0], [0]
for row in train_reader:
    current_date = row[1]
    epoch.append(float(current_date)+1)
    high = row[2]
    train_acc.append(float(row[2][ :4])*100)
trainfile.close()

"""validname"""
validfile = open(validfile,'r')
valid_reader = csv.reader(validfile)
header_row = next(valid_reader)#第一行的内容，表头
#print(header_row)
epoch, valid_acc = [0], [0]
for row in valid_reader:
    current_date = row[1]
    epoch.append(float(current_date)+1)
    high = row[2]
    valid_acc.append(float(row[2][ :4])*100)
validfile.close()


print("epoch",epoch)
print("train_acc", train_acc)
print("valid_acc", valid_acc)

# 根据数据绘制图形
fig = plt.figure(dpi=128, figsize=(10,6))#绘制输出图片大小
plt.plot(epoch, train_acc, c='red')
plt.plot(epoch, valid_acc, c='blue')
plt.xlabel('epoch', fontsize=10)
plt.ylabel("acc(%)", fontsize=10)
plt.legend(('train_acc', 'valid_acc'))
#plt.title("Oct16_19-31-04_-valid_acc", fontsize=12)# 设置标题的格式
#fig.autofmt_xdate()
#plt.tick_params(axis='both', which='major', labelsize=8)#坐标轴上数据大小
plt.savefig(savename) # 在plt.show()之前调用plt.savefig(),否则会出现空白图片
plt.show()
print("保存成功")
