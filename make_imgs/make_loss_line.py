import csv
import os
import matplotlib.pyplot as plt

# 定位CSV文件
model_dir = "..\\models"
dataset_name = "SK17_224"

model_name1 =  "UNetWithSAResnet50EncoderandAGsSAP2_woDS\\log.csv"
model_name1 = os.path.join(model_dir, dataset_name + "_" + model_name1)
print(model_name1)

model_name2 =  "UNet_woDS\\log.csv"
model_name2 = os.path.join(model_dir, dataset_name + "_" + model_name2)

# 设置X，Y
epoch = []
train_loss1 = []
train_loss2 = []
val_loss1 = []
val_loss2 = []
# 读取csv 文件
with open(model_name1)as f:
    f_csv = csv.reader(f)
    for i,row in enumerate(f_csv):
        if i == 0:
            continue
        epoch.append(i)
        train_loss1.append(float(row[2]))
        val_loss1.append(float(row[4]))

with open(model_name2)as f:
    f_csv = csv.reader(f)
    for i,row in enumerate(f_csv):
        if i == 0:
            continue
        train_loss2.append(float(row[2]))
        val_loss2.append(float(row[4]))

plt.plot(epoch, train_loss1, label ="DSAU-Net train loss")
plt.plot(epoch, train_loss2, label ="U-Net train loss")
plt.plot(epoch, val_loss1, label ="DSAU-Net vaild loss")
plt.plot(epoch, val_loss2, label ="U-Net vaild loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.xlim(-1, 105)
plt.ylim(0, 1)
plt.legend()
plt.show()