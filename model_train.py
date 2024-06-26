import copy
import time

import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from google_model import GoogLeNet,Inception
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train_val_data_process():

    Root = "cat_dog_data/training_set"
    nomalize = transforms.Normalize([0.488, 0.455, 0.417],[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),nomalize])
    train_data = ImageFolder(root=Root,transform=transform)

    # 数据集划分
    train_data,val_data = Data.random_split(train_data,[round(0.8*len(train_data)),round(0.2*len(train_data))])
    #加载训练数据到数据容器
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=36,
                                       shuffle=True,
                                       num_workers=3)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size=36,
                                       shuffle=True,
                                       num_workers=3)
    return train_dataloader,val_dataloader
train_val_data_process()


# 模型训练
def train_model_process(model,train_dataloader,val_dataloader,num_epochs):

    device = torch.device("cuda")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []

    since = time.time()
    writer = SummaryWriter()

    for epoch in range(num_epochs):
        # print(f"第 {epoch+1}/{num_epochs} 轮训练开始,使用{device}进行训练")
        # print("-"*20)

        train_loss = 0.0
        train_corrects = 0

        val_loss = 0.0
        val_corrects = 0

        train_num = 0
        val_num = 0

        # 使用 tqdm 包装训练数据加载器
        train_dataloader_tqdm = tqdm(train_dataloader, desc=f"开始训练 Epoch {epoch + 1}/{num_epochs}")

        for step,(b_x,b_y) in enumerate(train_dataloader_tqdm):

            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.train()

            output = model(b_x)

            pre_label = torch.argmax(output, dim=1)

            loss = criterion(output, b_y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss+= loss.item() * b_x.size(0)

            train_corrects += torch.sum(pre_label == b_y.data)

            train_num += b_x.size(0)

            # 更新 tqdm 描述信息
            train_dataloader_tqdm.set_postfix(loss=loss.item(), acc=train_corrects.double().item() / train_num)

        # 使用 tqdm 包装验证数据加载器
        val_dataloader_tqdm = tqdm(val_dataloader, desc=f"开始测试 Epoch {epoch + 1}/{num_epochs}")

        for step, (b_x, b_y) in enumerate(val_dataloader_tqdm):

            s = time.time()

            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)

            val_loss += loss.item() * b_x.size(0)
            val_corrects += torch.sum(pre_lab==b_y.data)
            val_num += b_x.size(0)
            # 更新 tqdm 描述信息
            val_dataloader_tqdm.set_postfix(loss=loss.item(), acc=val_corrects.double().item() / val_num)

        # 把每一轮的损失加到损失列表里
        train_loss_all.append(train_loss / train_num)
        # 把每一轮的正确率加到正确率列表里
        train_acc_all.append(train_corrects.double().item() / train_num)

        writer.add_scalar("Loss/Train", train_loss/train_num, epoch+1)
        writer.add_scalar("acc/Train", train_corrects.double().item()/train_num, epoch+1)

        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        writer.add_scalar("Loss/Val", val_loss / val_num, epoch+1)
        writer.add_scalar("acc/Val", val_corrects.double().item() / val_num, epoch+1)

        tqdm.write(f"第 {epoch+1} 轮 Train Loss: {train_loss_all[-1]:.4f} Train Acc: {train_acc_all[-1]:.4f}")
        tqdm.write(f"第 {epoch+1} 轮 Val Loss: {val_loss_all[-1]:.4f} Val Acc:{val_acc_all[-1]:.4f}")

        tqdm.write(f"----------第 {epoch+1} 轮训练完成----------")

        #找出最优的准确度
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_use = time.time() - since
        tqdm.write(f"训练耗费时间为：{time_use//60:.0f}m{time_use%60:.0f}s")

    # 选择最优参数
    # 加载最高准确率下的模型参数
    # model.load_state_dict(best_model_wts)
    torch.save(best_model_wts,'best_model.pth')


    train_process = pd.DataFrame(data={"epoch":range(num_epochs),
                                       "train_loss_all":train_loss_all,
                                       "val_loss_all":val_loss_all,
                                       "train_acc_all":train_acc_all,
                                       "val_acc_all":val_acc_all})
    return train_process

# def matplot_acc_loss(train_process):
#     plt.figure(figsize=(12,4))
#     plt.subplot(1,2,1)
#     plt.plot(train_process["epoch"],train_process.train_loss_all,'ro-',label="train loss")
#     plt.plot(train_process["epoch"],train_process.val_loss_all,'bs-',label="val loss")
#     plt.legend()
#     plt.xlabel("epoch")
#     plt.ylabel("loss")
#
#     plt.subplot(1,2,2)
#     plt.plot(train_process["epoch"],train_process.train_acc_all,'ro-',label="train loss")
#     plt.plot(train_process["epoch"],train_process.val_acc_all,'bs-',label="val loss")
#     plt.legend()
#     plt.xlabel("epoch")
#     plt.ylabel("acc")

if __name__ == "__main__":
    google = GoogLeNet(Inception)
    train_dataloader,val_dataloader = train_val_data_process()
    train_process = train_model_process(google,train_dataloader,val_dataloader,20)
    # matplot_acc_loss(train_process)








