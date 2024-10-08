import sys
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch import optim, nn
from torch.utils.data import DataLoader
from pokemon0 import Pokemon
from torchvision.models import resnet18
from utils import Flatten
from mealpy import FloatVar
import pickle
import os
from copy import deepcopy
import hpelm
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
from sklearn.metrics import accuracy_score

PopSize = 10
DimSize = 10
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 30
MaxFEs = DimSize * 1000
curFEs = 0

# MaxIter = int(MaxFEs / PopSize * 2)
MaxIter = 20
curIter = 0

Pop = np.zeros((PopSize, DimSize))
Off = np.zeros((PopSize, DimSize))

FitPop = np.zeros(PopSize)
FitOff = np.zeros(PopSize)

FuncNum = 0

BestIndi = None
BestFit = float("inf")


# initialize the Pop randomly
def Initialization(func):
    global Pop, FitPop, curFEs, DimSize, BestIndi, BestFit
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = func(Pop[i])
        curFEs += 1
    BestFit = min(FitPop)
    BestIndi = deepcopy(Pop[np.argmin(FitPop)])


def CARIME(func):
    global Pop, FitPop, Off, FitOff, curIter, MaxIter, LB, UB, PopSize, DimSize, curFEs, BestIndi, BestFit
    Sequence = list(range(PopSize))
    np.random.shuffle(Sequence)
    w = 5
    Off = deepcopy(Pop)
    RIME_factor = np.random.uniform(-1, 1) * np.cos(np.pi * (curIter + 1) / (MaxIter / 10)) * (
            1 - np.round((curIter + 1) * w / MaxIter) / w)
    E = np.sqrt((curIter + 1) / MaxIter)
    NorFit = FitPop / np.linalg.norm(FitPop, axis=0, keepdims=True)
    for i in range(int(PopSize / 2)):
        idx1, idx2 = Sequence[2 * i], Sequence[2 * i + 1]
        if FitPop[idx1] < FitPop[idx2]:
            FitOff[idx1] = FitPop[idx1]
            Off[idx1] = deepcopy(Pop[idx1])
            for j in range(DimSize):
                if np.random.rand() < E:  # Soft RIME
                    Off[idx2][j] = BestIndi[j] + RIME_factor * (np.random.rand() * (UB[j] - LB[j]) + LB[j])
                if np.random.rand() < NorFit[idx2]:  # Hard RIME
                    Off[idx2][j] = BestIndi[j]
            Off[idx2] = np.clip(Off[idx2], LB, UB)
            FitOff[idx2] = func(Off[idx2])
            curFEs += 1
            if FitOff[idx2] < BestFit:
                BestFit = deepcopy(FitOff[idx2])
                BestIndi = deepcopy(Off[idx2])
        else:
            FitOff[idx2] = FitPop[idx2]
            Off[idx2] = deepcopy(Pop[idx2])
            for j in range(DimSize):
                if np.random.rand() < E:  # Soft RIME
                    Off[idx1][j] = BestIndi[j] + RIME_factor * (np.random.rand() * (UB[j] - LB[j]) + LB[j])
                if np.random.rand() < NorFit[idx1]:  # Hard RIME
                    Off[idx1][j] = BestIndi[j]
            Off[idx1] = np.clip(Off[idx1], LB, UB)
            FitOff[idx1] = func(Off[idx1])
            curFEs += 1
            if FitOff[idx1] < BestFit:
                BestFit = deepcopy(FitOff[idx1])
                BestIndi = deepcopy(Off[idx1])

    for i in range(PopSize):
        if FitOff[i] < FitPop[i]:
            Pop[i] = deepcopy(Off[i])
            FitPop[i] = deepcopy(FitOff[i])
def RunCARIME(func):
    global curFEs, curIter, MaxFEs, TrialRuns, Pop, FitPop, DimSize,Runs
    All_Trial_Best = []
    for i in range(TrialRuns):
        Best_list = []
        curFEs = 0
        curIter = 0
        Initialization(func)
        Best_list.append(BestFit)
        np.random.seed(2024 + 1996 * i)
        while curIter < MaxIter:
            CARIME(func)
            curIter += 1
            Best_list.append(BestFit)
            with open('log.txt', 'a') as f:
                # print("Iter=", curIter, ":fit=", BestFit, file=f)
                print("Iter=", curIter, ":fit=", BestFit, file=f)
        All_Trial_Best.append(Best_list)
    with open("./CARIME_Data/D.csv", 'a') as f:
        np.savetxt(f, All_Trial_Best, delimiter=",")
    # np.savetxt("./CARIME_Data/ " + "D.csv", All_Trial_Best, delimiter=",")
def evalute(model, loader):
    model.eval()
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()

    return correct / total
def getevaluteY(model, loader):
    pre_Y=[]
    Y = []
    model.eval()
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            # 将预测的Y和实际的Y追加到列表中
            pre_Y.extend(pred.cpu().numpy())
            Y.extend(y.cpu().numpy())
        correct += torch.eq(pred, y).sum().float().item()

    return pre_Y,Y

from collections import defaultdict

def default_dict_factory():
    return defaultdict(dict)

def generativeModel():
    global device ,x_train, y_train
    batchsz = 30
    lr = 1e-3
    epochs = 10
    num_cuda_devices = torch.cuda.device_count()
    print(f"当前系统上有 {num_cuda_devices} 个可用的CUDA设备。")

    # 指定要使用的CUDA设备
    desired_device_id = 0  # 选择要使用的设备的ID
    if desired_device_id < num_cuda_devices:
        torch.cuda.set_device(desired_device_id)
        print(f"已将CUDA设备切换到设备ID为 {desired_device_id} 的设备。")
    else:
        print(f"指定的设备ID {desired_device_id} 超出可用的CUDA设备数量。")
    device = torch.device('cuda:0')
    parent_dir=os.path.dirname(os.getcwd())
    # 获取当前脚本文件的绝对路径
    script_path = os.path.abspath(__file__)
    # 获取当前脚本文件的父文件夹
    cwd_dir = os.path.dirname(script_path)

    for ii in range(1):
        filemame=f"images.csv"

        train_db = Pokemon(parent_dir+'/workspace/tomato/tomatodata/tomato', filemame,224, mode='train')
        val_db = Pokemon(parent_dir+'/workspace/tomato/tomatodata/tomato', filemame,224, mode='val')
        test_db = Pokemon(parent_dir+'/workspace/tomato/tomatodata/tomato',filemame, 224, mode='test')
        train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True,
                                num_workers=4)
        val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=2)
        test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=2)
        # model = ResNet18(5).to(device)
        trained_model = resnet18(pretrained=True)
        model = nn.Sequential(*list(trained_model.children())[:-1], #[b, 512, 1, 1]
                            Flatten(), # [b, 512, 1, 1] => [b, 512]
                            nn.Linear(512, 10)
                            ).to(device)
        # x = torch.randn(2, 3, 552, 224)
        # print(model(x).shape)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criteon = nn.CrossEntropyLoss()


        best_acc, best_epoch = 0, 0
        global_step = 0
        # viz.line(np.array([0]), np.array([-1]), win='loss', opts=dict(title='loss'))
        # viz.line(np.array([0]), np.array([-1]), win='val_acc', opts=dict(title='val_acc'))
        for epoch in range(epochs):

            for step, (x,y) in enumerate(train_loader):

                # x: [b, 3, 224, 224], y: [b]
                x, y = x.to(device), y.to(device)

                model.train()
                logits = model(x)
                loss = criteon(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # viz.line(np.array([loss.item()]), np.array([global_step]), win='loss', update='append')
                global_step += 1

            if epoch % 1 == 0:

                val_acc = evalute(model, val_loader)
                if val_acc> best_acc:
                    best_epoch = epoch
                    best_acc = val_acc
                    dirp=cwd_dir
                    torch.save(model.state_dict(), f'{dirp}/bestmodel/best.mdl')

                    # viz.line(np.array([val_acc]), np.array([global_step]), win='val_acc', update='append')
                print("epoch:",{epoch},":best_acc",{best_acc})
        print('best acc:', best_acc, 'best epoch:', best_epoch)
        model.load_state_dict(torch.load(f'{dirp}/bestmodel/best.mdl'))
        print('loaded from ckpt!')
        test_acc = evalute(model, test_loader)
        print('test acc:', test_acc)

#直接读取生成模型的参数，直接预测
def main():
    import numpy as np
    global x_train, y_train,x_val,y_val,device,TZnum,concatenate_x_train,concatenate_y_train,tag,test,test_y,dim
    batchsz = 30
    device = torch.device('cuda:0')
    parent_dir=os.path.dirname(os.getcwd())
    # 获取当前脚本文件的绝对路径
    script_path = os.path.abspath(__file__)
    # 获取当前脚本文件的父文件夹
    cwd_dir = os.path.dirname(script_path)

    elm_acc=[]

    result = defaultdict(default_dict_factory)
    for ii in range(1):

        filemame=f"images.csv"

        train_db = Pokemon(parent_dir+'/workspace/tomato/tomatodata/tomato', filemame,224, mode='train')
        val_db = Pokemon(parent_dir+'/workspace/tomato/tomatodata/tomato', filemame,224, mode='val')
        test_db = Pokemon(parent_dir+'/workspace/tomato/tomatodata/tomato',filemame, 224, mode='test')

        train_loader = DataLoader(train_db, batch_size=100, shuffle=True,num_workers=4)
        val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=2)
        test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=2)
        model = resnet18(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1], #[b, 512, 1, 1]
                            Flatten(), # [b, 512, 1, 1] => [b, 512]
                            nn.Linear(512, 10)
                            ).to(device)
        #加载已经训练好的模型的值
        model.load_state_dict(torch.load(f'{cwd_dir}/bestmodel/best.mdl'))


        #resnet生成的特征的路径
        file_path = os.path.join(cwd_dir,f'data/x_train.pkl')
        file_path1 = os.path.join(cwd_dir,f'data/y_train.pkl')
        file_path2 = os.path.join(cwd_dir,f'data/x_val.pkl')
        file_path3 = os.path.join(cwd_dir,f'data/y_val.pkl')
        file_path4 = os.path.join(cwd_dir,f'data/test.pkl')
        file_path5 = os.path.join(cwd_dir,f'data/test_y.pkl')


        #如果路径特征存在则直接读取处理好的特征，否则使用resnet提取特征
        train,train_y=get_features(model, train_loader,file_path,file_path1)
        val, val_y = get_features(model, val_loader,file_path2,file_path3)
        test, test_y = get_features(model, test_loader,file_path4,file_path5)


        x_train=train
        y_train=train_y
        x_val, y_val=val,val_y
        test, test_y=test, test_y


        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(test_y, 10)
        y_val = to_categorical(val_y, 10)

        global d, n, m
        d = 4
        n = 512
        m = 10
        dimensions = d * n * n + d * n


        global FuncNum, DimSize, Pop, MaxFEs, MaxIter, LB, UB
        DimSize = dimensions
        Pop = np.zeros((PopSize, DimSize))
        # MaxFEs = dim * 1000
        MaxIter = 20
        LB = [0] * DimSize
        UB = [1] * DimSize
        FuncNum = 1

        for i in range(1):
            FuncNum = i + 1
            RunCARIME(objective_functionELM)
            BestIndi
        X=BestIndi
        W = X[:d * n * n].reshape(d * n, n)
        B = X[d * n * n:d * n * n + d * n].reshape(d * n, 1)
        elm = hpelm.ELM(n, m)
        # 添加一些神经元，指定激活函数
        elm.add_neurons(d * n, "sigm")
        elm.W = W
        elm.B = B
        elm.train(x_train, y_train)
        # 预测结果
        y_pred = elm.predict(test)  # 假设X是输入数据
        # 计算准确率
        acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        print("acc=",acc)


def convert_defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        return {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    else:
        return d
def objective_functionELM(particle):
    global d,n,m
    global x_train, y_train, x_val, y_val

    # 把粒子转换成权重和阈值矩阵
    W = particle[:d * n * n].reshape(d * n, n)
    B = particle[d * n * n:d * n * n + d * n].reshape(d * n, 1)
    elm = hpelm.ELM(n, m)
    elm.add_neurons(d * n, "sigm")
    elm.W=W
    elm.B=B
    elm.train(x_train, y_train,reg=0.02)
    # 预测结果
    y_pred = elm.predict(x_val)

    acc = accuracy_score(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))
    return -1 * acc


def get_features(model, train_loader, x_path, y_path):
    global device
    if (not os.path.exists(x_path)) or (not os.path.exists(y_path)):

        model0 = nn.Sequential(*list(model.children())[:-1],  # [b, 512, 1, 1]
                               Flatten(),  # [b, 512, 1, 1] => [b, 512]
                               ).to(device)
        model0.eval()

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                features = model0(x)
                if step == 0:
                    result = features
                    result_y = y
                else:
                    result = torch.cat([result, features], dim=0)
                    result_y = torch.cat([result_y, y], dim=0)
        result, result_y = result.cpu(), result_y.cpu()
        with open(x_path, 'wb') as file:
            pickle.dump(result, file)
        with open(y_path, 'wb') as file:
            pickle.dump(result_y, file)

        return result.numpy(), result_y.numpy()
    else:
        with open(x_path, 'rb') as file:
            result = pickle.load(file)
        with open(y_path, 'rb') as file:
            result_y = pickle.load(file)

        return result.numpy(), result_y.numpy()



if __name__ == '__main__':
    # generativeModel() #获取深度学习模型，保存到本地
    main() #执行CARIME_RESNET_ELM

