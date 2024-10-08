import os
from copy import deepcopy
import numpy as np
from cec17_functions import cec17_test_func


PopSize = 50
DimSize = 10
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 30
MaxFEs = DimSize * 1000
curFEs = 0

MaxIter = int(MaxFEs / PopSize * 2)
curIter = 0

Pop = np.zeros((PopSize, DimSize))
Off = np.zeros((PopSize, DimSize))

FitPop = np.zeros(PopSize)
FitOff = np.zeros(PopSize)

FuncNum = 0

BestIndi = None
BestFit = float("inf")


def fitness(X):
    global DimSize, FuncNum
    f = [0]
    cec17_test_func(X, f, DimSize, 1, FuncNum)
    return f[0]


# initialize the Pop randomly
def Initialization():
    global Pop, FitPop, curFEs, DimSize, BestIndi, BestFit
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = fitness(Pop[i])
        curFEs += 1
    BestFit = min(FitPop)
    BestIndi = deepcopy(Pop[np.argmin(FitPop)])


def CFRIME():
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
                if np.random.uniform(-1, 1) < NorFit[idx2]:  # Hard RIME
                    Off[idx2][j] = BestIndi[j]
            Off[idx2] = np.clip(Off[idx2], LB, UB)
            FitOff[idx2] = fitness(Off[idx2])
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
            FitOff[idx1] = fitness(Off[idx1])
            curFEs += 1
            if FitOff[idx1] < BestFit:
                BestFit = deepcopy(FitOff[idx1])
                BestIndi = deepcopy(Off[idx1])

    for i in range(PopSize):
        if FitOff[i] < FitPop[i]:
            Pop[i] = deepcopy(Off[i])
            FitPop[i] = deepcopy(FitOff[i])


def RunCFRIME():
    global curFEs, curIter, MaxFEs, TrialRuns, Pop, FitPop, DimSize
    All_Trial_Best = []
    for i in range(TrialRuns):
        Best_list = []
        curFEs = 0
        curIter = 0
        Initialization()
        Best_list.append(BestFit)
        np.random.seed(2024 + 1996 * i)
        while curIter < MaxIter:
            CFRIME()
            curIter += 1
            Best_list.append(BestFit)
        All_Trial_Best.append(Best_list)
    np.savetxt("./CFRIME_Data/CEC2017/F" + str(FuncNum) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")


def main(dim):
    global FuncNum, DimSize, Pop, MaxFEs, MaxIter, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    MaxIter = int(MaxFEs / PopSize * 2)
    LB = [-100] * dim
    UB = [100] * dim
    FuncNum = 1
    for i in range(1, 31):
        FuncNum = i
        if i == 2:
            continue
        RunCFRIME()


if __name__ == "__main__":
    if os.path.exists('./CFRIME_Data/CEC2017') == False:
        os.makedirs('./CFRIME_Data/CEC2017')
    Dims = [30, 50]
    for Dim in Dims:
        main(Dim)
