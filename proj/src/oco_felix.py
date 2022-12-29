#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
# warnings.filterwarnings('ignore')
# get_ipython().run_line_magic('matplotlib', 'inline')

import pathlib

pathlib.Path("./indivResults").mkdir(exist_ok=True,parents=True)
pathlib.Path("./img").mkdir(exist_ok=True,parents=True)




import urllib.request
import json

dataLoc = "/tmp/f002nb9/oco/mnist.pickle"
try:
    #https://pjreddie.com/media/files/mnist_train.csv
    with open(dataLoc,"rb") as fp:
        data = pickle.load(fp)
except Exception as ex:
    print("no mnist.pickle",ex)
    data = {}
    for name in ["test","train"]:
        with urllib.request.urlopen('https://pjreddie.com/media/files/mnist_'+name+'.csv') as link:
            lines = link.read().decode('utf-8').split("\n")
            rows = [{"label":int(row[0]),"vec":np.array(json.loads("["+row[2:]+"]"))} for row in lines[:-1]]
            df = pd.DataFrame(rows)
            data[name] = df
    for df in data.values():
        df["b_label"] = df.label.apply(lambda x: 2*int((x == 0))-1)
        df["vecNorm"] = df.vec.apply(lambda vec: np.concatenate([vec/255,[1]]))
    with open(dataLoc,"wb") as fp:
        pickle.dump(data, fp)


def show(row):
    plt.figure()
    # print(row.label)
    plt.imshow(row.vecNorm[:-1].reshape(28,28), cmap='gray')
    plt.title(str(row.label) + ";" + str(row.b_label))

# data["test"].sample(10).apply(show,axis=1)
#


def get_toy_data(n=1000):

    
    x1 = np.dstack([np.arange(10)]*n).squeeze().T.astype("float")
    x1 += np.random.randn(*x1.shape)

    x2 = np.dstack([np.arange(10)+2]*n).squeeze().T.astype("float")
    x2 += np.random.randn(*x2.shape)
    # x2

    x = np.concatenate([x1,x2]).T
    
    y = np.concatenate([np.ones(n),-1*np.ones(n)])
    return x, y



def proj_l1_ball_weighted(y, d=1, D = None):
    if np.sum(D*y) <= d:
        return y

    yFabs = np.fabs(D*y)#maybe D*np.fabs(y)? probably not necessary tho
    
    idx = np.argsort(yFabs)[::-1]
    D_invSorted = np.cumsum([d/D[i] for i in idx])
    inp_sorted = np.cumsum([yFabs[i] for i in idx])
    
    dVals = [yFabs[i]-(inp_sorted[i]-1)*D_invSorted[i]]
    d0 = np.argmax(dVals)
    theta0 = (inp_sorted[d0]-1)*D_invSorted[d0]
    
    ret = (d/D)*np.maximum(yFabs-theta0, 0)
    
    return ret*np.sign(y)
    

def proj_l1_ball(y, d=1):
    '''
    Intuition:
    excess = 1/j(1-∑ui) is the excess that has to be distributed over j coordinates
    if coordinate uj is so small compared to the excess that uj-excess is less than 0,
    then this coordinate gets zeroed out and we can stop
    '''
    if np.linalg.norm(y,1) <= d:
        return y

    yFabs = np.fabs(y)
    
    y_sorted = np.sort(yFabs)[::-1]
    # print("snort",y_sorted)

    num = (d-np.cumsum(y_sorted))
    denom = np.arange(len(y_sorted))+1
    # print("num",num,"denom", denom)
    y_sorted_cs = num/denom

    must_be_greater_than_0 = y_sorted + y_sorted_cs

    ro = np.where(must_be_greater_than_0 > 0)[-1][-1]
    # print(must_be_greater_than_0,"ro:",ro)
    lamb = y_sorted_cs[ro]
    # print(y,"lam",lamb)

    ret = np.maximum(yFabs+lamb, 0)

    return ret*np.sign(y)



def gradreg(w, x, y, lmbda):
    #w, x, y
    threshold =y * (x @ w)
    b = np.expand_dims(y, -1)
    gradient = -y * x
    gradient[threshold >= 1] = 0

    return gradient.mean(axis=0)+lmbda*w


def get_data(data, easyBin, quickie, fake):
    x_train = np.vstack(data["train"].vecNorm).T
    x_test = np.vstack(data["test"].vecNorm).T
    
    y_train = np.array(data["train"].b_label)
    y_test = np.array(data["test"].b_label)


    if easyBin:
        mask = data["train"]["label"].isin([1,0])
        x_train = x_train.T[mask].T
        y_train = y_train[mask]

        mask = data["test"]["label"].isin([1,0])
        x_test = x_test.T[mask].T
        y_test = y_test[mask]
        
    elif fake:
        x_train, y_train = get_toy_data()
        x_test, y_test = get_toy_data()

    idx = np.random.permutation(np.arange(x_train.shape[1]))

    x_train = x_train[:,idx]
    y_train = y_train[idx]#[y_train[i] for i in idx]


    if quickie:
        x_train = x_train[:,:quickie]
        y_train = y_train[:quickie]

    print("fake and bake",fake)
    
        
    return x_train, y_train, x_test, y_test


# In[86]:


classes = [-1,1]
def getAcc(preds, y):
    acc = 0

    for c in classes:
        thisAcc = np.sum(preds[y==c] == y[y==c])/(np.sum(y==c))
        acc += thisAcc

    accHarm = acc/len(classes)

    accSimple = np.sum(preds==y)/len(y)

    return accHarm, accSimple

def comp_grad_hinge(w, x_train, y_train, regLamb, d = -1):
    modOut = (w@x_train)
    loss = np.maximum(1-modOut*y_train,0)

    if d == -1:
        grad_hinge_w = ((-1*x_train*y_train).T[loss > 0]).sum(axis=0)/len(y_train)
        grad_hinge_w += regLamb*w
    else:
        grad_hinge_w = ((-1*x_train[d]*y_train).T[loss > 0]).sum()/len(y_train)
        grad_hinge_w += regLamb*w[d]
    return grad_hinge_w, loss


def initParams(x_train, descType, gamma = 1/8, projDim = 100):
    params = {}
    if descType == "gradDesc":
        w = np.random.randn(x_train.shape[0])
        params["w"] = w
        
    elif descType == "mirrDesc":
        w = np.zeros(x_train.shape[0])
        params["w"] = w
        
        w_y = w.copy()
        params["w_y"] = w_y
        
    elif descType == "expGrad":
        w = np.zeros(x_train.shape[0])
        params["w"] = w
       
        thetaPlus = np.zeros(x_train.shape[0])
        params["thetaPlus"] = thetaPlus
        
        thetaMinus = np.zeros(x_train.shape[0])
        params["thetaMinus"] = thetaMinus
        
    elif descType == "adaGrad":
        w = np.zeros(x_train.shape[0])
        params["w"] = w
        
        S = np.ones(x_train.shape[0])*gamma
        params["S"] =S
        
    elif descType == "newtonONS":
        w = np.zeros(x_train.shape[0])
        params["w"] = w
        
        A = np.eye(x_train.shape[0])*(gamma*gamma)
        params["A"] = A
        
        A_inv = np.eye(x_train.shape[0])/(gamma*gamma)
        params["A_inv"] = A_inv
        
    elif descType == "randExp":
        w_t = np.ones(x_train.shape[0]*2)/(x_train.shape[0]*2)
        params["w_t"] = w_t
        
        w = np.zeros(x_train.shape[0])
        params["w"] = w
        
    elif descType == "randExp":
        w_inter_minus = np.ones(x_train.shape[0])/(x_train.shape[0])
        params["w_inter_minus"] = w_inter_minus
        
        w_inter_plus = np.ones(x_train.shape[0])/(x_train.shape[0])
        params["w_inter_plus"] = w_inter_plus
        
        print("musarak",np.linalg.norm(params["w_inter_plus"]),np.linalg.norm(params["w_inter_minus"]))
        
        w = np.zeros(x_train.shape[0])
#         w /= np.sum(w)
        params["w"] = w
    
    elif descType == "bandExp":
        w_x = np.zeros(x_train.shape[0])
        params["w_x"] = w_x
        
        probDist = np.ones(x_train.shape[0]*2)/(x_train.shape[0]*2)
        params["probDist"] = probDist
        
        w_tPrime = np.ones(x_train.shape[0]*2)/(x_train.shape[0]*2)
        params["w_tPrime"] = w_tPrime
        
        
    print(params.keys())
    return params


def bandExpStep(batch_x, batch_y, params, lr, regLamb, projDim):
    
    w_x = params["w_x"]
    
    probDist = params["probDist"]
    w_tPrime = params["w_tPrime"]
    
#     print("dubs",sorted(w_x)[:5],sorted(w_x)[-5:],sorted(w_tPrime)[:5],sorted(w_tPrime)[-5:])
    
    dim = batch_x.shape[0]
    gamma = min(1, lr*dim)
    
    try:
        randCoord = np.random.choice(np.arange(dim*2),p=probDist)
    except Exception as e:
        print(list(probDist))
        print(list(w_tPrime))
        print(list(w_x))
        raise e
    
    isNeg = randCoord < dim
    grad_hinge_w, loss = comp_grad_hinge(w_x, batch_x, batch_y, regLamb, d=randCoord%dim)
    
    negCoef = ((-1)**isNeg)
    w_tPrime[randCoord] = np.exp(lr*negCoef*grad_hinge_w/probDist[randCoord])*w_tPrime[randCoord]
    
    w_tPrime = w_tPrime/np.sum(w_tPrime)
    probDist = (1-gamma)*w_tPrime+gamma/(2*dim)
    
    w_x = projDim*(w_tPrime[:dim]-w_tPrime[dim:])
    
    params["w_x"] = w_x
    params["probDist"] = probDist/np.sum(probDist)
    params["w_tPrime"] = w_tPrime

    return w_x, loss



def randExpStep(batch_x, batch_y, params, lr, regLamb, projDim):
    
    dim = batch_x.shape[0]
    randCoord = np.random.randint(dim)
    
    w = params["w"]
    w_t = params["w_t"]
    
    grad_hinge_w, loss = comp_grad_hinge(w, batch_x, batch_y, regLamb, d=randCoord)

    expFill = lr*dim*grad_hinge_w
    

    w_t[randCoord] = np.exp(-1*expFill)*w_t[randCoord]
    w_t[randCoord+dim] = np.exp(expFill)*w_t[randCoord+dim]

    w_t = w_t/np.sum(np.fabs(w_t))
    
    w = projDim*(w_t[:dim]-w_t[dim:])
    
    if np.random.rand()<0.01:
        print("dubs",sorted(w)[:5],sorted(w)[-5:])
   
    
    params["w_t"] = w_t
    params["w"] = w


    return w, loss


def expGradStep(batch_x, batch_y, params, lr, regLamb, projDim):
    grad_hinge_w, loss = comp_grad_hinge(params["w"], batch_x, batch_y, regLamb)

    thetaPlus = params["thetaPlus"] + grad_hinge_w * lr
    thetaMinus = params["thetaMinus"] - grad_hinge_w * lr

    w_yNumPlus = np.exp(thetaPlus)
    w_yNumMinus = np.exp(thetaMinus)
    w_yDenom = np.sum(w_yNumPlus)+np.sum(w_yNumMinus)
    
    w_yPlus = w_yNumPlus/w_yDenom
    w_yMinus = w_yNumMinus/w_yDenom

        
    w = (w_yMinus-w_yPlus)*projDim
    
    params["w"] = w
    params["thetaPlus"] = thetaPlus
    params["thetaMinus"] = thetaMinus
    
    return w, loss

def gradDescStep(batch_x, batch_y, params, lr, regLamb, projDim):
    
    grad_hinge_w, loss = comp_grad_hinge(params["w"], batch_x, batch_y, regLamb)

    w_y = params["w"] - grad_hinge_w * lr
    
    if projDim != -1:
        w = proj_l1_ball(w_y,d=projDim)
    else:
        w = w_y
        
    params["w"] = w
    return w, loss

def mirrDescStep(batch_x, batch_y, params, lr, regLamb, projDim):
    
    grad_hinge_w, loss = comp_grad_hinge(params["w"], batch_x, batch_y, regLamb)

    w_y = params["w_y"] - grad_hinge_w * lr
    params["w_y"] = w_y

    w = proj_l1_ball(w_y,d=projDim)
    params["w"] = w
    
    return w, loss

def adaGradStep(batch_x, batch_y, params, regLamb, projDim):
    
    grad_hinge_w, loss = comp_grad_hinge(params["w"], batch_x, batch_y, regLamb)
    
    S = params["S"] + np.power(grad_hinge_w,2) #(grad_hinge_w.reshape(-1,1)@grad_hinge_w.reshape(1,-1))
        
    D = np.sqrt(S)#np.diag(

    y = params["w"] - grad_hinge_w/D
    

#    w = proj_l1_ball(y,d=projDim)
    w = proj_l1_ball_weighted(y, D = D, d = projDim)
    params["w"] = w
    params["S"] = S
    
    return w, loss

def newtonONSStep(batch_x, batch_y, params, regLamb, projDim, gamma):
    #ons lol
    
    grad_hinge_w, loss = comp_grad_hinge(params["w"], batch_x, batch_y, regLamb)
    
    gradMatrix = grad_hinge_w.reshape(-1,1)@grad_hinge_w.reshape(1,-1)


    A = params["A"] + gradMatrix
    
    
    prevA_inv = params["A_inv"]
    currA_inv_num = prevA_inv@gradMatrix@prevA_inv
    currA_inv_denom = 1 + grad_hinge_w@prevA_inv@grad_hinge_w
    
    currA_inv = prevA_inv - currA_inv_num/currA_inv_denom
    
    y = params["w"] - gamma*currA_inv@grad_hinge_w

    w = proj_l1_ball_weighted(y,D = A,d=projDim)
    params["w"] = w
    params["A"] = A
    params["A_inv"] = currA_inv
    
    return w, loss





def get_lr(descType, lrStrat, epoch, lr, d):
    if descType not in ["adaGrad"]:
        if lrStrat == "epochPro":
            if descType == "randExp":
                lr = np.sqrt(1/((1+epoch)*d))
            elif descType == "bandExp":
                lr = np.sqrt(1/((1+epoch)*d))
            elif epoch < 10:
                lr = 1
            elif descType == "gradDesc":
                lr = 1/(epoch+1)
            elif descType == "mirrDesc":
                lr = 1/(np.sqrt(epoch+1))
            elif descType == "expGrad":
                lr = 1/(np.sqrt(epoch+1)) 
    return lr


from tqdm import tqdm 
import time
from functools import partial
tqdm = partial(tqdm, position=0, leave=True)



def gradient_descent(data, lrStrat = "epochPro", n_epochs = 100, batch_size = -1, regLamb = 1,fake = False,
                     easyBin = False, projDim = -1, decInterval = 200, quickie = 0, descType ="gradDesc", gamma=1/8):
    '''
    :param data: dataframe containing train and test data
    :param lrStrat: learning rate for Gradient Descent. if "epochPro", then lr is proportional to epoch. Otherwise a scalar
    :param n_epochs: how many epochs to train for
    :param batch_size: how many images to use for each batch. If -1, then non-stochastic (i.e. all images are calculated at once)
    :param regLamb: regularization parameter for L2 regularization
    :param fake: use artificial, fairly linearly data to check whether the algorithm is working
    :param easyBin: use only two numbers, 1 and 8, which are pretty different and thus should get a higher accuracy
    :param projDim: dimension of L1 ball projection. If -1, no projection will be done
    :param decInterval: if lr is not "epochPro", then decrease the interval by one tenth after each decInterval epochs
    :param quickie: if quickie > 0, then only quickie many images will be taken
    '''

    x_train, y_train, x_test, y_test = get_data(data, easyBin, False, fake)
    
    params = initParams(x_train, descType)
    
    allLosses = []
    allAccsTrain = []
    allAccsTrainSimple = []
    allAccsTestSimple = []
    allAccsTest = []

    
    epochCounter = 0
    
    if batch_size == -1:
        batch_size = x_train.shape[1]
        
    if "lrStrat" != "epochPro":
        lr = lrStrat
        
    numBatches = int(np.ceil(len(x_train)/batch_size))
    
    print("iterating",descType,"for ",n_epochs,"with batch_size",batch_size)
    
    runtime = time.time()

    for epoch in tqdm(range(n_epochs)):
        
        epochLosses = []
        epochAccsTrain = []
        epochAccsTrainSimple = []
        epochAccsTestSimple = []
        epochAccsTest = []
        
        idxEpoch = np.random.permutation(np.arange(x_train.shape[1]))
        
        lr = get_lr(descType, lrStrat, epoch, lr, d = x_train.shape[0])

        
        for batch in range(numBatches):
            
            idx = idxEpoch[batch*batch_size:(batch+1)*batch_size]
            
            
            batch_x = x_train[:,idx]
            batch_y = y_train[idx]

            
            
            if descType == "gradDesc":
                w, loss = gradDescStep(batch_x, batch_y, params, lr, regLamb, projDim)
        
            elif descType == "mirrDesc":
                w, loss = mirrDescStep(batch_x, batch_y, params, lr, regLamb, projDim)
            
            elif descType == "expGrad":
                w, loss = expGradStep(batch_x, batch_y, params, lr, regLamb, projDim)
                
            elif descType == "adaGrad":
                w, loss = adaGradStep(batch_x, batch_y, params, regLamb, projDim)#no lr!
                
            elif descType == "newtonONS":
                w, loss = newtonONSStep(batch_x, batch_y, params, regLamb, projDim, gamma)
                
            elif descType == "randExp":
                w, loss = randExpStep(batch_x, batch_y, params, lr, regLamb, projDim)
            
            elif descType == "bandExp":
                w, loss = bandExpStep(batch_x, batch_y, params, lr, regLamb, projDim)
                
            else:
                raise Exception("Not implemented",descType)

            predsTrain = np.sign(w@x_train)
            predsTest = np.sign(w@x_test)

            accuracyTrain, accTrainSimple = getAcc(predsTrain, y_train)
            accuracyTest, accTestSimple = getAcc(predsTest, y_test)

            
            epochLosses.append(np.sum(loss))
            epochAccsTrain.append(accuracyTrain)
            epochAccsTest.append(accuracyTest)

            epochAccsTrainSimple.append(accTrainSimple)
            epochAccsTestSimple.append(accTestSimple)
            
            
        allLosses.append(np.mean(epochLosses))
        allAccsTrain.append(np.mean(epochAccsTrain))
        allAccsTest.append(np.mean(epochAccsTest))

        allAccsTrainSimple.append(np.mean(epochAccsTrainSimple))
        allAccsTestSimple.append(np.mean(epochAccsTestSimple))

        epochCounter += 1
        if lrStrat != "epochPro" and epochCounter >= decInterval:
            epochCounter = 0
            lr *= 0.1
            print("learnedddd")

        print("accuracy train",epoch,round(allAccsTrain[-1],3), "accuracy test",round(allAccsTest[-1],3),"loss train",round(allLosses[-1],2))

        if quickie > 0 and epoch >= quickie:
            break;

    if not fake and not quickie:
        data["test"]["preds"] = data["test"].vecNorm.apply(lambda x: np.sign(w@x))
        
    runtime = time.time()-runtime
    d = {}
    d["runtime"] = runtime
    d["loss"] = allLosses
    d["accTrain"] = allAccsTrain
    d["accTest"] = allAccsTest
    d["accTestSimple"] = allAccsTestSimple
    d["accTrainSimple"] = allAccsTrainSimple
    d["w_size"] = np.linalg.norm(w)
    d["regLamb"] = regLamb
    d["projDim"] = projDim
    d["batch_size"] = batch_size
    d["descType"] = descType

    title = ""
    if batch_size > 0:
        title += "Stochastic (" + str(batch_size) + ") "
    else:
        title += "Non-Stochastic "
    if projDim != -1:
        title += "with projection (" + str(projDim)+")"
    else:
        title += "without projection"
    plot_results(d, title)
    
    return d




def plot_results(d, title):
    allLosses, allAccsTrain, allAccsTrainSimple,\
    allAccsTest, allAccsTestSimple = d["loss"], d["accTrain"],\
                                     d["accTrainSimple"], d["accTest"], d["accTestSimple"]

    plt.figure()
    ax = plt.gca()
    ls = plt.plot(allLosses,label="loss")

    twinx = ax.twinx()
    tr = twinx.plot(allAccsTrainSimple,color="green",label="weighted train accuracy",linestyle='dashed')
    trh = twinx.plot(allAccsTrain,label="harmonic train accuracy",color="green")
    
    ts = twinx.plot(allAccsTestSimple,color="red",label="weighted test accuracy",linestyle='dashed')
    tsh = twinx.plot(allAccsTest,label="harmonic test accuracy",color="red",)

    lns = ls + trh + tr + tsh + ts
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs,loc="lower right")
    plt.title(title)

    plt.savefig("./img/"+d["descType"]+"_"+str(d["projDim"])+"_"+str(d["projDim"])+"_"+str(time.time()));


def write(d):
    with open("./results.pickle","rb") as fp:
        results = pickle.load(d)

    results.append(d)
    with open("./results.pickle","wb") as fp:
        pickle.dump(results, fp);

    path = str(time.time())
    with open("./indivResults/"+path, "wb") as fp:
        pickle.dump(d,fp);

