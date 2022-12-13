import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import model.DeepSleepNet
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

device='cuda'

def model_test(model,test_dataloader):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.cuda()
            y=y.cuda()
            out = model(x)
            pred = out.argmax(dim=1)
            correct = pred.eq(y).sum().float().item()
            total_correct += correct
        total = len(test_dataloader.dataset)
        acc = total_correct / total
    # 准确率
    print("acc %f"%acc)
    return acc
def plot_loss_curve(loss_list,model_name):
    plt.clf()
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig("log/"+model_name+"loss")

def plot_acc_curve(acc_list,model_name):
    plt.clf()
    plt.figure(figsize=(10, 6))
    plt.plot(acc_list, label='acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig("log/"+model_name+"_acc")
def train(model,epochs,train_loader,test_loader,model_name):
    model.train()
    loss_fn=nn.CrossEntropyLoss()
    loss_fn=loss_fn.cuda()
    train_loss=[]
    acc_list=[]
    learning_rate=1e-4
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,betas=(0.9,0.999))
    # writer=SummaryWriter("log/"+model_name)
    for epoch in range(epochs):
        # print("train epoch {}".format(epoch))
        train_loss_sum=0
        n=0
        c=0
        for idx,(x,y) in enumerate(train_loader):
            torch.cuda.empty_cache()
            x=x.cuda()
            y=y.cuda()
            optimizer.zero_grad()

            pred=model(x)
            loss=loss_fn(pred,y.long())
            loss.backward()
            optimizer.step()
            c+=1
            train_loss_sum+=loss.item()
        train_loss.append(train_loss_sum/c)
        print(('epoch %d, train loss %f'% (epoch + 1, train_loss_sum/c)))
        torch.save(model.state_dict(), model_name+".pt")
        if((epoch+1)%5==0):
            acc=model_test(model,test_loader)
            acc_list.append(acc)

    return train_loss,acc_list




def train_salient(model,epochs,dataloader):
    loss_fn=nn.CrossEntropyLoss()
    loss_fn=loss_fn.cuda()
    b1 = 0.9
    b2 = 0.999
    learning_rate=0.001
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,betas=(b1,b2))

    for epoch in range(epochs):
        for idx,(x,y) in enumerate(dataloader):
            EEG=x[:,0:1].cuda()
            EOG=x[:,1:2].cuda()


            y=y.cuda()
            optimizer.zero_grad()

            pred=model(EEG,EOG)
            loss=loss_fn(pred,y)
            loss.backward()
            optimizer.step()
            if idx % 100 == 0:
                print(epoch, idx, loss.item())

# 2*EEG+EOG+EMG
def train_4ch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # print("train size:",size)
    model.train()
    # print(enumerate(dataloader).shape)
    for batch, (x,y) in enumerate(dataloader):
        # print(list(enumerate(dataloader)))
        # print(X_0.type(torch.FloatTensor).to(device).shape,y.type(torch.LongTensor).shape)
        X_0=x[:,:2]
        X_1=x[:,2:3]
        X_2=x[:,3:4]
        X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
        X1 = X_1.type(torch.FloatTensor).to(device)
        X2 = X_2.type(torch.FloatTensor).to(device)
        # Compute prediction error
        pred = model(X0,X1,X2)
        # print('y.shape',y.shape)
        loss = loss_fn(pred, y)
        # print('pred.shape',pred.shape)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X_0)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct = (pred.argmax(1) == y).type(torch.float).sum().item()/len(X_0)
    return loss.item(), 100*correct
    # return loss, 100*correct

def test_4ch(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    # print("test size:",size)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x,y in dataloader:
            X_0 = x[:, :2]
            X_1 = x[:, 2:3]
            X_2 = x[:, 3:4]
            X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
            X1 = X_1.type(torch.FloatTensor).to(device)
            X2 = X_2.type(torch.FloatTensor).to(device)
            pred = model(X0,X1,X2)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return  test_loss,100*correct