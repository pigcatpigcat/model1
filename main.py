import time

import numpy as np
from captum.attr import Saliency, IntegratedGradients,DeepLift,NoiseTunnel,Deconvolution,FeatureAblation,InputXGradient
from model.DeepSleepNet import DeepSleepNet
from model.MMAsleepnet import MMASleepNet
from dataloader import dataloader
import torch
from trainer import train
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

device = "cuda" if torch.cuda.is_available() else "cpu"
class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "UNKNOWN"
}

def module_test(module,test_dataloader):

    total_correct = 0
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.cuda()
            y=y.cuda()
            out = module(x)
            pred = out.argmax(dim=1)
            correct = pred.eq(y).sum().float().item()
            total_correct += correct
        total = len(test_dataloader.dataset)
        acc = total_correct / total
    # 准确率
    print(acc)

def captum():
    train_data= dataloader.dataset("data", mode="train", rate=0.8, load=True)
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)

    net=DeepSleepNet()
    net.load_state_dict(torch.load("deepsleepnet.pt"))


    fig = plt.figure()
    plt.title("target:{}".format(train_data[0][1]))
    x1=np.arange(0,30,0.01)

    # plt.show()
    idx,(x,y)=next(enumerate(train_dataloader))
    m1=NoiseTunnel(Saliency(net))
    m2=NoiseTunnel(IntegratedGradients(net))
    m3=NoiseTunnel(DeepLift(net))
    m4=NoiseTunnel(InputXGradient(net))
    m5=NoiseTunnel(Deconvolution(net))
    m6=NoiseTunnel(FeatureAblation(net))
    # xx=x.reshape([1,1,3000])


    attribution1=m1.attribute(x,target=int(y))
    attribution2=m2.attribute(x,target=int(y))
    attribution3=m3.attribute(x,target=int(y))
    attribution4=m4.attribute(x,target=int(y))
    # attribution4=torch.zeros((1,1,3000))
    attribution5=m5.attribute(x,target=int(y))
    attribution6=m6.attribute(x, target=int(y))
    norm1 = mpl.colors.Normalize(vmin=0, vmax=1)
    label=int(y)
    # plt.title(label)
    plt.subplot(6,1,1)
    plt.title("saliency {}".format(class_dict[label]))
    plt.plot(x1,train_data[0][0].numpy().ravel())
    im1=plt.imshow(attribution1[0],cmap="Reds",aspect="auto",extent=(0,30,min(train_data[0][0][0]),max(train_data[0][0][0])))
    cbar1 = fig.colorbar(im1)

    plt.subplot(6,1,2)
    plt.title("IntegratedGradients {}".format(class_dict[label]))
    plt.plot(x1,train_data[0][0].numpy().ravel())
    im2=plt.imshow(attribution2[0],cmap="Reds",aspect="auto",extent=(0,30,min(train_data[0][0][0]),max(train_data[0][0][0])))
    cbar2 = fig.colorbar(im2)

    plt.subplot(6,1,3)
    plt.title("DeepLift {}".format(class_dict[label]))
    plt.plot(x1,train_data[0][0].numpy().ravel())
    im3=plt.imshow(attribution3[0].detach().numpy(),cmap="Reds",aspect="auto",extent=(0,30,min(train_data[0][0][0]),max(train_data[0][0][0])))
    cbar3 = fig.colorbar(im3)

    plt.subplot(6,1,4)
    plt.title("InputXGradient {}".format(class_dict[label]))
    plt.plot(x1,train_data[0][0].numpy().ravel())
    im4=plt.imshow(attribution4[0].detach().numpy(),cmap="Reds",aspect="auto",extent=(0,30,min(train_data[0][0][0]),max(train_data[0][0][0])))
    cbar4 = fig.colorbar(im4)

    plt.subplot(6,1,5)
    plt.title("Decon {}".format(class_dict[label]))
    plt.plot(x1,train_data[0][0].numpy().ravel())
    im5=plt.imshow(attribution5[0].detach().numpy(),cmap="Reds",aspect="auto",extent=(0,30,min(train_data[0][0][0]),max(train_data[0][0][0])))
    cbar5 = fig.colorbar(im5)

    plt.subplot(6,1,6)
    plt.title("FeatureAblation {}".format(class_dict[label]))
    plt.plot(x1,train_data[0][0].numpy().ravel())
    im6=plt.imshow(attribution6[0].detach().numpy(),cmap="Reds",aspect="auto",extent=(0,30,min(train_data[0][0][0]),max(train_data[0][0][0])))
    cbar5 = fig.colorbar(im6)

    plt.show()


# from torch.utils.tensorboard import SummaryWriter
# from torchvision import datasets, transforms
# import torchvision
#
# def use_tensorboard():
#     writer = SummaryWriter()
#
#     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
#     trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
#     model = torchvision.models.resnet50(False)
#     # Have ResNet model take in grayscale rather than RGB
#     model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#     images, labels = next(iter(trainloader))
#
#     grid = torchvision.utils.make_grid(images)
#     writer.add_image('images', grid, 0)
#     writer.add_graph(model, images)
#     writer.close()


from model import MyModel2u,MyModel6u,MyModel_RectifiedLinearAttention2u
if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    batch_size=32
    model_name="MyModel_RectifiedLinearAttention2u"
    if(model_name=="MyModel2u"):
        net=MyModel2u.MyModel()
    elif(model_name=="MyModel6u"):
        net=MyModel6u.MyModel()
    elif(model_name=="MyModel_RectifiedLinearAttention2u"):
        net=MyModel_RectifiedLinearAttention2u.MyModel()

    if(os.path.exists(model_name+".pt")):
        net.load_state_dict(torch.load(model_name+".pt"))

    net.cuda()

    npz_files=os.listdir("data")
    for i in range(len(npz_files)):
        npz_files[i]="data/"+npz_files[i]

    # t1 = time.perf_counter()
    # dataloader1= dataloader.getDataloader(npz_files[0:1], batch_size=batch_size, num_worker=2, shuffle=True)
    # print("data is ready")
    # train(net,20,dataloader1,model_name)
    # # torch.save(net.state_dict(),model_name)
    # t2 = time.perf_counter()
    # print(t2 - t1)


    net.eval()
    dataset1= dataloader.LoadDataset_from_numpy(npz_files[122:153])
    dataloader1 = torch.utils.data.DataLoader(dataset=dataset1,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)
    module_test(net,dataloader1)


# train_and_test_mymodel()
# use_tensorboard()
# salientsleepnet_test()