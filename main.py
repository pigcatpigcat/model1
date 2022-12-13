import time
import os

import trainer

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
# os.environ["RANK"]="0"
# os.environ["WORLD_SIZE"]="1"
# os.environ["MASTER_PORT"]="5678"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'
import argparse
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
import torch.distributed as dist
from model import MyModel2u,MyModel6u,MyModel_RectifiedLinearAttention2u,MyModel_SAMNet,DeepSleepNet,MyModel_easy

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"
class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "UNKNOWN"
}



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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--nnodes', default=1, type=int)
    parser.add_argument('--nproc_per_node', default=0, type=int)
    parser.add_argument('--node_rank', default=0, type=int)
    # parser.add_argument('--master_port', default=6005, type=int)
    args = parser.parse_args()
    batch_size=32
    model_name="MyModel2u"
    if(model_name=="MyModel2u"):
        net=MyModel2u.MyModel()
    elif(model_name=="MyModel6u"):
        net=MyModel6u.MyModel()
    elif(model_name=="MyModel_RectifiedLinearAttention2u"):
        net=MyModel_RectifiedLinearAttention2u.MyModel()
    elif(model_name=="MyModel_SAMNet"):
        net=MyModel_SAMNet.MyModel()
    elif(model_name=="DeepSleepNet"):
        net=DeepSleepNet.DeepSleepNet(in_channel=6)
    elif(model_name=="DeepSleepNet_no_fft"):
        net=DeepSleepNet.DeepSleepNet(in_channel=3)
    elif(model_name=="MyModel_easy"):
        net=MyModel_easy.MyModel()

    # torch.distributed.init_process_group(backend="nccl",rank=0,world_size=2)
    net.to(device)
    print(torch.cuda.device_count()+1)
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    if(os.path.exists(model_name+".pt")):
        net.load_state_dict(torch.load(model_name+".pt"))
        print("load model")


    npz_files=os.listdir("data")
    for i in range(len(npz_files)):
        npz_files[i]="data/"+npz_files[i]



    trainloader= dataloader.getDataloader(npz_files[0:122], batch_size=batch_size, num_worker=0, shuffle=True,
                                          pin_memory=True)

    testloader=dataloader.getDataloader(npz_files[122:153],batch_size=batch_size,num_worker=0,shuffle=False,
                                        pin_memory=True)

    print("data is ready")
    t1 = time.perf_counter()
    loss_list,acc_list=train(net,20,trainloader,testloader,model_name)
    t2 = time.perf_counter()
    print(t2 - t1)
    trainer.plot_acc_curve(acc_list,model_name)
    trainer.plot_loss_curve(loss_list, model_name)

# train_and_test_mymodel()
# use_tensorboard()
# salientsleepnet_test()