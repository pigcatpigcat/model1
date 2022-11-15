import glob
import os
import re

import numpy as np
from torchvision.io import image
import torch
from torch.utils.data import Dataset,DataLoader
# data=np.load("data/SC4001E0.npz")

# print(data.files)
# print(data["x"].shape)
# print(data["y"][0])
# print(data["fs"])
class dataset(Dataset):
    def __init__(self,data_dir,mode,rate,load=False):
        if(load==True):
            if(mode=="train"):
                self.x=torch.load("traindata_x.pt")
                self.y=torch.load("traindata_y.pt")
            elif(mode=="test"):
                self.x=torch.load("testdata_x.pt")
                self.y=torch.load("testdata_y.pt")
            return
        self.dir=data_dir
        self.x=[]
        self.y=[]

        fs=None
        npz_files = glob.glob(os.path.join(self.dir, "*.npz"))


        strstr=re.sub("\D","",npz_files[-1])
        total=int(strstr[1:3])
        self.len = 0
        self.len2=int(int(strstr[1:3])*rate)
        if(mode=="test"):
            npz_files.reverse()
            for f in npz_files:
                strstr = re.sub("\D", "", f)

                if (int(strstr[1:3]) <= total-self.len2):
                    break

                tmp_x, tmp_y, sampling_rate = self._load_npz_file(f)
                if fs is None:
                    fs = sampling_rate
                elif fs != sampling_rate:
                    raise Exception("Found mismatch in sampling rate.")

                for i in range(len(tmp_y)):
                    self.x.append(tmp_x[i])
                    self.y.append(tmp_y[i])

            self.x = torch.Tensor(self.x).reshape([len(self.y), 1, 3000])
            self.y = torch.Tensor(self.y)
        elif(mode=="train"):
            for f in npz_files:
                strstr = re.sub("\D", "", f)

                if(int(strstr[1:3])>=self.len2):
                    break

                tmp_x,tmp_y,sampling_rate=self._load_npz_file(f)
                if fs is None:
                    fs = sampling_rate
                elif fs != sampling_rate:
                    raise Exception("Found mismatch in sampling rate.")

                for i in range(len(tmp_y)):
                    self.x.append(tmp_x[i])
                    self.y.append(tmp_y[i])


            self.x=torch.Tensor(self.x).reshape([len(self.y),1,3000])
            self.y=torch.Tensor(self.y)
        if(mode=="train"):
            torch.save(self.x, "../traindata_x.pt")
            torch.save(self.y, "../traindata_y.pt")
        elif(mode=="test"):
            torch.save(self.x, "../testdata_x.pt")
            torch.save(self.y, "../testdata_y.pt")


    def _load_npz_file(self, npz_file):
        """Load data and labels from a npz file."""
        with np.load(npz_file) as f:
            data = f["x"]
            labels = f["y"]
            data.shape=(labels.size,3000)
            sampling_rate = f["fs"]
        return data, labels, sampling_rate

    def __len__(self):

        return len(self.y)
    def __getitem__(self, item):

        return self.x[item],self.y[item]










import torch
from torch.utils.data import Dataset
import os
import numpy as np

class LoadDataset_from_numpy(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset,fft=False):
        super(LoadDataset_from_numpy, self).__init__()
        # load files
        X_train = np.load(np_dataset[0])["x"]
        y_train = np.load(np_dataset[0])["y"]
        # x_fft=np.fft.fft(X_train)

        for np_file in np_dataset[1:]:
            X_train = np.vstack((X_train, np.load(np_file)["x"]))
            y_train = np.append(y_train, np.load(np_file)["y"])
            # x_fft=np.vstack((x_fft,np.fft.fft(X_train)))

        self.len = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()
        # x_fft=torch.from_numpy(x_fft)
        # self.x_data=torch.cat((X_train,x_fft),dim=2)
        # Correcting the shape of input to be (Batch_size, #channels, seq_len) where #channels=1
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)
        if(fft==True):
            self.x_data=torch.cat((self.x_data,torch.fft.fft(self.x_data).real),dim=1)


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator_np(training_files, subject_files, batch_size):
    train_dataset = LoadDataset_from_numpy(training_files)
    print("train_data is ready")
    test_dataset = LoadDataset_from_numpy(subject_files)
    print("test_data is ready")
    # to calculate the ratio for the CAL
    all_ys = np.concatenate((train_dataset.y_data, test_dataset.y_data))
    all_ys = all_ys.tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    return train_loader, test_loader, counts

def getDataloader(npz_files,batch_size,num_worker=0,shuffle=True,fft=False):
    dataset1 = LoadDataset_from_numpy(npz_files,fft)
    dataloader1 = torch.utils.data.DataLoader(dataset=dataset1,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              drop_last=False,
                                              num_workers=num_worker,
                                              pin_memory=True)
    return dataloader1