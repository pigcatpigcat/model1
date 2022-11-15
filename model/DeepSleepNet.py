import torch
import torch.nn as nn


class conv_bn_activate_layer(nn.Module):
    def __init__(self, in_channel, out_channel, filter_size, stride, padding, activation='relu', bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=filter_size, stride=stride, padding=padding,
                              bias=bias)
        self.bn = nn.BatchNorm1d(out_channel)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'prelu':
            self.activation == nn.PReLU()
        elif activation == 'leakyrelu':
            self.activation == nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class feature_extractor(nn.Module):
    def __init__(self, version='big', in_channel=1, layer=[64, 128, 128, 128], activation='relu', sample_rate=100,
                 dropout_p=0.5):
        super().__init__()
        if version == 'big':
            first_conv_filter_size = int(sample_rate * 4)
            first_conv_stride = int(sample_rate // 2)
            conv_filter_size = 6
            mp1_size = 4
            mp2_size = 2
        elif version == 'small':
            first_conv_filter_size = int(sample_rate // 2)
            first_conv_stride = int(sample_rate // 16)
            conv_filter_size = 8
            mp1_size = 8
            mp2_size = 4

        first_conv_padding = int((first_conv_filter_size - 1) // 2)
        conv_padding_size = int((conv_filter_size - 1) // 2)
        '''
        In this paper, authors doesn't mention about padding size in convolution layer
        And, they use even number in convoltuion layer filter, so, symmety padding size can't use here.
        if you want to use padding, you can changed 'bias' parameter.
        '''

        self.conv1 = conv_bn_activate_layer(in_channel=in_channel, out_channel=layer[0],
                                            filter_size=first_conv_filter_size, stride=first_conv_stride,
                                            padding=first_conv_padding, activation=activation, bias=False)
        self.maxpool1 = nn.MaxPool1d(kernel_size=mp1_size, stride=mp1_size, padding=0)
        if dropout_p > 0.:
            self.dropout = nn.Dropout(p=dropout_p)
        else:
            self.dropout = nn.Identitiy()

        self.conv2_1 = conv_bn_activate_layer(in_channel=layer[0], out_channel=layer[1], filter_size=conv_filter_size,
                                              stride=1, padding=conv_padding_size, activation=activation, bias=False)
        self.conv2_2 = conv_bn_activate_layer(in_channel=layer[1], out_channel=layer[2], filter_size=conv_filter_size,
                                              stride=1, padding=conv_padding_size, activation=activation, bias=False)
        self.conv2_3 = conv_bn_activate_layer(in_channel=layer[2], out_channel=layer[3], filter_size=conv_filter_size,
                                              stride=1, padding=conv_padding_size, activation=activation, bias=False)

        self.mxpool2 = nn.MaxPool1d(kernel_size=mp2_size, stride=mp2_size, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.dropout(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        return x


class DeepSleepNet_featureExtractor(nn.Module):
    def __init__(self, in_channel=1, layer=[64, 128, 128, 128], activation='relu', sample_rate=100, dropout_p=0.5):
        super().__init__()

        self.big_extractor = feature_extractor(version='big', in_channel=in_channel, layer=layer, activation=activation,
                                               sample_rate=sample_rate, dropout_p=dropout_p)
        self.small_extractor = feature_extractor(version='small', in_channel=in_channel, layer=layer,
                                                 activation=activation, sample_rate=sample_rate, dropout_p=dropout_p)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        big_feature = self.big_extractor(x)
        small_feature = self.small_extractor(x)

        big_feature = torch.flatten(big_feature, 1)
        small_feature = torch.flatten(small_feature, 1)
        # print('big : ',big_feature.shape)
        # print('small : ',small_feature.shape)
        output = torch.cat((big_feature, small_feature), dim=1)
        return output


class DeepSleepNet_CNN(nn.Module):  # input channel = 8channel / output = 5
    def __init__(self, in_channel=1, out_channel=5, layer=[64, 128, 128, 128], activation='relu', sample_rate=100,
                 dropout_p=0.5):
        super(DeepSleepNet_CNN, self).__init__()
        self.featureExtractor = DeepSleepNet_featureExtractor(in_channel=in_channel, layer=layer, activation=activation,
                                                              sample_rate=sample_rate, dropout_p=dropout_p)
        if dropout_p > 0.:
            self.dropout = nn.Dropout(p=dropout_p)
        else:
            self.dropout = nn.Identity()
        self.fc = nn.Linear(9088, out_channel)  # big and small conv concat

    def forward(self, x):
        x = self.featureExtractor(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x

class DeepSleepNet(nn.Module):
    def __init__(self,in_channel=1, out_channel=512, layer=[64, 128, 128, 128], activation='relu', sample_rate=100,
                 dropout_p=0.5):
        super(DeepSleepNet, self).__init__()
        self.CNN=DeepSleepNet_CNN(in_channel, out_channel, layer, activation, sample_rate,
                 dropout_p)
        self.dropout1=nn.Dropout(p=dropout_p)
        self.lstm1=nn.LSTM(out_channel,512,bidirectional=True)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.lstm2=nn.LSTM(1024,512,bidirectional=True)
        self.dropout3=nn.Dropout(p=dropout_p)
        self.dropout4=nn.Dropout(p=dropout_p)
        self.fc=nn.Linear(out_channel,1024)
        self.fc2=nn.Linear(1024,5)
        self.softmax=nn.Softmax(dim=1)
    def forward(self,x):
        self.x=self.CNN(x)
        self.x=self.dropout1(self.x)
        self.y,(self.hn,self.cn)=self.lstm1(self.x)
        self.x=self.fc(self.x)
        self.y=self.dropout2(self.y)
        self.y,(self.hn,self.cn)=self.lstm2(self.y,(self.hn,self.cn))
        self.y=self.dropout3(self.y)
        self.y=torch.add(self.x,self.y)
        self.y=self.dropout4(self.y)
        self.y=self.fc2(self.y)

        return self.softmax(self.y)
