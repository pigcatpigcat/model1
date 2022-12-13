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
    def __init__(self, version='big', in_channel=1, layer=[16, 32, 32, 32], activation='relu', sample_rate=100,
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
    def __init__(self, in_channel=1, layer=[16, 32, 32, 32], activation='relu', sample_rate=100, dropout_p=0.5):
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
    def __init__(self, in_channel=6, out_channel=300, layer=[16, 32, 32, 32], activation='relu', sample_rate=100,
                 dropout_p=0.5):
        super(DeepSleepNet_CNN, self).__init__()
        self.featureExtractor = DeepSleepNet_featureExtractor(in_channel=in_channel, layer=layer, activation=activation,
                                                              sample_rate=sample_rate, dropout_p=dropout_p)
        if dropout_p > 0.:
            self.dropout = nn.Dropout(p=dropout_p)
        else:
            self.dropout = nn.Identity()
        self.fc = nn.Linear(2272, out_channel)  # big and small conv concat

    def forward(self, x):
        x = self.featureExtractor(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class AFR(nn.Module):
    def __init__(self, block, planes, blocks, stride=1):
        super(AFR, self).__init__()

        self.inplanes=1
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        self.AFR=nn.Sequential(*layers)
    def forward(self,x):
        return self.AFR(x)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d_model=96
        self.nhead=4
        self.num_layers=1
        self.afr_reduced_cnn_size=30

        self.feature=DeepSleepNet_CNN()
        self.linear0=nn.Sequential(nn.Linear(300, 96), nn.ReLU(True))
        # self.AFR=AFR(block=SEBasicBlock, planes=30, blocks=1)
        encoder_layer = nn.TransformerEncoderLayer(self.d_model, nhead=self.nhead, batch_first=True, activation="relu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        # self.autoformer=Autoformer()
        self.linear1 = nn.Sequential(nn.Linear(96, 32), nn.ReLU(True))
        self.linear2 = nn.Sequential(nn.Linear(32, 5), nn.Softmax(dim=1))

    def forward(self,x):


        x=torch.cat((x,torch.fft.fft(x).real),dim=1)

        #without autoformer
        x=self.feature(x)
        # x=self.AFR(x)
        # encoded_features = self.transformer_encoder(x_afr.view(x_afr.shape[0], -1, 96))
        x=self.linear0(x)
        encoded_features = self.transformer_encoder(x)
        # print(encoded_features.shape)

        encoded_features = x * encoded_features
        encoded_features = encoded_features.contiguous().view(encoded_features.shape[0], -1)
        # print(encoded_features.shape)
        x_final = self.linear1(encoded_features)
        x_final = self.linear2(x_final)
        return x_final

# net=MyModel()
# net.eval()
# x=torch.rand((1,3,3000))
# y=net(x)
# print(y)