import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class ContBatchNorm2d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 4:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        #super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x):
        return self.double_conv(x)


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, padding=1)
        # self.bn1 = ContBatchNorm2d(out_chan)


        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif act == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        # else:
        #     raise

    def forward(self, x):
        # out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(F.group_norm(self.conv1(x),8))
        return out


def _make_nConv(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        # layer1 = LUConv(in_channel, 32 * (2 ** (depth+1)),act)
        layer1 = LUConv(in_channel, 8 * (2 ** (depth+1)),3, act)
        # layer2 = LUConv(32 * (2 ** (depth+1)), 32 * (2 ** (depth+1)),act)
    else:
        # layer1 = LUConv(in_channel, 32*(2**depth),act)
        layer1 = LUConv(in_channel, 8*(2**depth),3, act)
        # layer2 = LUConv(32*(2**depth), 32*(2**depth)*2,act)

    # return nn.Sequential(layer1,layer2)
    return nn.Sequential(layer1)



# class InputTransition(nn.Module):
#     def __init__(self, outChans, elu):
#         super(InputTransition, self).__init__()
#         self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
#         self.bn1 = ContBatchNorm3d(16)
#         self.relu1 = ELUCons(elu, 16)
#
#     def forward(self, x):
#         # do we want a PRELU here as well?
#         out = self.bn1(self.conv1(x))
#         # split input in to 16 channels
#         x16 = torch.cat((x, x, x, x, x, x, x, x,
#                          x, x, x, x, x, x, x, x), 1)
#         out = self.relu1(torch.add(out, x16))
#         return out


class DownTransition(nn.Module):
    def __init__(self, in_channel,out_chan, act):
        super(DownTransition, self).__init__()
        self.ops1 = LUConv(in_channel, out_chan, 3, act)
        self.ops2 = LUConv(out_chan, out_chan, 3, act)
        # self.maxpool = nn.MaxPool2d(2)
        # self.pool = nn.Conv2d(32 * (2 ** (depth+1)), 32 * (2 ** (depth+1)), kernel_size=3, stride=2, padding=1)
        self.pool = nn.Conv2d(out_chan,out_chan, kernel_size=3, stride=2, padding=1)
        self.activation = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):

        x = self.ops1(x)
        out_before_pool = self.ops2(x)
        out = self.pool(out_before_pool)
        out = self.activation(F.group_norm(out, 8))

        return out, out_before_pool


class DownTrans(nn.Module):
    def __init__(self, in_channel,depth, act):
        super(DownTrans, self).__init__()
        self.ops = _make_nConv(in_channel*2, depth,act,double_chnnel = True)
        # self.maxpool = nn.MaxPool2d(2)
        # self.pool = nn.Conv2d(32 * (2 ** (depth+1)), 32 * (2 ** (depth+1)), kernel_size=3, stride=2, padding=1)
        self.pool = nn.Conv2d(8 * (2 ** (depth+1)), 8 * (2 ** (depth+1)), kernel_size=3, stride=2, padding=1)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 4:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.pool(out_before_pool)
        return out, out_before_pool


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, act):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose2d(inChans, outChans, kernel_size=2, stride=2)
        self.activation = nn.LeakyReLU(negative_slope=0.1)

        self.ops1 = LUConv(outChans*2, outChans,3,act)
        self.ops2 = LUConv(outChans, outChans,3,act)


    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        out_up_conv = self.activation(F.group_norm(out_up_conv, 8))

        concat = torch.cat((out_up_conv, skip_x),1)
        out = self.ops1(concat)
        out = self.ops2(out)
        return out


class UpTrans(nn.Module):
    def __init__(self, inChans, outChans,  act):
        super(UpTrans, self).__init__()
        self.up_conv = nn.ConvTranspose2d(inChans, outChans, kernel_size=2, stride=2)
        self.activation = nn.LeakyReLU(negative_slope=0.1)

        self.ops1 = LUConv(outChans * 3, outChans, 3, act)
        self.ops2 = LUConv(outChans, outChans, 3, act)

    def forward(self, x, skip_x1):
        out_up_conv = self.up_conv(x)
        out_up_conv = self.activation(F.group_norm(out_up_conv, 8))

        concat = torch.cat((out_up_conv, skip_x1), 1)
        out = self.ops1(concat)
        out = self.ops2(out)
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):

        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv2d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.final_conv(x)
        return out

class UNet2D(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, n_class=1, act='lrelu'):
        super(UNet2D, self).__init__()

        self.doubleconv = DoubleConv(1, 64)
        self.block1 = DownTransition(64, 128, act)
        self.block2 = DownTransition(128, 256, act)
        self.block3 = DownTransition(256, 512, act)
        self.block4 = DownTransition(512, 1024, act)

        # self.convmid = nn.Sequential(LUConv(64, 64,3,act),
        #                              LUConv(64, 64,3,act))

        self.block5 = UpTransition(1024, 1024, act)
        self.block6 = UpTransition(1024, 512, act)
        self.block7 = UpTransition(512, 256, act)
        self.block8 = UpTransition(256, 128, act)

        self.block9 = OutputTransition(128, n_class)


    def forward(self, x):
        self.douconv = self.doubleconv(x)

        self.b1, self.skip_b1 = self.block1(self.douconv)

        self.b2,self.skip_b2 = self.block2(self.b1)

        self.b3,self.skip_b3 = self.block3(self.b2)

        self.b4,self.skip_b4 = self.block4(self.b3)

        #pdb.set_trace()
        # self.outmid= self.convmid(self.b4)
        # print(self.outmid.shape)
        self.b5 = self.block5(self.b4, self.skip_b4)

        self.b6 = self.block6(self.b5, self.skip_b3)

        self.b7 = self.block7(self.b6, self.skip_b2)

        self.b8 = self.block8(self.b7, self.skip_b1)

        self.out = self.block9(self.b8)

        #pdb.set_trace()
        return self.out
