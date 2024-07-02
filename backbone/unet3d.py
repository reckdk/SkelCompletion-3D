from torch.nn import Module, Sequential
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d, Dropout3d
from torch.nn import AdaptiveAvgPool3d, Linear, Dropout1d
from torch.nn import ReLU, Sigmoid
import torch


class UNet(Module):
    '''
    Adapted from davidiommi:
        https://github.com/davidiommi/Pytorch-Unet3D-single_channel
    '''
    # __                            __
    #  N|__   ________________   __|1
    #     2|__  ____________  __|2
    #        3|__  ______  __|3
    #           4|__ __ __|4
    # The convolution operations on either side are residual subject to 1*1 Convolution for channel homogeneity

    def __init__(self, num_channels=1, feat_channels=[64, 256, 256, 512, 1024], residual='conv'):
        # residual: conv for residual input x through 1*1 conv across every layer for downsampling, None for removal of residuals

        super(UNet, self).__init__()

        # Encoder downsamplers
        self.pool1 = MaxPool3d((2, 2, 2))
        self.pool2 = MaxPool3d((2, 2, 2))
        self.pool3 = MaxPool3d((2, 2, 2))
        self.pool4 = MaxPool3d((2, 2, 2))

        # Encoder convolutions
        self.conv_blk1 = Conv3D_Block(num_channels, feat_channels[0], residual=residual)
        self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual)
        self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual)
        self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3], residual=residual)
        self.conv_blk5 = Conv3D_Block(feat_channels[3], feat_channels[4], residual=residual)

        # Decoder convolutions
        self.dec_conv_blk4 = Conv3D_Block(2 * feat_channels[3], feat_channels[3], residual=residual)
        self.dec_conv_blk3 = Conv3D_Block(2 * feat_channels[2], feat_channels[2], residual=residual)
        self.dec_conv_blk2 = Conv3D_Block(2 * feat_channels[1], feat_channels[1], residual=residual)
        self.dec_conv_blk1 = Conv3D_Block(2 * feat_channels[0], feat_channels[0], residual=residual)

        # Decoder upsamplers
        self.deconv_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3])
        self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2])
        self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0])

        # Final 1*1 Conv Segmentation map
        #self.one_conv = Conv3d(feat_channels[0], num_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.one_conv = Conv3d(feat_channels[0], 1, kernel_size=1, stride=1, padding=0, bias=True)

        # Activation function
        self.sigmoid = Sigmoid()

    def forward(self, x, loss_mask):
        loss_mask.requires_grad = False

        # Encoder part
        x1 = self.conv_blk1(x)

        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)

        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)

        x_low3 = self.pool3(x3)
        x4 = self.conv_blk4(x_low3)

        x_low4 = self.pool4(x4)
        base = self.conv_blk5(x_low4)

        # Decoder part

        d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
        d_high4 = self.dec_conv_blk4(d4)

        d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)
        d_high3 = Dropout3d(p=0.5)(d_high3)

        d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)
        d_high2 = Dropout3d(p=0.5)(d_high2)

        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)

        seg = self.sigmoid(self.one_conv(d_high1))

        return seg

class UNet_ODT(UNet):
    '''
    #UNet with loss_mask for ODT.
    The feat_channels are reduced to fit VRAM.
    There is no activation in the last regression layer.
    '''
    def __init__(self, num_channels=1, feat_channels=[32, 64, 64, 128, 256], residual=True):
        super().__init__(num_channels=num_channels, feat_channels=feat_channels, residual=residual)

    #def forward(self, x, loss_mask):
    def forward(self, x):
        #loss_mask.requires_grad = False

        # Encoder part.
        x1 = self.conv_blk1(x)

        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)

        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)

        x_low3 = self.pool3(x3)
        x4 = self.conv_blk4(x_low3)

        x_low4 = self.pool4(x4)
        base = self.conv_blk5(x_low4)

        # Decoder part.

        d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
        d_high4 = self.dec_conv_blk4(d4)

        d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)
        d_high3 = Dropout3d(p=0.5)(d_high3)

        d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)
        d_high2 = Dropout3d(p=0.5)(d_high2)

        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)

        skel = self.sigmoid(self.one_conv(d_high1))

        return skel

class UNet_ODT_wCLS(UNet):
    '''
    #UNet with classification and segmentation branch.
    The feat_channels are reduced to fit VRAM.
    There is no activation in CLS branch.
    '''
    def __init__(self, num_channels=1, feat_channels=[32, 64, 64, 128, 256], residual=True):
        super().__init__(num_channels=num_channels, feat_channels=feat_channels, residual=residual)
        self.cls_feat_channels = [feat_channels[0], 32, 64, 16]
        #self.cls_feat_channels = [feat_channels[0], 16, 16, 8] # for 2-channel input.

        self.cls_branch = Sequential(
            Conv3d(self.cls_feat_channels[0], self.cls_feat_channels[1], kernel_size=3,
            #Conv3d(self.cls_feat_channels[0]+1, self.cls_feat_channels[1], kernel_size=3,
                   stride=1, padding=0, bias=True),
            ReLU(),
            MaxPool3d((2, 2, 2)),
            Conv3d(self.cls_feat_channels[1], self.cls_feat_channels[2], kernel_size=3,
                   stride=1, padding=0, bias=True),
            AdaptiveAvgPool3d(1))

        # The Sigmoid() will be added within nn.BCEWithLogitsLoss().
        self.cls_head = Sequential(
            Linear(self.cls_feat_channels[2], self.cls_feat_channels[3]),
            Dropout1d(p=0.5),
            Linear(self.cls_feat_channels[3], 1))

        '''
        self.cls_feat_channels = [feat_channels[0], 32, 64, 32, 16]
        self.cls_branch = Sequential(
            Conv3d(self.cls_feat_channels[0], self.cls_feat_channels[1], kernel_size=3,
                   stride=1, padding=0, bias=True),
            ReLU(),
            MaxPool3d((2, 2, 2)),
            Conv3d(self.cls_feat_channels[1], self.cls_feat_channels[2], kernel_size=3,
                   stride=1, padding=0, bias=True),
            ReLU(),
            MaxPool3d((2, 2, 2)),
            Conv3d(self.cls_feat_channels[2], self.cls_feat_channels[2], kernel_size=3,
                   stride=1, padding=0, bias=True),
            AdaptiveAvgPool3d(1))

        # The Sigmoid() will be added within nn.BCEWithLogitsLoss().
        self.cls_head = Sequential(
            Linear(self.cls_feat_channels[2], self.cls_feat_channels[3]),
            Dropout1d(p=0.5),
            Linear(self.cls_feat_channels[3], self.cls_feat_channels[4]),
            Dropout1d(p=0.5),
            Linear(self.cls_feat_channels[4], 1))
        '''

    def forward(self, x):
        # Encoder part.
        x1 = self.conv_blk1(x)

        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)

        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)

        x_low3 = self.pool3(x3)
        x4 = self.conv_blk4(x_low3)

        x_low4 = self.pool4(x4)
        base = self.conv_blk5(x_low4)

        # Decoder part.
        d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
        d_high4 = self.dec_conv_blk4(d4)

        d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)
        d_high3 = Dropout3d(p=0.5)(d_high3)

        d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)
        d_high2 = Dropout3d(p=0.5)(d_high2)

        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)

        # Skel branch.
        reg = self.one_conv(d_high1)
        seg = self.sigmoid(reg)

        # Classification branch.
        c_feat = self.cls_branch(d_high1)
        #c_feat = self.cls_branch(torch.cat((x, d_high1), dim=1))
        y = self.cls_head(c_feat.view(x.shape[0], self.cls_feat_channels[2]))

        return seg, y

class UNet_ODT_wCLS_wEDT(UNet):
    '''
    #UNet with classification, segmentation, and EDT branches.
    The feat_channels are further reduced to fit VRAM.
    There is no activation in CLS branch.
    '''
    def __init__(self, num_channels=1, feat_channels=[32, 32, 64, 128, 256], residual=True):
        super().__init__(num_channels=num_channels, feat_channels=feat_channels, residual=residual)
        self.cls_feat_channels = [feat_channels[0], 32, 64, 16]
        #self.cls_feat_channels = [feat_channels[0], 16, 16, 8] # for 2-channel input.

        self.cls_branch = Sequential(
            Conv3d(self.cls_feat_channels[0], self.cls_feat_channels[1], kernel_size=3,
            #Conv3d(self.cls_feat_channels[0]+1, self.cls_feat_channels[1], kernel_size=3,
                   stride=1, padding=0, bias=True),
            ReLU(),
            MaxPool3d((2, 2, 2)),
            Conv3d(self.cls_feat_channels[1], self.cls_feat_channels[2], kernel_size=3,
                   stride=1, padding=0, bias=True),
            AdaptiveAvgPool3d(1))

        # The Sigmoid() will be added within nn.BCEWithLogitsLoss().
        self.cls_head = Sequential(
            Linear(self.cls_feat_channels[2], self.cls_feat_channels[3]),
            Dropout1d(p=0.5),
            Linear(self.cls_feat_channels[3], 1))

        self.one_conv_edt = Conv3d(feat_channels[0], 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        # Encoder part.
        x1 = self.conv_blk1(x)

        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)

        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)

        x_low3 = self.pool3(x3)
        x4 = self.conv_blk4(x_low3)

        x_low4 = self.pool4(x4)
        base = self.conv_blk5(x_low4)

        # Decoder part.
        d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
        d_high4 = self.dec_conv_blk4(d4)

        d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)
        d_high3 = Dropout3d(p=0.5)(d_high3)

        d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)
        d_high2 = Dropout3d(p=0.5)(d_high2)

        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)

        # Skel branch.
        skel = self.sigmoid(self.one_conv(d_high1))

        # Classification branch.
        c_feat = self.cls_branch(d_high1)
        y = self.cls_head(c_feat.view(x.shape[0], self.cls_feat_channels[2]))

        # Seg branch.
        #seg = Sigmoid()(self.one_conv_edt(d_high1))

        # EDT branch.
        reg = self.one_conv_edt(d_high1)

        return skel, y, reg


class UNet_ODT_wEDT(UNet):
    '''
    #UNet with EDT branches.
    The feat_channels are reduced to fit VRAM.
    still 64
    '''
    def __init__(self, num_channels=1, feat_channels=[32, 64, 64, 128, 256], residual=True):
        super().__init__(num_channels=num_channels, feat_channels=feat_channels, residual=residual)
        self.cls_feat_channels = [feat_channels[0], 32, 64, 16]
        #self.cls_feat_channels = [feat_channels[0], 16, 16, 8] # for 2-channel input.

        self.one_conv_edt = Conv3d(feat_channels[0], 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        # Encoder part.
        x1 = self.conv_blk1(x)

        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)

        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)

        x_low3 = self.pool3(x3)
        x4 = self.conv_blk4(x_low3)

        x_low4 = self.pool4(x4)
        base = self.conv_blk5(x_low4)

        # Decoder part.
        d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
        d_high4 = self.dec_conv_blk4(d4)

        d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)
        d_high3 = Dropout3d(p=0.5)(d_high3)

        d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)
        d_high2 = Dropout3d(p=0.5)(d_high2)

        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)

        # Skel branch.
        skel = self.sigmoid(self.one_conv(d_high1))
        # Seg branch.
        #seg = Sigmoid()(self.one_conv_edt(d_high1))

        # EDT branch.
        reg = self.one_conv_edt(d_high1)

        return skel, reg

class UNet3D_4L(Module):
    '''
    Adapted from davidiommi:
        https://github.com/davidiommi/Pytorch-Unet3D-single_channel
    '''
    # __                            __
    #  1|__   ________________   __|1
    #     2|__  ____________  __|2
    #        3|__  ______  __|3
    # The convolution operations on either side are residual subject to 1*1 Convolution for channel homogeneity

    def __init__(self, num_channels=1, feat_channels=[64, 256, 256, 512], residual='conv'):
        # residual: conv for residual input x through 1*1 conv across every layer for downsampling, None for removal of residuals

        super(UNet3D_4L, self).__init__()

        # Encoder downsamplers
        self.pool1 = MaxPool3d((2, 2, 2))
        self.pool2 = MaxPool3d((2, 2, 2))
        self.pool3 = MaxPool3d((2, 2, 2))

        # Encoder convolutions
        self.conv_blk1 = Conv3D_Block(num_channels, feat_channels[0], residual=residual)
        self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual)
        self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual)
        self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3], residual=residual)

        # Decoder convolutions
        self.dec_conv_blk3 = Conv3D_Block(2 * feat_channels[2], feat_channels[2], residual=residual)
        self.dec_conv_blk2 = Conv3D_Block(2 * feat_channels[1], feat_channels[1], residual=residual)
        self.dec_conv_blk1 = Conv3D_Block(2 * feat_channels[0], feat_channels[0], residual=residual)

        # Decoder upsamplers
        self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2])
        self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0])

        # Final 1*1 Conv Segmentation map
        self.one_conv = Conv3d(feat_channels[0], num_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # Encoder part
        x1 = self.conv_blk1(x)

        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)

        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)

        x_low3 = self.pool3(x3)
        x4 = self.conv_blk4(x_low3)

        # Decoder part
        d3 = torch.cat([self.deconv_blk3(x4), x3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)
        d_high3 = Dropout3d(p=0.5)(d_high3)

        d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)
        d_high2 = Dropout3d(p=0.5)(d_high2)

        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)

        seg = self.one_conv(d_high1)

        return seg


class Conv3D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):

        super(Conv3D_Block, self).__init__()

        self.conv1 = Sequential(
            Conv3d(inp_feat, out_feat, kernel_size=kernel,
                   stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_feat),
            ReLU())

        self.conv2 = Sequential(
            Conv3d(out_feat, out_feat, kernel_size=kernel,
                   stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_feat),
            ReLU())

        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):

        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)


class Deconv3D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=2, padding=1):
        super(Deconv3D_Block, self).__init__()

        self.deconv = Sequential(
            ConvTranspose3d(inp_feat, out_feat, kernel_size=(kernel, kernel, kernel),
                            stride=(stride, stride, stride), padding=(padding, padding, padding), output_padding=1, bias=True),
            ReLU())

    def forward(self, x):
        return self.deconv(x)


class ChannelPool3d(AvgPool1d):

    def __init__(self, kernel_size, stride, padding):
        super(ChannelPool3d, self).__init__(kernel_size, stride, padding)
        self.pool_1d = AvgPool1d(self.kernel_size, self.stride, self.padding, self.ceil_mode)

    def forward(self, inp):
        n, c, d, w, h = inp.size()
        inp = inp.view(n, c, d * w * h).permute(0, 2, 1)
        pooled = self.pool_1d(inp)
        c = int(c / self.kernel_size[0])
        return inp.view(n, c, d, w, h)


if __name__ == '__main__':
    import time
    import torch
    from torchinfo import summary

    torch.cuda.set_device(0)
    net =UNet(residual='conv').cuda().eval()

    #data = torch.Tensor(torch.randn(1, 1, 64, 700, 64)).cuda()
    #out = net(data)

    summary(net, input_size=(1, 1, 64, 700, 64))
    print("out size: {}".format(out.size()))