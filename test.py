#!/usr/bin/env python
# coding: utf-8

# # Network Testing

# ## Includes

# In[ ]:


# mass includes
from os import listdir
from os.path import splitext
from gc import collect
from rawpy import imread
from numpy import float32 as np32
from torch import float as t32
from torch import tensor, no_grad, from_numpy, stack, clamp, load, cat
from torch.nn import Module, Sequential
from torch.nn import MaxPool2d, Conv2d, LeakyReLU, ConvTranspose2d, PixelShuffle
from torchvision.utils import save_image


# ## U-Net

# In[ ]:


class downBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(downBlock, self).__init__()

        self.features = Sequential(
            MaxPool2d(2, 2), Conv2d(in_channels, out_channels, 3, padding=1),
            LeakyReLU(0.2, inplace=True),
            Conv2d(out_channels, out_channels, 3, padding=1),
            LeakyReLU(0.2, inplace=True))

    def forward(self, x):

        return self.features(x)


class upBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(upBlock, self).__init__()

        inter_channels = out_channels * 2
        self.features = Sequential(
            Conv2d(in_channels, inter_channels, 3, padding=1),
            LeakyReLU(0.2, inplace=True),
            Conv2d(inter_channels, inter_channels, 3, padding=1),
            LeakyReLU(0.2, inplace=True),
            ConvTranspose2d(inter_channels, out_channels, 2, stride=2))

    def forward(self, x):

        return self.features(x)


class UNet(Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.model_name = 'UNet'

        # head block
        self.head = Sequential(
            Conv2d(4, 32, 3, padding=1), LeakyReLU(0.2, inplace=True),
            Conv2d(32, 32, 3, padding=1), LeakyReLU(0.2, inplace=True))

        # block 1-4
        self.d1 = downBlock(32, 64)
        self.d2 = downBlock(64, 128)
        self.d3 = downBlock(128, 256)

        # bottom block
        self.bottom = Sequential(
            MaxPool2d(2, 2), Conv2d(256, 512, 3, padding=1),
            LeakyReLU(0.2, inplace=True), Conv2d(512, 512, 3, padding=1),
            LeakyReLU(0.2, inplace=True), ConvTranspose2d(
                512, 256, 2, stride=2))

        # blcok 5-8
        self.u1 = upBlock(512, 128)
        self.u2 = upBlock(256, 64)
        self.u3 = upBlock(128, 32)

        # final block
        self.final = Sequential(
            Conv2d(64, 32, 3, padding=1), LeakyReLU(0.2, inplace=True),
            Conv2d(32, 32, 3, padding=1), LeakyReLU(0.2, inplace=True),
            Conv2d(32, 12, 1), PixelShuffle(2))

    def forward(self, x):
        out_head = self.head(x)
        out_d1 = self.d1(out_head)
        out_d2 = self.d2(out_d1)
        out_d3 = self.d3(out_d2)
        out_bottom = self.bottom(out_d3)
        out_u1 = self.u1(cat([out_d3, out_bottom], dim=1))
        del out_bottom, out_d3
        collect()
        out_u2 = self.u2(cat([out_d2, out_u1], dim=1))
        del out_u1, out_d2
        collect()
        out_u3 = self.u3(cat([out_d1, out_u2], dim=1))
        del out_u2, out_d1
        collect()
        out_final = self.final(cat([out_head, out_u3], dim=1))
        del out_u3, out_head
        collect()

        return out_final


# ## Data pre-processing

# In[ ]:


def readRaw(file_path, ratio):
    raw_data = imread(file_path)
    bk_level = tensor(raw_data.black_level_per_channel, dtype=t32)
    raw_data = raw_data.raw_image_visible.astype(np32)
    raw_data = from_numpy(raw_data)
    hei, wid = raw_data.size()
    raw_4d = stack((raw_data[0:hei:2, 0:wid:2], raw_data[0:hei:2, 1:wid:2],
                    raw_data[1:hei:2, 1:wid:2], raw_data[1:hei:2, 0:wid:2]),
                   dim=2)

    # ensure raw_4d to be divisible by 16
    while raw_4d.size(0) % 16 != 0:
        raw_4d = raw_4d[:-3, :, :]
    while raw_4d.size(1) % 16 != 0:
        raw_4d = raw_4d[:, :-3, :]

    raw_4d = (raw_4d - bk_level) / (16383 - bk_level)
    raw_4d = clamp(raw_4d * ratio, 0.0, 1.0)

    return raw_4d.permute(2, 0, 1)


# ## Testing

# In[ ]:


def main():
    # load pre-trained model
    model = UNet().to('cpu')
    model.load_state_dict(load('./UNet_0212-110222.pth', map_location='cpu'))
    model.eval()

    # user inputs
    ratio = float(input('Specify an amplify ratio (e.g. 300): '))

    # load data
    file_list = listdir('./')
    for file in file_list:
        name, ext = splitext(file)
        if ext == '.dng' or ext == '.DNG':
            # run network
            with no_grad():
                raw_img = readRaw('./' + file, ratio)
                raw_img = raw_img.unsqueeze(0)
                out_img = model(raw_img)

                # save image
                save_image(out_img.squeeze(0), './%s.jpg' % name)


if __name__ == '__main__':
    main()

