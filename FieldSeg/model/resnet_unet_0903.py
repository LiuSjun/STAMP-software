import torch.nn as nn, torch
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
# import segmentation_models_pytorch as smp

class ResnetEncoder(ResNet):
    # in_channels = 8  # R, G, B, Nir

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]

    def __init__(self, in_channels, block, layers, url, **kwargs):
        # 调用ResNet超类，使用官方提供的resnet构建resnet网络
        super(ResnetEncoder, self).__init__(block, layers, **kwargs)
        self.in_channels = in_channels
        # 使用Image Net初始化的权重来给新的ResNet赋权
        zoo_load = model_zoo.load_url(url)
        self.load_state_dict(zoo_load)

        # 因为在这里的resnet充当feature extracter的作用，删除resnet末尾的fc层和avgpool层
        del self.fc
        del self.avgpool


        # 将第一层的conv2d由3通道调整为4通道
        module = None
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                break
        module.in_channels = self.in_channels

        # 重新建立conv2d的通道
        new_weight = torch.Tensor(
            module.out_channels,
            self.in_channels,
            *module.kernel_size
        )

        # 将原始的3通道参数，由%的方式，赋予4通道，即1->1, 2->2, 3->3, 4->1,...
        weight = module.weight.detach()
        for i in range(self.in_channels):
            new_weight[:, i] = weight[:, i % 3]  # 3 is default in_channels
        new_weight = new_weight * (3 / self.in_channels)
        module.weight = nn.parameter.Parameter(new_weight)

    def get_stages(self):
        return [
            nn.Identity(),  # 该位置不改变网络结构，只是为了forward时，建立原始图像的link
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()
        features = []
        for i in range(len(stages)):
            x = stages[i](x)
            features.append(x)
        return features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
    ):

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        use_batchnorm=True,
    ):
        super().__init__()

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        # combine decoder keyword arguments
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        x = features[0]
        skips = features[1:]

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return x


def get_encoder(in_channels):
    params = {
        # "out_channels": (3, 64, 64, 128, 256, 512),
        "in_channels": in_channels,
        "block": BasicBlock,
        "layers": [3, 4, 6, 3],
        "url": "https://download.pytorch.org/models/resnet34-333f7ec4.pth"
    }
    encoder = ResnetEncoder(**params)
    return encoder


def get_decoder():
    encoder_channels = (4, 64, 64, 128, 256, 512)
    decoder_channels = (256, 128, 64, 32, 16)
    decoder = ResnetDecoder(
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
    )
    return decoder


def get_segmentation_head():
    kernel_size = 3
    in_channels = 16
    out_channels = 1
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), padding=kernel_size // 2)
    return nn.Sequential(conv2d, nn.Sigmoid())


class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        # x3 = x[:, 3]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=1)

        # x3_v = F.conv2d(x3.unsqueeze(1), self.weight_v, padding=1)
        # x3_h = F.conv2d(x3.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)
        # x3 = torch.sqrt(torch.pow(x3_v, 2) + torch.pow(x3_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1) #, x3
        return x


class UNet(nn.Module):
    @staticmethod
    def initialize_decoder(module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def initialize_head(module):
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def __init__(self):
        super().__init__()
        self.encoder = get_encoder(4)
        self.decoder = get_decoder()
        self.segmentation_head = get_segmentation_head()

        # 初始化decoder和segmentation head中的Parameters
        self.initialize_decoder(self.decoder)
        self.initialize_head(self.segmentation_head)
        self.get_grad = Get_gradient()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        H_grad = self.get_grad(x)  # 预测结果的梯度
        features = self.encoder(H_grad)
        decoder_output = self.decoder(features)
        masks = self.segmentation_head(decoder_output)
        return masks

    @torch.no_grad()
    def predict(self, x):
        if self.training:
            self.eval()
        x = self.forward(x)
        return x


class UNet_Bou(nn.Module):
    @staticmethod
    def initialize_decoder(module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def initialize_head(module):
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def __init__(self):
        super().__init__()
        self.encoder = get_encoder(in_channels=6)  # 修改波段数的时候记得把梯度数也改回来
        # self.encoder2 = get_encoder(in_channels=8)
        self.decoder = get_decoder()
        self.segmentation_head = get_segmentation_head()

        # 初始化decoder和segmentation head中的Parameters
        self.initialize_decoder(self.decoder)
        self.initialize_head(self.segmentation_head)
        self.get_grad = Get_gradient()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        H_grad = self.get_grad(x)  # 预测结果的梯度
        cat = torch.cat((x, H_grad), 1)
        features = self.encoder(cat)
        decoder_output = self.decoder(features)
        masks = self.segmentation_head(decoder_output)
        return masks

    @torch.no_grad()
    def predict(self, x):
        if self.training:
            self.eval()
        x = self.forward(x)
        return x


if __name__ == "__main__":
    model = UNet()
    print(model)
    pass