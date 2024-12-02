import torch
import torch.nn as nn


class CustomSequential(nn.Sequential):
    def forward(self, x, memory_flag=False):

        for module in self:
            x = module(x, memory_flag).to(x.device)
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):  # For ResNet18 and ResNet34
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        downsample=None,
        stride=1,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        name=None,
    ):
        super(BasicBlock, self).__init__()
        self.name = name
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        if downsample is not None:
            self.downsample_conv, self.downsample_bn = (
                self.downsample[0],
                self.downsample[1],
            )

    def forward(self, x, memory_flag=False):
        identity = x

        out = self.conv1(x)
        if not memory_flag:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if not memory_flag:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample_conv(x)
            if not memory_flag:
                identity = self.downsample_bn(identity)

        out += identity
        out = self.relu(out)

        return out

    def __str__(self):
        return "{} contains...".format(type(self).__name__)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        downsample=None,
        stride=1,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        name=None,
    ):
        super(Bottleneck, self).__init__()
        self.name = name
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if downsample is not None:
            self.downsample_conv, self.downsample_bn = (
                self.downsample[0],
                self.downsample[1],
            )

    def forward(self, x, memory_flag=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample_conv(x)
            if not memory_flag:
                identity = self.downsample_bn(identity)

        out += identity
        out = self.relu(out)

        return out

    def __str__(self):
        return "{} contains...".format(type(self).__name__)


class ResNet(nn.Module):  # layers in each block - [3,4,6,3] in ResNet50
    def __init__(
        self,
        block,
        layers: list,
        image_channels,
        num_classes_total=100,
        num_tasks=10,
        multi_head=False,
        args=None,
    ):
        super(ResNet, self).__init__()
        self.num_classes_total = num_classes_total
        self.num_tasks = num_tasks

        self.inplanes = 64

        assert args is not None, "you should pass args to resnet"
        if "cifar" in args["dataset"].lower():
            self.conv1 = nn.Sequential(
                nn.Conv2d(
                    image_channels,
                    self.inplanes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(inplace=True),
            )
        elif "imagenet" in args["dataset"].lower():
            if "init_cls" in args.keys():
                if args["init_cls"] == args["increment"]:
                    self.conv1 = nn.Sequential(
                        nn.Conv2d(
                            image_channels,
                            self.inplanes,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False,
                        ),
                        nn.BatchNorm2d(self.inplanes),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    )
            else:
                self.conv1 = nn.Sequential(
                    nn.Conv2d(
                        image_channels,
                        self.inplanes,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.inplanes),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers

        self.layer1 = self._make_layer(
            block, layers[0], planes=64, stride=1, block_name="layer1"
        )  # 256 out
        self.layer2 = self._make_layer(
            block,
            layers[1],
            planes=128,
            stride=2,
            block_name="layer2",
        )
        self.layer3 = self._make_layer(
            block,
            layers[2],
            planes=256,
            stride=2,
            block_name="layer3",
        )
        self.layer4 = self._make_layer(
            block,
            layers[3],
            planes=512,
            stride=2,
            block_name="layer4",
        )  # 2048 out

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512*4, num_classes_total)
        self.out_dim = 512 * block.expansion
        if multi_head:
            self.task_classifiers = nn.ModuleList(
                [
                    nn.Linear(self.out_dim, self.num_classes_total // self.num_tasks)
                    for _ in range(self.num_tasks)
                ]
            )
        else:
            self.task_classifiers = nn.ModuleList(
                [nn.Linear(self.out_dim, self.num_classes_total)]
            )

    def forward(self, x, task_id, memory_flag=False):
        x = self.conv1(x)

        x = self.layer1(x, memory_flag=memory_flag)
        x = self.layer2(x, memory_flag=memory_flag)
        x = self.layer3(x, memory_flag=memory_flag)
        x = self.layer4(x, memory_flag=memory_flag)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.task_classifiers[task_id](x)

        return x

    def _make_layer(self, block, num_residual_blocks, planes, stride, block_name):
        downsample = None
        layers = []

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample_conv = conv1x1(
                self.inplanes, planes * block.expansion, stride
            ).to(self.conv1[0].weight.device)
            downsample_bn = nn.BatchNorm2d(planes * block.expansion).to(
                self.conv1[0].weight.device
            )
            downsample = [downsample_conv, downsample_bn]

        layers.append(block(self.inplanes, planes, downsample, stride, name=block_name))
        self.inplanes = planes * block.expansion

        for i in range(1, num_residual_blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    name=block_name,
                )
            )  # 256, 64

        return CustomSequential(*layers)


def ResNet18(
    img_channels=3, num_classes_total=1000, num_tasks=10, multi_head=False, args=None
):
    return ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        img_channels,
        num_classes_total,
        num_tasks,
        args=args,
        multi_head=multi_head,
    )


def ResNet50(
    img_channels=3, num_classes_total=1000, num_tasks=10, multi_head=False, args=None
):
    return ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        img_channels,
        num_classes_total,
        num_tasks,
        args=args,
        multi_head=multi_head,
    )


def ResNet101(img_channels=3, num_classes_total=1000, multi_head=False):
    return ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        img_channels,
        num_classes_total,
        multi_head=multi_head,
    )


def ResNet152(img_channels=3, num_classes_total=1000, multi_head=False):
    return ResNet(
        Bottleneck,
        [3, 8, 36, 3],
        img_channels,
        num_classes_total,
        multi_head=multi_head,
    )
