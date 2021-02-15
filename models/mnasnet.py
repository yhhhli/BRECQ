# !/usr/bin/env python

import torch
import torch.nn as nn

__all__ = ['mnasnet']


class _InvertedResidual(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, expansion_factor):
        super(_InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        self.apply_residual = (in_ch == out_ch and stride == 1)
        self.layers = nn.Sequential(
            # Pointwise
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            BN(mid_ch),
            nn.ReLU(inplace=True),
            # Depthwise
            nn.Conv2d(mid_ch, mid_ch, kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=mid_ch, bias=False),
            BN(mid_ch),
            nn.ReLU(inplace=True),
            # Linear pointwise. Note that there's no activation.
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            BN(out_ch))

    def forward(self, input):
        if self.apply_residual:
            return self.layers(input) + input
        else:
            return self.layers(input)


def _stack(in_ch, out_ch, kernel_size, stride, exp_factor, repeats):
    """ Creates a stack of inverted residuals. """
    assert repeats >= 1
    # First one has no skip, because feature map size changes.
    first = _InvertedResidual(in_ch, out_ch, kernel_size, stride, exp_factor)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(
            _InvertedResidual(out_ch, out_ch, kernel_size, 1, exp_factor))
    return nn.Sequential(first, *remaining)


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(scale):
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * scale, 8) for depth in depths]


class MNASNet(torch.nn.Module):
    # Version 2 adds depth scaling in the initial stages of the network.
    _version = 2

    def __init__(self, scale=2.0, num_classes=1000, dropout=0.0):
        super(MNASNet, self).__init__()

        global BN
        BN = nn.BatchNorm2d

        assert scale > 0.0
        self.scale = scale
        self.num_classes = num_classes
        depths = _get_depths(scale)
        layers = [
            # First layer: regular conv.
            nn.Conv2d(3, depths[0], 3, padding=1, stride=2, bias=False),
            BN(depths[0]),
            nn.ReLU(inplace=True),
            # Depthwise separable, no skip.
            nn.Conv2d(depths[0], depths[0], 3, padding=1, stride=1,
                      groups=depths[0], bias=False),
            BN(depths[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(depths[0], depths[1], 1,
                      padding=0, stride=1, bias=False),
            BN(depths[1]),
            # MNASNet blocks: stacks of inverted residuals.
            _stack(depths[1], depths[2], 3, 2, 3, 3),
            _stack(depths[2], depths[3], 5, 2, 3, 3),
            _stack(depths[3], depths[4], 5, 2, 6, 3),
            _stack(depths[4], depths[5], 3, 1, 6, 2),
            _stack(depths[5], depths[6], 5, 2, 6, 4),
            _stack(depths[6], depths[7], 3, 1, 6, 1),
            # Final mapping to classifier input.
            nn.Conv2d(depths[7], 1280, 1, padding=0, stride=1, bias=False),
            BN(1280),
            nn.ReLU(inplace=True),
        ]
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                        nn.Linear(1280, num_classes))
        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        # Equivalent to global avgpool and removing H and W dimensions.
        x = x.mean([2, 3])
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out",
                                         nonlinearity="sigmoid")
                nn.init.zeros_(m.bias)


def mnasnet(**kwargs):
    model = MNASNet(**kwargs)
    return model

