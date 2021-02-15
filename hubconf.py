from models.resnet import resnet18 as _resnet18
from models.resnet import resnet50 as _resnet50
from models.mobilenetv2 import mobilenetv2 as _mobilenetv2
from models.mnasnet import mnasnet as _mnasnet
from models.regnet import regnetx_600m as _regnetx_600m
from models.regnet import regnetx_3200m as _regnetx_3200m
from torch.hub import load_state_dict_from_url

dependencies = ['torch']


def resnet18(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _resnet18(**kwargs)
    if pretrained:
        load_url = 'https://github.com/yhhhli/BRECQ/releases/download/v1.0/resnet18_imagenet.pth.tar'
        checkpoint = load_state_dict_from_url(url=load_url, map_location='cpu', progress=True)
        model.load_state_dict(checkpoint)
    return model


def resnet50(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _resnet50(**kwargs)
    if pretrained:
        load_url = 'https://github.com/yhhhli/BRECQ/releases/download/v1.0/resnet50_imagenet.pth.tar'
        checkpoint = load_state_dict_from_url(url=load_url, map_location='cpu', progress=True)
        model.load_state_dict(checkpoint)
    return model


def mobilenetv2(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _mobilenetv2(**kwargs)
    if pretrained:
        load_url = 'https://github.com/yhhhli/BRECQ/releases/download/v1.0/mobilenetv2.pth.tar'
        checkpoint = load_state_dict_from_url(url=load_url, map_location='cpu', progress=True)
        model.load_state_dict(checkpoint['model'])
    return model


def regnetx_600m(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _regnetx_600m(**kwargs)
    if pretrained:
        load_url = 'https://github.com/yhhhli/BRECQ/releases/download/v1.0/regnet_600m.pth.tar'
        checkpoint = load_state_dict_from_url(url=load_url, map_location='cpu', progress=True)
        model.load_state_dict(checkpoint)
    return model


def regnetx_3200m(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _regnetx_3200m(**kwargs)
    if pretrained:
        load_url = 'https://github.com/yhhhli/BRECQ/releases/download/v1.0/regnet_3200m.pth.tar'
        checkpoint = load_state_dict_from_url(url=load_url, map_location='cpu', progress=True)
        model.load_state_dict(checkpoint)
    return model


def mnasnet(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _mnasnet(**kwargs)
    if pretrained:
        load_url = 'https://github.com/yhhhli/BRECQ/releases/download/v1.0/mnasnet.pth.tar'
        checkpoint = load_state_dict_from_url(url=load_url, map_location='cpu', progress=True)
        model.load_state_dict(checkpoint)
    return model
