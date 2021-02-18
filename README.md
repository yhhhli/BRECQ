# BRECQ
Pytorch implementation of BRECQ, ICLR 2021

```latex
@inproceedings{
li2021brecq,
title={{\{}BRECQ{\}}: Pushing the Limit of Post-Training Quantization by Block Reconstruction},
author={Yuhang Li and Ruihao Gong and Xu Tan and Yang Yang and Peng Hu and Qi Zhang and Fengwei Yu and Wei Wang and Shi Gu},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=POWv6hDd9XH}
}
```



## Pretrained models

We provide all the pretrained models and they can accessed via  ```torch.hub```

For example: use ```res18 = torch.hub.load('yhhhli/BRECQ', model='resnet18', pretrained=True)``` to get the pretrained ResNet-18 model.

If you encounter URLError when downloading the pretrained network,  it's probably a network failure. 
An alternative way is to use wget to manually download the file,  then move it to `~/.cache/torch/checkpoints`, where the ```load_state_dict_from_url``` function will check before downloading it. 

For example:

```bash
wget https://github.com/yhhhli/BRECQ/releases/download/v1.0/resnet50_imagenet.pth.tar 
mv resnet50_imagenet.pth.tar ~/.cache/torch/checkpoints
```

## Usage

```bash
python main_imagenet.py --data_path PATN/TO/DATA --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 4 --act_quant --test_before_calibration
```

You can get the following output:

```bash
Quantized accuracy before brecq: 0.13599999248981476
Weight quantization accuracy: 66.2760009765625
Full quantization (W2A4) accuracy: 65.00599670410156
```



