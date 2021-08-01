# BRECQ
Pytorch implementation of BRECQ, ICLR 2021

```latex
@article{li2021brecq,
  title={BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction},
  author={Li, Yuhang and Gong, Ruihao and Tan, Xu and Yang, Yang and Hu, Peng and Zhang, Qi and Yu, Fengwei and Wang, Wei and Gu, Shi},
  journal={arXiv preprint arXiv:2102.05426},
  year={2021}
}
```



## Update (Jul 30): Add Multi-GPU Reconstruction

We release the code for multi-GPU reconstruction. 

Note that this cannot be simply performed with `torch.nn.DataParallel` or DDP. To synchorize the gradients, activation scale, etc., we have to manully call `torch.distributed.allreduce`. 

The first step is to initialize the distributed envrionment, and then use distributed sampler for data loading. 

Please use `main_imagenet_dist` for multi-GPU reconstruction. With this, you can reconstruct larger models and use more data samples!

```bash
python -m main_imagenet_dist **KWARGS_FOR_RECON
```



## Pretrained models

We provide all the pretrained models and they can be accessed via  ```torch.hub```

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
Weight quantization accuracy: 66.32799530029297
Full quantization (W2A4) accuracy: 65.21199798583984
```

MobileNetV2 Quantization:

```bash
python main_imagenet.py --data_path PATN/TO/DATA --arch mobilenetv2 --n_bits_w 2 --channel_wise --weight 0.1
```

Results: `Weight quantization accuracy: 59.48799896240234`

