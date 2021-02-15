import torch.nn as nn
from quant.quant_block import specials, BaseQuantBlock
from quant.quant_layer import QuantModule, StraightThrough
from quant.fold_bn import search_fold_and_remove_bn


class QuantModel(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.model = model
        search_fold_and_remove_bn(self.model)
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)
        self.use_spike = False

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, weight_quant_params, act_quant_params))

            elif isinstance(child_module, nn.Conv2d):
                setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, input):
        return self.model(input)

