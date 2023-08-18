
import torch

from deept.util.debug import my_print
from deept.util.globals import Settings


def prepare_model_for_qat(config, model):
    
    model_fused = fuse_modules_if_applicable(model)

    model = check_fused_model(model, model_fused)

    return model

def fuse_modules_if_applicable(config, model):

    fused_model = copy.deepcopy(model)

    return model

def check_fused_model(model, model_fused):
    return model_fused


class PostTrainingQuantizer:

    def __init__(self, **kwargs):
        
        for k, v in kwargs.items():
            setattr(self, k, v)

    def create_from_config(config):
        return PostTrainingQuantizer(
            quant_type = config['quantization_type', 'dynamic']
        )

    def quantize(self, model):
        
        size_before = self.get_model_size_in_kb(model)

        if self.quant_type == 'dynamic':
            model = self.quantize_dynamic(model)
        else:
            raise ValueError(f'Unrecognized quantization type {self.quant_type}!')

        size_after = self.get_model_size_in_kb(model)
        
        my_print(f'Quantization: '
            f'Size before {size_before:.1f}KB, '
            f'Size after {size_after:.1f}KB, '
            f'Savings {((size_before-size_after)/size_before)*100:4.2f}%!'
        )

        return model

    def quantize_dynamic(self, model):
        my_print('Applying dynamic quantization!')
        return torch.ao.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

    def get_model_size_in_kb(self, model):
        size = 0
        for param in model.parameters():
            size += param.nelement() * param.element_size()
        for buffer in model.buffers():
            size += buffer.nelement() * buffer.element_size()
        return (size/1e3)
