
import torch

from deept.util.debug import my_print
from deept.util.globals import Settings
from deept.data.datapipe import create_dp_from_config
from deept.data.dataloader import create_dataloader_from_config


def prepare_model_for_qat(config, model):
    # This might be needed if we support PyTorch quantization
    return model


class PostTrainingQuantizer:

    def __init__(self, **kwargs):
        
        for k, v in kwargs.items():
            setattr(self, k, v)

    def create_from_config(config):

        post_training_quant_type = config['post_training_quant_type', 'dynamic']
        calib_datapipe = None
        calib_dataloader = None

        if post_training_quant_type == 'calibration':

            calib_datapipe = create_dp_from_config(config,
                config['data_calib_root'],
                config['data_calib_mask'],
                name='calibration',
                chunk=False,
                drop_last=False,
                use_max_token_bucketize=False,
            )

            calib_dataloader = create_dataloader_from_config(config, calib_datapipe, 
                shuffle=False,
                num_worker_overwrite=1
            )

        return PostTrainingQuantizer(
            calib_datapipe = calib_datapipe,
            calib_dataloader = calib_dataloader,
            quant_type = post_training_quant_type,
            quant_calibration_steps = config['quant_calibration_steps', 0]
        )

    def quantize(self, model):
        
        size_before = self.get_model_size_in_kb(model)

        if self.quant_type == 'dynamic':
            model = self.quantize_dynamic(model)
        elif self.quant_type == 'calibration':
            model = self.calibrate(model)
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

    def calibrate(self, model):

        my_print('Quantization: Start calibration!')

        model = model.to(Settings.get_device())

        Settings.set_calibration_flag(True)

        steps = 0

        for data in self.calib_dataloader:

            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] =  data[k].to(Settings.get_device())

            with torch.no_grad():
                output, _ = model(*[data[k] for k in model.input_keys])

            steps += 1

            if steps >= self.quant_calibration_steps:
                my_print('Calibration finished!')
                break

        Settings.set_calibration_flag(False)

        return model

    def get_model_size_in_kb(self, model):
        size = 0
        for param in model.parameters():
            size += param.nelement() * param.element_size()
        for buffer in model.buffers():
            size += buffer.nelement() * buffer.element_size()
        return (size/1e3)
