

from deept.util.globals import Settings

__DATALOADER__ = {}

def register_dataloader(name):
    def register_dataloader_fn(cls):
        if name in __DATALOADER__:
            raise ValueError(f'Dataloader {name} already registered!')
        __DATALOADER__[name] = cls
        return cls
    return register_dataloader_fn

def create_dataloader_from_config(config, datapipe, shuffle=True):
    if config['dataloader', ''] in __DATALOADER__:
        dataloader = __DATALOADER__[config['dataloader']].create_from_config(config, datapipe, shuffle)
        return pipe
    else:
        return create_default_dataloader(config, datapipe, shuffle)
        
def create_default_dataloader(config, datapipe, shuffle):

    from torchdata.dataloader2 import DataLoader2
    from torchdata.dataloader2.adapter import Shuffle

    rs = None

    if Settings.get_number_of_workers() > 1:
        from torchdata.dataloader2 import DistributedReadingService
        rs = DistributedReadingService()

    if config['dataloader_workers', 1] > 1:
        from torchdata.dataloader2 import MultiProcessingReadingService
        if rs is not None:
            from torchdata.dataloader2 import SequentialReadingService
            rs = SequentialReadingService(
                rs,
                MultiProcessingReadingService(num_workers=config['dataloader_workers'])
            )
        else:
            rs = MultiProcessingReadingService(num_workers=config['dataloader_workers'])

    dataloader = DataLoader2(
        datapipe,
        [Shuffle(shuffle)],
        reading_service=rs
    )

    return dataloader

def get_all_dataloader_keys():
    return list(__DATALOADER__.keys())
