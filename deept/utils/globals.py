
class Context:
    """Objects used to train and search."""

    __CONTEXT = {}

    @staticmethod
    def add_context(key, obj):
        if key not in Context.__CONTEXT:
            Context.__CONTEXT[key] = obj
        else:
            from deept.utils.debug import my_print
            my_print(f'Warning! Context {key} is already set. Skipping!')

    @staticmethod
    def print_all_context_elements():
        for k, v in Context.__CONTEXT.items():
            print(k, v)

    @staticmethod
    def has_context(key):
        return key in Context.__CONTEXT
    
    @staticmethod
    def overwrite(key, obj):
        if key in Context.__CONTEXT:
            Context.__CONTEXT[key] = obj
        else:
            from deept.utils.debug import my_print
            my_print(f'Warning! Context {key} has not been set yet!')

    @staticmethod
    def reset():
        Context.__CONTEXT = {}

    def __class_getitem__(cls, key):
        return Context.__CONTEXT[key]


class Settings:
    """Internal config variables."""

    __IS_TRAIN  = None
    __IS_CALIB  = False
    __TIME      = None
    __WORKERS   = None
    __GPU       = True
    __DEVICE    = None
    __SEED      = None
    __RANK      = None
    __USE_WANDB = None
    __DIRS      = {}

    @staticmethod
    def reset_directories():
        Settings.__DIRS = {}

    @staticmethod
    def set_train_flag(flag):
        if Settings.__IS_TRAIN is None:
            Settings.__IS_TRAIN = flag

    @staticmethod
    def is_training():
        return Settings.__IS_TRAIN

    @staticmethod
    def set_calibration_flag(flag):
        Settings.__IS_CALIB = flag

    @staticmethod
    def is_calibration():
        return Settings.__IS_CALIB

    @staticmethod
    def set_time_flag(flag):
        if Settings.__TIME is None:
            Settings.__TIME = flag

    @staticmethod
    def do_timing():
        return Settings.__TIME

    @staticmethod
    def set_number_of_workers(workers, force=False):
        if Settings.__WORKERS is None or force:
            Settings.__WORKERS = workers

    @staticmethod
    def get_number_of_workers():
        return Settings.__WORKERS

    @staticmethod
    def set_cpu():
        if Settings.__GPU:
            Settings.__GPU = False

    @staticmethod
    def is_gpu():
        return Settings.__GPU

    @staticmethod
    def set_device(device):
        if Settings.__DEVICE is None:
            Settings.__DEVICE = device

    @staticmethod
    def get_device():
        return Settings.__DEVICE

    @staticmethod
    def set_global_seed(seed):
        Settings.__SEED = seed
    
    @staticmethod
    def get_global_seed():
        return Settings.__SEED

    @staticmethod
    def increase_global_seed():
        Settings.__SEED += 1

    @staticmethod
    def set_rank(rank):
        if Settings.__RANK is None:
            Settings.__RANK = rank
    
    @staticmethod
    def rank():
        return Settings.__RANK

    @staticmethod
    def add_dir(name, path):
        if path not in Settings.__DIRS:
            Settings.__DIRS[name] = path
        else:
            from deept.utils.debug import my_print
            my_print(f'Warning! Directory {name} is already set. Skipping!')

    @staticmethod
    def get_dir(name):
        return Settings.__DIRS[name]

    @staticmethod
    def has_dir(name):
        return name in Settings.__DIRS

    @staticmethod
    def set_use_wandb(value):
        Settings.__USE_WANDB = value

    @staticmethod
    def use_wandb():
        return Settings.__USE_WANDB
