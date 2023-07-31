
import time

import torch
import torch.nn as nn

from deept.util.debug import my_print
from deept.util.globals import Settings



class ContextTimer:

    timings = {}

    def __init__(self, name):
        
        self.name = name
        self.timestamp_start = None

        if self.name not in ContextTimer.timings.keys():
            ContextTimer.timings[self.name] = 0.

    @staticmethod
    def print_summary():

        total_time = 0

        my_print('======== ContextTimer MEASURMENTS ========')

        for k, v in ContextTimer.timings.items():
            total_time += v

        for k,v in ContextTimer.timings.items():
            my_print(f'{k}'.ljust(50, ' '), f'{v:4.3f}s {(v/total_time)*100:4.1f}%')
            ContextTimer.timings[k] = 0.

    @staticmethod
    def timestamp():
        return time.perf_counter()

    def start(self):
        self.timestamp_start = ContextTimer.timestamp()

    def end(self):
        ContextTimer.timings[self.name] += ContextTimer.timestamp() - self.timestamp_start

    def __enter__(self):
        if Settings.do_timing():
            self.start()

    def __exit__(self, exception_type, exception_value, traceback):
        if Settings.do_timing():
            self.end()
