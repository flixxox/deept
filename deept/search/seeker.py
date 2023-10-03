
import time
from os.path import join

import torch

from deept.util.globals import Settings
from deept.util.debug import my_print, print_memory_usage
from deept.data.postprocessing import get_postprocessing_fn


class Seeker:

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.print_per_step = len(self.keys_print_per_step) > 0

    @staticmethod
    def create_from_config(config, 
        dataloader, search_algorithm, checkpoint_count, corpus_size):
        
        ckpt_suffix = config['checkpoint_path'].split('ckpt-')[-1].replace('.pt', '')

        if ckpt_suffix.isdigit():
            output_file_prefix = join(Settings.get_dir('search_dir'), f'search_{checkpoint_count}')
        else:
            output_file_prefix = join(Settings.get_dir('search_dir'), f'search_{ckpt_suffix}')

        return Seeker(
            dataloader = dataloader,
            search_algorithm = search_algorithm,
            output_file_prefix = output_file_prefix,
            corpus_size = corpus_size,
            postprocessing_fn = get_postprocessing_fn(config),
            keys_print_per_step = config['search_print_per_step_keys', []],
        )

    def search(self):

        results = {}
        step_times = []
        read_order = []
        tokens_accum = 0

        start = time.perf_counter()

        for data in self.dataloader:
            
            data['tensors'] =  data['tensors'].to(Settings.get_device())

            start_step = time.perf_counter()

            cur_results = self.search_algorithm(*[data['tensors'][k] for k in self.search_algorithm.input_keys])

            end_step = time.perf_counter()
            step_times.append((end_step - start_step))

            cur_results = self.process_results(cur_results)
            results = self.append_cur_results(cur_results, results)
            read_order += data['__keys__']
            tokens_accum += self.get_number_of_tokens(data)

            if self.print_per_step:
                self.print_search_result(cur_results, data)

        self.write_results_to_files_in_order(read_order, results)

        end = time.perf_counter()

        print_memory_usage()

        time_in_s = end - start
        
        my_print(f"Searching dataset: {time_in_s:4.2f}s, {(time_in_s) / 60:4.2f}min")
        my_print(f"Average step time: {sum(step_times)/len(step_times):4.2f}s")
        my_print(f"Decoded {tokens_accum} tokens with {tokens_accum/time_in_s:4.2f} tokens/s")

        return results
    
    def process_results(self, results):
        processed_results = {}
        for k, v in results.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().detach()
                processed_results[k] = self.postprocessing_fn(k, v)
        return processed_results

    def append_cur_results(self, cur_results, results):
        for k, v in cur_results.items():
            if k in results:
                if isinstance(v, list):
                    results[k] += v
                else:
                    results[k].append(v)
            else:
                if isinstance(v, list):
                    results[k] = v
                else:
                    results[k] = [v]
        return results

    def get_number_of_tokens(self, data):
        num_tokens = 0 
        for k in self.search_algorithm.input_keys:
            inp = data[k].cpu().detach()
            inp = self.postprocessing_fn(k, inp)
            num_tokens += sum([len(e.split(' ')) for e in inp])
        return num_tokens

    def print_search_result(self, cur_results, data):
        
        values = []
        names = self.keys_print_per_step
        num_entries = -1

        for k in names:
            if k in cur_results:
                values.append(cur_results[k])
            elif k in data:
                d = data[k].cpu().detach()
                values.append(self.postprocessing_fn(k, d))
            else:
                raise ValueError(f'Error! Key {k} of search_print_per_step_keys not valid!')
        
        for entries in zip(*values):
            my_print('===')
            for i, entry in enumerate(entries):
                my_print(f'{names[i].ljust(10, " ")} : {entry}')

    def write_results_to_files_in_order(self, read_order, results):

        num_sentences = len(read_order)

        for k, v in results.items():

            assert len(v) == num_sentences, f"""For every decoded sample, one element must be stored
                in every results entry. Have searched samples {num_sentences}, elements in {k}: {len(v)}."""
    
            with open(self.output_file_prefix + f'_{k}', 'w') as file:

                for i in range(self.corpus_size):
                    if i in read_order:
                        idx = read_order.index(i)
                        file.write(v[idx] + "\n")
                    else:
                       file.write("<SENTENCE OMITTED BY DATAPIPE>\n")