
__SEARCH_ALGORITHM_DICT__ = {}

def register_search_algorithm(name):
    def register_search_algorithm_fn(cls):
        if name in __SEARCH_ALGORITHM_DICT__:
            raise ValueError(f'Search algorithm {name} already registered!')
        __SEARCH_ALGORITHM_DICT__[name] = cls
        return cls
    return register_search_algorithm_fn

def create_search_algorithm_from_config(config):
    if config['search_algorithm'] in __SEARCH_ALGORITHM_DICT__:
        from deept.util.debug import my_print
        search_algorithm = __SEARCH_ALGORITHM_DICT__[config['search_algorithm']].create_from_config(config)
        check_search_algorithm(search_algorithm)
        return search_algorithm
    else:
        raise ValueError(f'Error! Unrecognized search algorithm {config["search_algorithm"]}!')

def check_search_algorithm(search_algorithm):

    assert hasattr(search_algorithm, 'input_keys'), """Your search_algorithm must contain the member variable 'input_keys'.
        Those keys specify the tensors fed to the __call__ function of your search_algorithm.
        It is expected to be a list of strings."""

    assert isinstance(search_algorithm.input_keys, list) 

def get_all_search_algorithm_keys():
    return list(__SEARCH_ALGORITHM_DICT__.keys())
