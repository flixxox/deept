# deept #

This repository is used to train and search NMT models created with PyTorch.
It allows to easily define the model, and if needed optimizers and scores, using native PyTorch API.
Furthermore, it takes care of other functionalities used in training and search in an easy to understand manner.

The main features are:

- Easy to define models, optimizers and scores
- The code is lightweight and easy to understand
- Multi-threaded data loading
- Data-parallel multi-gpu training using Horovod
- An easy to use API to implement statefull models that are needed for stepwise decoding
- Implementation for Transformer (Base + Big) and stepwise BeamSearch
- Checkpoint manager implementing checkpoint delay, checkpoint frequency and checkpoint strategies ALL and BEST

## Setup ##

We use poetry to manage the python environment and manage dependencies.
Hence, a list of python dependencies can be found in `pyproject.toml`.
Since we had trouble installing Horovod with poetry, it must be installed manually for now.
This is only temporary and must be fixed in the future.

Except from the python dependencies we recommend to use cuda-11.3 and nccl-2.14.
On the AppTek cluster installations for these can be found in /home/fschmidt/lib. 
Before installing Horovod, the corresponding path variables must be set which is done in `/home/fschmidt/code/workflowManager/bash/setup_cuda.bash`.

### Install Poetry ####

```bash
curl -sSL https://install.python-poetry.org | POETRY_HOME=YOUR_POETRY_DIR python3 -
```

Add poetry to your PATH variable:
```bash
export PATH=YOUR_POETRY_DIR/bin:$PATH
```

### Install Python Dependencies except Horovod ####

With the following steps poetry will create a virtualenv, by default in ~/.cache/pypoetry/virtualenvs.
From the root folder call:
```bash
poetry shell
poetry install
```

If you manually want to select the virtualenv, run the following commands with your paths from our root folder.
```bash
python3 -m venv YOUR_VENV_FOLDER
source YOUR_VENV_FOLDER/bin/activate
pip3 install --upgrade pip3
poetry install
```

## Run ##

To this point, we provide the two main functionalities for machine translation models: training and search.
Both have their entry point in deept/train.py and deept/search.py.
Since we use argparse you can view a most up-to-date version of the parameters with
```bash
deept/train.py --help
deept/search.py --help
```

### Train ###

Usage of train.py:

```bash
usage: train.py [-h] --config CONFIG --output-folder OUTPUT_FOLDER [--resume-training RESUME_TRAINING] [--resume-training-from RESUME_TRAINING_FROM] [--number-of-gpus NUMBER_OF_GPUS]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       The path to the config.yaml which contains all user defined parameters.
  --output-folder OUTPUT_FOLDER
                        The folder in which to write the training output (ckpts, learning-rates, perplexities etc.)
  --resume-training RESUME_TRAINING
                        If you want to resume a training, set this flat to 1 and specify the directory with "resume-training-from".
  --resume-training-from RESUME_TRAINING_FROM
                        If you want to resume a training, specify the output directory here. We expect it to have the same layout as a newly created one.
  --number-of-gpus NUMBER_OF_GPUS
                        This is usually specified in the config but can also be overwritten from the cli.
```

### Search ###

Usage of search.py:

```bash
usage: search.py [-h] --config CONFIG --checkpoint-path CHECKPOINT_PATH [--output-folder OUTPUT_FOLDER] [--number-of-gpus NUMBER_OF_GPUS]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       The path to the config.yaml which contains all user defined parameters. It may or may not match the one trained with. This is up to the user to ensure.
  --checkpoint-path CHECKPOINT_PATH
                        The checkpoint.pt file containing the model weights.
  --output-folder OUTPUT_FOLDER
                        The output folder in which to write the score and hypotheses.
  --number-of-gpus NUMBER_OF_GPUS
                        This is usually specified in the config but can also be overwritten from the cli. However, in search this can only be 0 or 1. We do not support multi-gpu decoding. If you set it to >1 we will set it back to 1 so that you dont need to modify the config in search.
```