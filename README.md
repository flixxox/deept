# A Modular Deep Learning Toolkit

DeepT provides a generic framework to train and test models in PyTorch.
It establishes a standardized structure for train/test runs and provides an interface to insert user-specific code.
By providing core functionality in a modular fashion, DeepT simplifies the process of starting and managing PyTorch projects.

Core Features:

- **Comprehensive Training & Evaluation** – Supports training, evaluation, stepwise decoding, and hyperparameter search for PyTorch models.
- **Multi-GPU Training** – Enables data-parallel training across multiple GPUs.
- **Quantization Support** – Includes post-training quantization and quantization-aware training.
- **Cluster-Ready Hyperparameter Sweeping** – Uses an SQLite-based hyperparameter sweeper to coordinate sweep runs in distributed environments.
- **Checkpoint Management** – Handles step- and epoch-sensitive checkpoints, restarts crashed training runs, and supports various checkpointing strategies.
- **Minimal Overhead for Customization** – Lightweight and modular, making it easy to adapt for project-specific needs.
- **Comprehensive Logging** – Captures and logs all scores.
- **Advanced Configuration using YAML files** – Features a YAML-based configuration system using [cool-config](https://github.com/flixxox/yaml-config-mngr).
- **Stateful Model API** –  Stepwise decoding for stateful models.
- **Reproducibility** – IF needed, ensures deterministic behavior for consistent results.
- **Utility Tools** – Includes features for inspecting sweep databases and averaging checkpoints.


## Setup ##

DeepT uses poetry to manage the python environment and dependencies.
Thus, a list of python dependencies can be found in `pyproject.toml`.

### Install Poetry ####

```bash
curl -sSL https://install.python-poetry.org | POETRY_HOME=YOUR_POETRY_DIR python3 -
```

Add poetry to your PATH variable:
```bash
export PATH=YOUR_POETRY_DIR/bin:$PATH
```

### Install Python Dependencies ####

With the following steps poetry will create a virtualenv, by default in ~/.cache/pypoetry/virtualenvs.
From DeepT's root folder call:
```bash
poetry shell
poetry install
```

If you manually want to select the virtualenv, run the following commands with your paths from DeepT's root folder.
```bash
python3 -m venv YOUR_VENV_FOLDER
source YOUR_VENV_FOLDER/bin/activate
pip3 install --upgrade pip3
poetry install
```

## Run ##

To this point, we provide the three entry points to DeepT: train, eval, and search.
Since we use argparse you can view a most up-to-date version of the call instruction with
```bash
deept/train.py --help
deept/search.py --help
deept/eval.py --help
```

### Train ###

Usage of train.py:

```bash
usage: train.py [-h] --config CONFIG [--user-code USER_CODE [USER_CODE ...]] --output-folder OUTPUT_FOLDER
                [--resume-training RESUME_TRAINING] [--resume-training-from RESUME_TRAINING_FROM]
                [--number-of-gpus NUMBER_OF_GPUS] [--experiment-name EXPERIMENT_NAME] [--use-wandb USE_WANDB]

options:
  -h, --help            show this help message and exit
  --config CONFIG       The path to the config.yaml which contains all user defined parameters.
  --user-code USER_CODE [USER_CODE ...]
                        One or multiple paths to directories containing user code.
  --output-folder OUTPUT_FOLDER
                        The folder in which to write the training output (ckpts, learning-rates, perplexities etc.)
  --resume-training RESUME_TRAINING
                        If you want to resume a training, set this flag to 1 and specify the directory with "resume-
                        training-from".
  --resume-training-from RESUME_TRAINING_FROM
                        If you want to resume a training, specify the output directory here. We expect its folder
                        structure to have the same layout as a newly created one by DeepT.
  --number-of-gpus NUMBER_OF_GPUS
                        This is usually specified in the config but can also be overwritten from the cli.
  --experiment-name EXPERIMENT_NAME
                        The name of the experiment this training runs in.
  --use-wandb USE_WANDB
                        Whether the training shall be logged with wandb.
```

### Eval ###

Usage of eval.py:

```bash
usage: eval.py [-h] --config CONFIG [--user-code USER_CODE [USER_CODE ...]] --output-folder OUTPUT_FOLDER
               [--load-ckpt-from LOAD_CKPT_FROM] [--number-of-gpus NUMBER_OF_GPUS] [--use-wandb USE_WANDB]

options:
  -h, --help            show this help message and exit
  --config CONFIG       The path to the config.yaml which contains all user defined parameters.
  --user-code USER_CODE [USER_CODE ...]
                        One or multiple paths to directories containing user code.
  --output-folder OUTPUT_FOLDER
                        The folder in which to write the evaluation numbers.
  --load-ckpt-from LOAD_CKPT_FROM
                        The checkpoint to evaluate.
  --number-of-gpus NUMBER_OF_GPUS
                        This is usually specified in the config but can also be overwritten from the cli.
  --use-wandb USE_WANDB
                        Whether the evaluation shall be logged with wandb.
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


## Configuration ##

In DeepT the config documentation is embedded into the code.
Thus you can run the following and obtain an overview of the parameters and what they can be used for:
```bash
python3 /Users/fschmidt/code/deept/tools/print_deept_param_description.py
```