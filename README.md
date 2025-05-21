# This is a Repo for Joint Flashback Adaptation
Our codes are built upon the LLama-Factory{https://github.com/hiyouga/LLaMA-Factory/}.
We modify the origin transformer code for multi-task gradient projection (PCGrad)
Run `cd LLaMA-Factory` and execute the following steps.

## 🛠️ Environment Setup
### 1. Install the dependencies for LLama-Factory and other dependencies.
Run `sh setup.sh`
### 2. Replace `trainer.py` in default transformers lib with our `trainer.py` file. 
It provides gradient projection (PCGrad) during training.
Run `mv trainer.py {your_transformers_location}/trainer.py`. 
Mine is `/usr/local/lib/python3.8/dist-packages/transformers/`

## Run Train
Before train, you should:
### 1. Download the models.
like `llama3.1-8B-instruct`
### 2. Download the datasets.
Configure them in `data/dataset_info.json`. 
The format should the same as example data `data/naturalinstructions_eval.json`
### 3. Revise the configuration.
Revise `config_train_JFA.yaml`. Revise `model_name_or_path, dataset, output_dir` and other parameters.
To train the model, run the following command:
Run `sh run_train.sh`

## Run Test
Before test, you should:
### Reivse the `config_predict_{your_task}.yaml`. 
Revise `model_name_or_path, output_dir, eval_dataset`. Reivse `adapter_name_or_path` to `{your_path}`, `finetuning_type=lora`
Run the command: `sh run_prefict.sh`