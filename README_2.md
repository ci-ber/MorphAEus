# Paper 1367 - MorphAEus

MorphAEus Architecture 

![Architecture overview](./fig_architecture.png)

Overview Deep Learning Framework

![Framework overview](./dl.png)


# Installation guide: 

0). Set up wandb. (https://docs.wandb.ai/quickstart)
 *  Sign up for a free account at https://wandb.ai/site and then login to your wandb account.
 * Login to wandb with `wandb login`
 * Paste the API key when prompted. You will find your API key here: https://wandb.ai/authorize. 
 
1). Clone (or copy) paper_1367 to desired location
 `git clone https://anonymous.4open.science/r/paper_1367-E2E5/ *TARGET_DIR*` (cloning from anonymous repos is not possible) 
 * download repo to *TARGET_DIR*

2). Create a virtual environment with the needed packages (use conda_environment-osx.yaml for macOS)
```
cd ${TARGET_DIR}/paper_1367
conda env create -f conda_environment.yaml
source activate py308 *or* conda activate py308
```

3). Install pytorch
* *with cuda*: 
```
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
* *w/o cuda*:
```
pip3 install torch==1.9.1 torchvision==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```

4). Download the CXR dataset from kaggle (https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) and move it to the data folder inside the project. (Expected paths are also listed in the data/splits csv files. 

5). Run a script: 
* [Optional] set config 'task' to test and load model from ./weights/MorphAEus/best_model.pt * 

```
python core/Main.py --config_path projects/23_morphaeus_cvpr/configs/cxr/morphaeus.yaml
```

# That's it, enjoy! :rocket:
