## 💻 Installation

M2RL is built on top of imitation learning frameworks [GNFactor](https://github.com/YanjieZe/GNFactor/tree/main), [PerAct](https://github.com/peract/peract), [RLBench](https://github.com/stepjam/RLBench).
It is recommended to have a conda envirionment for running the code.

# 0 create python env

```
conda remove --m2rl_env --all
conda create -n m2rl_env python=3.9
conda activate m2rl_env
```

# 1 download coppeliasim 
```
wget https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz --no-check-certificate

tar -xvf CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz

rm CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz
```

# 2 add following lines to your `~/.bashrc` file. 
Remember to source your bashrc (source ~/.bashrc) and reopen a new terminal then.

You should replace the path here with your own path to the coppeliasim installation directory.
```
export COPPELIASIM_ROOT=EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT

export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

# 3 install PyRep
You should open a new terminal here, to make your .bashrc work.
```
cd <install_dir>
git clone https://github.com/stepjam/PyRep.git
cd PyRep
```

# 4 install RLBench
```
cd <install_dir>
git clone -b peract https://github.com/MohitShridhar/RLBench.git # note: 'peract' branch

cd RLBench
pip install -r requirements.txt
python setup.py develop
```

# 5 install YARR
```
cd <install_dir>
git clone -b peract https://github.com/MohitShridhar/YARR.git # note: 'peract' branch

cd YARR
pip install -r requirements.txt
python setup.py develop
```

# 6 install M2RL
```
Clone:
cd <install_dir>
git clone https://github.com/M2RL/m2rl-dataset.git
```


```
cd m2rl-dataset/code
pip install -r requirements.txt
python setup.py develop
```



## Quickstart

## Updating dataset variables
```
Please update the following variables in your config.py file.
C.TRAIN_REPLAY_STORAGE_DIR = '<path-to-training-data>'
C.TRAIN_REPLAY_STORAGE_DIR = '<path-to-test-data>'
C.data_dir = "<path-to-dataset>"

```

## 📈 Training and Evaluation
```
python train.py
```

## 📝 Citation

If you find our work useful, please consider citing:
```
@article{M2RLDataset,
  title={M2RL: A Multimodal Multi-Interface Dataset for Robot Learning from Human Demonstrations},
  author={Anonymous},
  journal={ICMI}, 
  year={2024},
}
```
