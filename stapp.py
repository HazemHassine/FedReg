import contextlib
import io
import json
import pprint
import sys
from git import os

import streamlit as st
from FedUtils.models.mnist.cnn import Model
from functools import partial
from FedUtils.fed.fedreg import FedReg
from torch.optim import SGD

def main(config_file_instance):
    import numpy as np
    import random
    import torch
    from FedUtils.models.utils import read_data, CusDataset, ImageDataset
    from torch.utils.data import DataLoader
    from loguru import logger
    from functools import partial
    import os
    torch.backends.cudnn.deterministic = True

    config = config_file_instance
    logger.add(config["log_path"])

    random.seed(1+config["seed"])
    np.random.seed(12+config["seed"])
    torch.manual_seed(123+config["seed"])
    torch.cuda.manual_seed(123+config["seed"])

    Model = config["model"]
    inner_opt = config["inner_opt"]
    if "landmarks" in config["train_path"]:  # load landmark data
        assert "image_path" in config
        Dataset = partial(ImageDataset, image_path=config["image_path"])
        clients, groups, train_data, eval_data = read_data(config["train_path"], config["test_path"])
    else:  # load other data
        clients, groups, train_data, eval_data = read_data(config["train_path"], config["test_path"])
        Dataset = CusDataset

    if config["use_fed"]:
        Optimizer = config["optimizer"]
        t = Optimizer(config, Model, [clients, groups, train_data, eval_data], train_transform=config["train_transform"],
                      test_transform=config['test_transform'], traincusdataset=Dataset, evalcusdataset=Dataset)
        t.train()
    else:
        train_data_total = {"x": [], "y": []}
        eval_data_total = {"x": [], "y": []}
        for t in train_data:
            train_data_total["x"].extend(train_data[t]["x"])
            train_data_total["y"].extend(train_data[t]["y"])
        for t in eval_data:
            eval_data_total["x"].extend(eval_data[t]["x"])
            eval_data_total["y"].extend(eval_data[t]["y"])
        train_data_size = len(train_data_total["x"])
        eval_data_size = len(eval_data_total["x"])
        train_data_total_fortest = DataLoader(Dataset(train_data_total, config["test_transform"]), batch_size=config["batch_size"], shuffle=False,)
        train_data_total = DataLoader(Dataset(train_data_total, config["train_transform"]), batch_size=config["batch_size"], shuffle=True, )
        eval_data_total = DataLoader(Dataset(eval_data_total, config["test_transform"]), batch_size=config["batch_size"], shuffle=False,)
        model = Model(*config["model_param"], optimizer=inner_opt)
        for r in range(config["num_rounds"]):
            model.solve_inner(train_data_total)
            stats = model.test(eval_data_total)
            train_stats = model.test(train_data_total_fortest)
            logger.info("-- Log At Round {} --".format(r))
            logger.info("-- TEST RESULTS --")
            logger.info("Accuracy: {}".format(stats[0]*1.0/eval_data_size))
            logger.info("-- TRAIN RESULTS --")
            logger.info(
                "Accuracy: {} Loss: {}".format(train_stats[0]/train_data_size, train_stats[1]/train_data_size))



class config:
    def __init__(self):
        self.clients_per_round = clients_per_round
        self.num_rounds = num_rounds
        self.eval_every = eval_every
        self.drop_percent = drop_percent
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.config = {
            "seed": 1,  # random seed
            "model": partial(Model, learning_rate=1e-1, p_iters=10, ps_eta=2e-1, pt_eta=2e-3),  # the model to be trained
            "inner_opt": None,  # optimizer, in FedReg, only the learning rate is used
            "optimizer": FedReg,  # FL optimizer, can be FedAvg, FedProx, FedCurv or SCAFFOLD
            "model_param": (10,),  # the input of the model, used to initialize the model
            "inp_size": (784,),  # the input shape
            "train_path": "data/mnist_10000/data/train/",  # the path to the train data
            "test_path": "data/mnist_10000/data/valid/",  # the path to the test data
            "use_fed": 1,  # whether use federated learning alrogithms
            "log_path": "tasks/mnist/FedReg/train.log",  # the path to save the log file
            "train_transform": None,  # the preprocessing of train data, please refer to torchvision.transforms
            "test_transform": None,  # the preprocessing of test dasta
            "eval_train": True,  # whether to evaluate the model performance on the training data. Recommend to False when the training dataset is too large
            "clients_per_round": self.clients_per_round,
            "num_rounds": self.num_rounds,
            "eval_every": self.eval_every,
            "drop_percent": self.drop_percent,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "gamma": self.gamma
        }
        pass
    def update_config(self):

        self.config = {**self.config, **{
            "optimizer": self.get_optimizer(algorithm),
            "clients_per_round": self.clients_per_round,
            "num_rounds": self.num_rounds,
            "eval_every": self.eval_every,
            "drop_percent": self.drop_percent,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "gamma": self.gamma
            }
        }
    
    def get_optimizer(self,algorithm):
        match algorithm.lower():
            case "fedavg":
                from FedUtils.fed.fedavg import FedAvg
                return FedAvg
            case "fedprox":
                from FedUtils.fed.fedprox import FedProx
                return FedProx
            case "fedcurv":
                from FedUtils.fed.fedcurv import FedCurv
                return FedCurv
            case "fedreg":
                return FedReg
            case "scaffold":
                from FedUtils.fed.scaffold import SCAFFOLD
                return SCAFFOLD

    def get_config(self) -> dict:
        return self.config


def run(configuration):
    import threading
    thread = threading.Thread(target=main, args=[configuration])
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        thread.start()
        prev = ""
        while True:
            if prev != stdout.getvalue():
                prev = stdout.getvalue()
                st.write(prev)

st.header("Federated Learning Algorithms testing")
algorithm = st.selectbox('Select Algorithm name:',('FedAvg', 'FedProx', 'FedReg', 'Scaffold', 'FedCurv'), index=2)
st.write('You selected:', algorithm)
st.subheader("Hyper parameters")
# st.number_input("model_param")
col1, col2 = st.columns(2)
clients_per_round = 20
num_rounds = 10
eval_every = 1
drop_percent = 0.1
num_epochs = 5
batch_size = 64
gamma = 0.4
config_file = config()
with col1:
    clients_per_round = st.number_input("Clients per round", step=1, on_change=config_file.update_config())
    st.caption("number of clients sampled in each round")
    num_rounds = st.number_input("Number of rounds", step=1, on_change=config_file.update_config())
    st.caption("number of total rounds")
    eval_every = st.number_input("Evaluate every", step=1, on_change=config_file.update_config())
    st.caption("the number of rounds to evaluate the model performance. 1 is recommend here.")
    drop_percent = st.number_input("Drop percent" , step=0.1, on_change=config_file.update_config())
    st.caption("the rate to drop a client. 0 is used in our experiments")
with col2:
    num_epochs = st.number_input("Number epochs", step=1, on_change=config_file.update_config())
    st.caption("the number of epochs in local training stage")
    batch_size = st.number_input("Batch size", step=16, on_change=config_file.update_config())
    st.caption("the batch size in local training stage")
    gamma = st.number_input("Gamma", step=0.1, on_change=config_file.update_config())
    st.caption("the value of gamma when FedReg is used, the weight for the proximal term when FedProx is used, or the value of lambda when FedCurv is used")
    see_config = st.button("see config file")
if see_config:
    st.code(pprint.pformat(config_file.get_config(), indent=2))

details = st.text_input("Enter experiments details")
run = st.button("Run experiment", on_click=run(config_file.get_config()))
st.caption("Click the button only once")
