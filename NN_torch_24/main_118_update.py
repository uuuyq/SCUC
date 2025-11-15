import math
import os
import pickle
from functools import partial

import numpy
import pandas as pd
from time import time
import torch
from config import Config

def compare(trainer, config):
    num_instances = 8
    max_lag = 2
    num_threads = 4

    result, integrated_result = trainer.compare_multiprocess(
        num_instances=num_instances,
        fw_n_samples=50,
        file_name=f"3000_train_data_(1024,1024)_standard_gamma-{math.log10(gamma)}_max_lag-{max_lag}",
        timeout_sec=60 * 60 * 5,  # 整体的每个进程的时间限制
        max_lag=max_lag,
        num_threads=num_threads
    )





def main(trainer, config):
    trainer.load_dataset()


    trainer.train()

    compare(trainer, config)



if __name__ == '__main__':
    train_data_path = r"..\data_gen_24_bus118\train_data"
    tensor_data_path = r".\tensor_118"
    result_path = r".\result_118_11"
    gamma = 1e12
    config = Config(
        num_data=3000,
        num_stage=24,
        n_vars=378,
        feat_dim=238,  # instance_index, stage, feat(118 * 2)
        single_cut_dim=379,
        n_pieces=15,
        train_data_path=train_data_path,
        tensor_data_path=tensor_data_path,
        result_path=result_path,
        N_EPOCHS=200,
        batch_size=16,
        weight_decay=0.001,
        LEARNING_RATE=1e-4,
        hidden_arr=(1024, 1024),
        gamma=gamma,
        n_realizations=6,
        scenario_flag=False,
        x_input_flag=True,
        additional_data="prefix",
        standard_flag=True,
    )
    # 模型训练后数据保存位置
    from trainer_update import TrainerUpdate

    trainer = TrainerUpdate(config)
    main(trainer, config)



