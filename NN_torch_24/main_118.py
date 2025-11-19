import multiprocessing
import os
import pickle
from functools import partial

import numpy
import pandas as pd
from time import time
import torch
from NN_torch_24.config import Config


def model_gamma_choice(trainer):
    import math
    def error_ratio_calculate(obj_candidate, obj_optimal):
        error_ratio = [abs(obj_candidate[i] - obj_optimal[i]) / obj_optimal[i] for i in range(len(obj_candidate))]
        # 计算均值和方差
        average_ratio = sum(error_ratio) / len(error_ratio)
        std_ratio = math.sqrt(sum([(r - average_ratio) ** 2 for r in error_ratio]) / len(error_ratio))
        return average_ratio, std_ratio

    # 不同model使用相同的划分数据集
    trainer.load_dataset()

    model_error_ratio_res = {
        "gamma": [],
        "average": [],
        "std": [],
        "average_re": [],
        "std_re": [],
    }
    gamma_list = [0.0001, 0.001, 0.1, 0.5, 0.9, 1, 1.2, 1.6, 2, 10, 100, 1000]
    # gamma_list = [0.5]
    data_sampled = torch.load("result_7_20/compare/num-50_sampled_sddip_fixed.pkl")

    for gamma in gamma_list:
        trainer.gamma = gamma

        data_sampled = trainer.pred_and_re(data_sampled, save_flag=False)
        result = trainer.compare_obj(data_sampled)
        obj_pred = torch.tensor(result["obj_pred"], dtype=torch.float32)
        obj_prd_re = torch.tensor(result["obj_pred_re"], dtype=torch.float32)
        obj_sddip = torch.tensor(data_sampled["obj_sddip_only_one"], dtype=torch.float32)
        obj_pred = obj_pred.mean(dim=1)
        obj_prd_re = obj_prd_re.mean(dim=1)

        average, std = error_ratio_calculate(obj_pred, obj_sddip)
        average_re, std_re = error_ratio_calculate(obj_prd_re, obj_sddip)
        model_error_ratio_res["gamma"].append(gamma)
        model_error_ratio_res["average"].append(average)
        model_error_ratio_res["average_re"].append(average_re)
        model_error_ratio_res["std"].append(std)
        model_error_ratio_res["std_re"].append(std_re)

    save_path_csv = "./result_7_20/gamma_choice.csv"
    df = pd.DataFrame(model_error_ratio_res)
    # 保存为 CSV
    df.to_csv(save_path_csv, index=False)

    def model_gamma_choice_parallel(trainer, num_processes=None):
        """多进程并行处理gamma选择的函数"""

        def error_ratio_calculate(obj_candidate, obj_optimal):
            error_ratio = [abs(obj_candidate[i] - obj_optimal[i]) / obj_optimal[i]
                           for i in range(len(obj_candidate))]
            average_ratio = sum(error_ratio) / len(error_ratio)
            std_ratio = math.sqrt(sum([(r - average_ratio) ** 2 for r in error_ratio]) / len(error_ratio))
            return average_ratio, std_ratio

        # 加载数据集
        trainer.load_dataset()
        data_sampled = torch.load("result_7_20/compare/num-50_sampled_sddip_fixed.pkl")

        # 准备gamma列表
        gamma_list = [0.0001, 0.001, 0.1, 0.5, 0.9, 1, 1.2, 1.6, 2, 10, 100, 1000]

        # 设置进程数
        if num_processes is None:
            num_processes = min(len(gamma_list), multiprocessing.cpu_count())

        # 创建进程池
        with multiprocessing.Pool(processes=num_processes) as pool:
            # 准备工作函数
            worker_func = partial(
                _process_single_gamma,
                trainer=trainer,
                data_sampled=data_sampled
            )

            # 并行处理
            results = []
            for result in pool.imap(worker_func, gamma_list):
                results.append(result)

        # 整理结果
        model_error_ratio_res = {
            "gamma": [],
            "average": [],
            "std": [],
            "average_re": [],
            "std_re": [],
        }

        for gamma, avg, std, avg_re, std_re in results:
            model_error_ratio_res["gamma"].append(gamma)
            model_error_ratio_res["average"].append(avg)
            model_error_ratio_res["std"].append(std)
            model_error_ratio_res["average_re"].append(avg_re)
            model_error_ratio_res["std_re"].append(std_re)

        # 保存结果
        save_path_csv = "./result_7_20/gamma_choice_parallel.csv"
        pd.DataFrame(model_error_ratio_res).to_csv(save_path_csv, index=False)
        return model_error_ratio_res

    def _process_single_gamma(gamma, trainer, data_sampled):
        """处理单个gamma值的计算任务"""
        # 需要创建新的模型实例避免进程间冲突
        local_model = type(trainer)()
        local_model.load_state_dict(trainer.state_dict())
        local_model.load_dataset()  # 重新加载数据集

        # 设置gamma并计算
        local_model.gamma = gamma
        data = local_model.pred_and_re(data_sampled, save_flag=False)
        result = local_model.compare_obj(data)

        # 计算结果
        obj_pred = torch.tensor(result["obj_pred"], dtype=torch.float32).mean(dim=1)
        obj_prd_re = torch.tensor(result["obj_pred_re"], dtype=torch.float32).mean(dim=1)
        obj_sddip = torch.tensor(result["obj_sddip"], dtype=torch.float32).mean(dim=1)

        avg, std = error_ratio_calculate(obj_pred, obj_sddip)
        avg_re, std_re = error_ratio_calculate(obj_prd_re, obj_sddip)

        return (gamma, avg, std, avg_re, std_re)


def compare(trainer):
    num_instances = 7

    # 选择test_dataset中的部分数据用于比较
    sampled = trainer.sample_test_dataset(num_instances=num_instances, save_flag=True)

    # 计算sddip时间和obj
    print("sampled_read")
    sampled_sddip = trainer.sampled_read(data_sampled=sampled)



    # sampled_sddip = torch.load(os.path.join(trainer.config.result_path,"compare", "num-50_sampled_sddip_07-31-04-21.pkl"))

    # 计算x和pred_cut, 重算截距
    print("pred_and_re")
    sampled_pred_re = trainer.pred_and_re(sampled_sddip, save_flag=True)

    # pred sddip
    # sampled_sddip_pred_and_re_maxl = trainer.pred_sddip(sampled_pred_re, max_lag=5)

    # # compare
    # sampled_sddip_pred_and_re_maxl = torch.load(os.path.join(trainer.config.result_path, "compare", "num-5_sampled_sddip_pred_and_re_09-16-17-51.pkl"))
    print("compare")
    trainer.compare(fw_n_samples=50, max_lag=None, data_sampled=sampled_pred_re)

    # model_gamma_choice(trainer)




def main(trainer):
    trainer.load_dataset()


    trainer.train()

    # compare(trainer)



if __name__ == '__main__':
    train_data_path = r"..\data_gen_24_bus118\train_data"
    tensor_data_path = r".\tensor_118"
    result_path = r".\result_118_11"
    gamma = 0.05
    config = Config(
        num_data=4060,
        num_stage=24,
        n_vars=378,
        feat_dim=238,  # instance_index, stage, feat(118 * 2)
        single_cut_dim=379,
        n_pieces=15,
        train_data_path=train_data_path,
        tensor_data_path=tensor_data_path,
        result_path=result_path,
        N_EPOCHS=100,
        batch_size=16,
        weight_decay=0.001,
        LEARNING_RATE=1e-4,
        hidden_arr=(1024, 1024),
        gamma=gamma,
        n_realizations=6,
        scenario_flag=False,
        x_input_flag=True,
        additional_data="prefix"
    )
    # 模型训练后数据保存位置
    from trainer import Trainer

    trainer = Trainer(config)
    main(trainer)



