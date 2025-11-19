import multiprocessing
import pickle
from functools import partial
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
    df.to_csv(save_path_csv , index=False)

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
    trainer.load_dataset()
    num_instances = 50

    # 选择test_dataset中的部分数据用于比较
    # sampled = trainer.sample_test_dataset(num_instances=num_instances, save_flag=True)
    # 计算sddip时间和obj
    # sampled_sddip = trainer.sampled_sddip(data_sampled=sampled)




    # sampled_sddip = torch.load(f"{trainer.data_path}/num-50_sampled_sddip_fixed.pkl")

    # 计算x和pred_cut, 重算截距
    # sampled_pred_re = trainer.pred_and_re(sampled_sddip, save_flag=True)

    # # compare
    # start = time()
    # print("start compare")
    # Trainer.compare(fw_n_samples=20, max_lag=3, data_sampled=data_sampled, num_instances=num_instances)
    # print(f"end compare, time spend {time() - start}")

    model_gamma_choice(trainer)




def main(trainer):

    # trainer.load_dataset()

    # trainer.train()

    # 返回的label_tensor是完整的，包含intercept
    # feat_tensor, x_tensor,  label_tensor, Q_tensor, pred_cuts_tensor, cuts_tensor = trainer.predict()

    # 使用trainer 比较obj和运行时间
    # num_instances = 50
    # result_storage_temp = trainer.compare(num_instances=num_instances, save_flag=True)
    # pickle.dump(result_storage_temp, open(f"./result_x_Q_cuts/result_{num_instances}.pkl", "wb"))

    # model_gamma_choice(trainer, num_instances=50)

    compare(trainer)


    """
    len(test_dataset) 750
    feat_tensor.shape torch.Size([17250, 14])
    x_tensor.shape torch.Size([17250, 15, 13])
    label_tensor.shape torch.Size([17250, 15, 1])
    Q_tensor.shape torch.Size([17250, 15, 1])
    pred_cuts_tensor.shape torch.Size([17250, 15, 14])
    cuts_tensor.shape torch.Size([17250, 15, 14])
    """
    # print("feat_tensor.shape", feat_tensor.shape)
    # print("x_tensor.shape", x_tensor.shape)
    # print("label_tensor.shape", label_tensor.shape)
    # print("Q_tensor.shape", Q_tensor.shape)
    # print("pred_cuts_tensor.shape", pred_cuts_tensor.shape)
    # print("cuts_tensor.shape", cuts_tensor.shape)



if __name__ == '__main__':
    train_data_path = r"../data_gen_bus6_24/train_data",
    tensor_data_path = r"./tensor"
    result_path = r"./result_7_20"
    gamma = 0.5
    config = Config(
        num_data=5000,
        num_stage=24,
        n_vars=13,
        feat_dim=14,  # instance_index, stage, feat(12)
        single_cut_dim=14,
        n_pieces=15,
        train_data_path=train_data_path,
        tensor_data_path=tensor_data_path,
        result_path=result_path,
        N_EPOCHS=200,
        batch_size=16,
        weight_decay=0.001,
        LEARNING_RATE=1e-4,
        hidden_arr=(512, 512),
        gamma=gamma,
        n_realizations=6,
        scenario_flag=False,
        x_input_flag=True
    )
    # 模型训练后数据保存位置
    from trainer import Trainer

    trainer = Trainer(config)

    main(trainer)


