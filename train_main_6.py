import math
import os
from pathlib import Path
import torch
from NN_torch_24.trainer_update import TrainerUpdate
from NN_torch_24.config import Config


def get_prefix_int_list(folder_path):
    prefix_list = []
    
    for filename in os.listdir(folder_path):
        # 忽略目录，只处理文件
        full_path = os.path.join(folder_path, filename)
        if os.path.isfile(full_path):
            # 按 `_` 分割，取第一个部分
            prefix = filename.split('_')[0]
            try:
                prefix_int = int(float(prefix))
                prefix_list.append(prefix_int)
            except ValueError:
                # 如果前缀不是数字，跳过
                pass
    
    return prefix_list

def sampled_sddip(trainer, instance_index_list):
    num_instances = 30
    num_threads = 5
    max_iterations = 30

    

    data_sampled = trainer.sample_test_dataset(num_instances, instance_index_list
       )
    
    # sddip_fw_n_samples = 1
    # sddip_timeout_sec = None

    #  只是读取
    return trainer.sampled_sddip_read(data_sampled)
    


def compare_obj(trainer, data_sampled):
    obj_fw_n_samples = 200
    num_threads = 6
    max_lag = 5

    compare_timeout_sec = None
    trainer.compare_obj_multiprocess(data_sampled, obj_fw_n_samples, max_lag, compare_timeout_sec, num_threads)

def compare_LB(trainer, data_sampled):
    num_instances = 6 
    num_threads = 6
    sddip_fw_n_samples = 1  # 重点在LB，因此计算obj的采样次数可以选择少一些，而如果obj也需要的话，需要增大采样次数
    max_iterations = 25  # 太多也没有必要

    compare_timeout_sec = None
    trainer.compare_LB_multiprocess(data_sampled, num_instances, sddip_fw_n_samples, max_iterations, compare_timeout_sec, num_threads)

def compare_obj_stage(trainer, data_sampled, fw_n_samples):

    trainer.compare_obj_stage(data_sampled, fw_n_samples)




def load_result(config, file_name, instance_index_list=None):
    if instance_index_list is not None:
        print(f"加载指定的instance index: {instance_index_list}")

    save_path = os.path.join(config.compare_path, file_name)
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"保存路径不存在: {save_path}")

    # 获取所有匹配的文件
    all_files = [f for f in os.listdir(save_path) if f.endswith(".pkl")]
    all_files.sort()  # 按文件名排序，保证顺序一致

    all_instances = []

    for file_name in all_files:
        # 如果需要筛选特定 instance_index
        if instance_index_list is not None:
            # 文件名前缀，例如 '123.0_result.pkl' -> '123'
            prefix = file_name.split('.')[0]
            try:
                idx = int(prefix)
            except ValueError:
                continue  # 如果文件名不符合格式就跳过
            if idx not in instance_index_list:
                continue  # 不在筛选列表中就跳过

        file_path = os.path.join(save_path, file_name)
        instance_dict = torch.load(file_path)
        all_instances.append(instance_dict)

    print(f"加载{file_name}数据，总共：{len(all_instances)}")

    return all_instances

def main(trainer, config):
    print("config.compare_path: ", config.compare_path)

    trainer.load_dataset()

    trainer.train()

    instance_index_list=None

    # instance_index_list = get_prefix_int_list(os.path.join(config.compare_path, "compare_obj_result"))
    # print(f"已经存在的数据instance index: {instance_index_list}")

    data_sampled_sddip = sampled_sddip(trainer, instance_index_list)
    if data_sampled_sddip is None:
        data_sampled_sddip = load_result(config, "sddip_result")
    compare_obj(trainer, data_sampled_sddip)


    # print("start compare_obj_stage")
    # data_sampled = load_result(config, "compare_obj_result", instance_index_list=[1627])
    # compare_obj_stage(trainer, data_sampled, fw_n_samples=200)




    print("start compare_LB")
    data_sampled_sddip = load_result(config, "compare_obj_result", )
    compare_LB(trainer, data_sampled_sddip)

if __name__ == '__main__':
    ROOT = Path(__file__).parent.resolve()  # 项目根目录
    train_data_path = ROOT / "data_gen_24_bus6_CV" / "train_data"
    tensor_data_path = ROOT / "NN_torch_24" / "tensor_6_CV"
    result_path = ROOT / "NN_torch_24" / "result_6_CV"
    hidden_arr = (1024, 1024)
    gamma = 1e8
    standard_flag = True

    config = Config(
        num_data=1000,
        num_stage=24,
        n_vars=13,
        feat_dim=14,  # instance_index, stage, feat(6 * 2)
        single_cut_dim=14,
        n_pieces=15,
        train_data_path=train_data_path,
        tensor_data_path=tensor_data_path,
        result_path=result_path,
        N_EPOCHS=75,
        batch_size=16,
        weight_decay=0.001,
        LEARNING_RATE=1e-4,
        hidden_arr=hidden_arr,
        gamma=gamma,
        n_realizations=6,
        scenario_flag=False,
        x_input_flag=True,
        additional_data="prefix",
        standard_flag=standard_flag,
    )
    # 模型训练后数据保存位置

    trainer = TrainerUpdate(config)
    config.compare_path = os.path.join(result_path, "compare",
                                       f"{config.num_data}_data_{hidden_arr}_standard-{standard_flag}_gamma-{math.log10(gamma)}-1128")

    main(trainer, config)




