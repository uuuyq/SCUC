import math
import os
from pathlib import Path

from NN_torch_24.trainer_update import TrainerUpdate
from NN_torch_24.config import Config

def sampled_sddip(trainer):
    num_instances = 10
    num_threads = 5
    max_iterations = 40
    sddip_fw_n_samples = 10
    sddip_timeout_sec = None
    data_sampled_sddip = trainer.sampled_sddip(num_instances, sddip_fw_n_samples, max_iterations, sddip_timeout_sec, num_threads)
    return data_sampled_sddip


def compare_obj(trainer, data_sampled_sddip):
    obj_fw_n_samples = 200
    max_lag = 5
    num_threads = 5
    compare_timeout_sec = None
    compare_obj_result = trainer.compare_obj_multiprocess(data_sampled_sddip, obj_fw_n_samples, max_lag, compare_timeout_sec, num_threads)
    return compare_obj_result

def compare_LB(trainer, data_sampled_sddip):
    num_instances = 5
    num_threads = 5

    sddip_fw_n_samples = 10  # 重点在LB，因此计算obj的采样次数可以选择少一些，而如果obj也需要的话，需要增大采样次数
    max_iterations = 40
    compare_timeout_sec = None
    trainer.compare_LB_multiprocess(data_sampled_sddip, num_instances, sddip_fw_n_samples, max_iterations, compare_timeout_sec, num_threads)


def main(trainer, config):
    trainer.load_dataset()

    trainer.train()

    data_sampled_sddip = sampled_sddip(trainer)

    compare_obj_result = compare_obj(trainer, data_sampled_sddip)

if __name__ == '__main__':
    ROOT = Path(__file__).parent.resolve()  # 项目根目录
    train_data_path = ROOT / "data_gen_24_bus118" / "train_data"
    tensor_data_path = ROOT / "NN_torch_24" / "tensor_118"
    result_path = ROOT / "NN_torch_24" / "result_118_11"
    hidden_arr = (1024, 1024)
    gamma = 1e12
    standard_flag = True
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
                                       f"{config.num_data}_data_{hidden_arr}_standard-{standard_flag}_gamma-{math.log10(gamma)}")

    main(trainer, config)




