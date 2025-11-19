import math
import os
from pathlib import Path

from NN_torch_24.trainer_fixed import TrainerFixed
from NN_torch_24.config import Config


def compare(trainer, config):
    num_instances = 10
    num_threads = 5
    max_iterations = 40
    max_lag = 5
    fw_n_samples = 200
    config.compare_path = os.path.join(result_path, "compare", f"fixed_{config.num_data}_train_data_(1024,1024)_standard_gamma-{math.log10(gamma)}_max_lag-{max_lag}_max_iterations-{max_iterations}_fw-{fw_n_samples}")
    try:
    
        result = trainer.compare_complete(
            num_instances = num_instances,
            # TODO 200个路径采样
            fw_n_samples=fw_n_samples,
            max_iterations=max_iterations,   # sddip迭代的最大次数，以及pred sddip和pred_re sddip 的最大迭代次数
            # TODO 绘制这个图只需要几个数据就行
            max_lag=max_lag,
            sddip_timeout_sec=None,
            compare_timeout_sec=None,
            num_threads=num_threads,
        )
    finally:
        from datetime import datetime
        import torch
        # 获取当前时间
        now = datetime.now()
        time_str = now.strftime("%Y%m%d_%H%M%S")
        torch.save(result, f"compare_result_{time_str}.pkl")




def main(trainer, config):
    trainer.load_dataset()

    trainer.train()

    compare(trainer, config)

if __name__ == '__main__':
    ROOT = Path(__file__).parent.resolve()  # 项目根目录
    train_data_path = ROOT / "data_gen_24_bus118" / "train_data"
    tensor_data_path = ROOT / "NN_torch_24" / "tensor_118"
    result_path = ROOT / "NN_torch_24" / "result_118_11"

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

    trainer = TrainerFixed(config)
    main(trainer, config)



