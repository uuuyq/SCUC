import math
import os
from pathlib import Path
import torch
from NN_torch_24.trainer_update import TrainerUpdate
from NN_torch_24.config import Config

def sampled_sddip(trainer):
    num_instances = 6
    data_sampled = trainer.sample_test_dataset(num_instances,
       )


    num_threads = 3
    max_iterations = 40
    sddip_fw_n_samples = 1
    sddip_timeout_sec = None
    trainer.sampled_sddip(data_sampled, sddip_fw_n_samples, max_iterations, sddip_timeout_sec, num_threads)
    


def compare_obj(trainer, data_sampled_sddip):
    obj_fw_n_samples = 200
    max_lag = 5
    num_threads = 3
    compare_timeout_sec = None
    trainer.compare_obj_multiprocess(data_sampled_sddip, obj_fw_n_samples, max_lag, compare_timeout_sec, num_threads)

def compare_LB(trainer, data_sampled_sddip):
    num_instances = 3
    num_threads = 3

    sddip_fw_n_samples = 10  # 重点在LB，因此计算obj的采样次数可以选择少一些，而如果obj也需要的话，需要增大采样次数
    max_iterations = 40
    compare_timeout_sec = None
    trainer.compare_LB_multiprocess(data_sampled_sddip, num_instances, sddip_fw_n_samples, max_iterations, compare_timeout_sec, num_threads)


def load_result(config, file_name):
    save_path = os.path.join(config.compare_path, file_name)
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"保存路径不存在: {save_path}")

    # 获取所有匹配的文件
    all_files = [f for f in os.listdir(save_path) if f.endswith(".pkl")]
    all_files.sort()  # 按文件名排序，保证顺序一致

    all_instances = []
    for file_name in all_files:
        file_path = os.path.join(save_path, file_name)
        instance_dict = torch.load(file_path)
        all_instances.append(instance_dict)

    print(f"加载{file_name}数据，总共：{len(all_instances)}")

    return all_instances

import threading
import time
import tracemalloc
import gc
import psutil

def monitor_memory_objects(interval=20.0):
    proc = psutil.Process()
    tracemalloc.start()
    
    while True:
        # 打印总内存
        mem = proc.memory_info()
        print(f"[主进程监控] RSS={mem.rss/1024**2:.2f} MB, VMS={mem.vms/1024**2:.2f} MB")

        # 获取当前 Python 对象数量按类型统计
        obj_counts = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            obj_counts[obj_type] = obj_counts.get(obj_type, 0) + 1
        
        # 打印数量最多的前10种对象类型
        top_types = sorted(obj_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print("---- Top object types ----")
        for t, count in top_types:
            print(f"{t}: {count}")
        
        # tracemalloc 打印前 10 大分配
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        print("---- Top 10 Python allocations (tracemalloc) ----")
        for stat in top_stats[:10]:
            print(stat)
        print("--------------------------\n")

        time.sleep(interval)

    

def main(trainer, config):
    print("config.compare_path: ", config.compare_path)

    trainer.load_dataset()

    trainer.train()

    # 启动监控线程
    # monitor_thread = threading.Thread(target=monitor_memory_objects, daemon=True)
    # monitor_thread.start()

    sampled_sddip(trainer)

    data_sampled_sddip = load_result(config, "sddip_result")
    compare_obj(trainer, data_sampled_sddip)

    data_sampled_sddip = load_result(config, "compare_obj_result")
    compare_LB(trainer, data_sampled_sddip)

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
                                       f"{config.num_data}_data_{hidden_arr}_standard-{standard_flag}_gamma-{math.log10(gamma)}-1128")

    main(trainer, config)




