import pathlib
import time

from sddip_script_update import sddipclassical
from multiprocessing import Pool
import logging
import multiprocessing
import os


# 设置日志配置
def setup_logger(log_file, process_name, sample_index):
    logger = logging.getLogger(f"{process_name}_{sample_index}")
    logger.setLevel(logging.INFO)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 将处理器添加到logger
    logger.addHandler(file_handler)

    return logger



def running(sample_index, log_file, cut_file_name, train_data_path):

    process_name = multiprocessing.current_process().name
    logger = setup_logger(log_file, process_name, sample_index)

    # Parameters
    test_case = "case6ww"
    n_stages = 24
    n_realizations = 6
    # Setup
    algo = sddipclassical.Algorithm(
        test_case,
        n_stages,
        n_realizations,
        logger,
        train_data_path=train_data_path
    )
    algo.n_samples_primary = 3
    logger.info(f"sample {sample_index+1}:")
    scenario_dir = os.path.join(train_data_path, "scenarios", f"{sample_index+1}_scenario.csv")
    # 重新初始化参数problem_params、存储字典等
    algo.init(scenario_dir)


    try:
        algo.run(cut_file_name)
    except KeyboardInterrupt:
        logger.warning("Shutdown request ... exiting")


def main(train_data_path):
    path = os.path.join(train_data_path, "cut_csv_sddip")
    index_list = []
    for i in range(1, 11):
        file = os.path.join(path, f"{i + 1}_cuts.csv")
        if not os.path.exists(file):
            index_list.append(i)
    print(f"index_list: {index_list}")
    for sample_index in index_list:
        log_file = os.path.join(train_data_path, "log", f"{sample_index + 1}_logs.txt")
        running(sample_index, log_file, f"{sample_index + 1}_cuts", train_data_path)



    # pool = Pool(7)
    # try:
    #     for sample_index in index_list:
    #         log_file = os.path.join(train_data_path, "log", f"{sample_index + 1}_logs.txt")
    #         pool.apply_async(running, args=(sample_index, log_file, f"{sample_index + 1}_cuts", train_data_path))
    # finally:
    #     pool.close()
    #     pool.join()


if __name__ == "__main__":

    train_data_path = r"./data_gen_24_bus6_CV/train_data"
    start = time.time()
    main(train_data_path)
    print(f"测试样例耗时: {time.time() - start}")  # 10个测试样例耗时: 1771.442275762558

    # result_proc
    num_cuts = 15  # 指定cuts数量
    train_data_path = pathlib.Path(r"D:\tools\workspace_pycharm\sddip-SCUC-6-24\ifr_data")  # 原始cut地址





