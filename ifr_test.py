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

    # 输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


    return logger



def running(sample_index, log_file, train_data_path):

    process_name = multiprocessing.current_process().name
    logger = setup_logger(log_file, process_name, sample_index)

    # Parameters
    test_case = "case6ww"
    n_stages = 24
    n_realizations = 6
    # n_stages = 6
    # n_realizations = 6
    # Setup
    algo = sddipclassical.Algorithm(
        test_case,
        n_stages,
        n_realizations,
        logger,
        train_data_path=train_data_path
    )
    algo.n_samples_primary = 3
    logger.info(f"sample {sample_index}:")
    scenario_dir = r"D:\Desktop\SCUC\SCUC\data\01_test_cases\case6ww\t06_n06\scenario_data.txt"
    # scenario_dir = os.path.join(train_data_path, "scenarios", "scenario_data.txt")
    # 重新初始化参数problem_params、存储字典等
    algo.init(scenario_dir)


    try:
        algo.run_IFR()
    except KeyboardInterrupt:
        logger.warning("Shutdown request ... exiting")


def main(train_data_path):
    sample_index = 1
    log_file = os.path.join(train_data_path, "log", f"{sample_index}_logs.txt")
    running(sample_index, log_file, train_data_path)




if __name__ == "__main__":
    train_data_path = r"D:\Desktop\SCUC\SCUC\ifr_data"
    main(train_data_path)


