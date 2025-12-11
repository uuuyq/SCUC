from sddip_script_update import sddipclassical
from multiprocessing import Pool
import logging
import multiprocessing
import os


# 设置日志配置
def setup_logger(log_file, process_name, sample_index):
    logger = logging.getLogger(f"{process_name}_{sample_index}")
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 将处理器添加到logger
    logger.addHandler(file_handler)


    #  # 输出到控制台
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

    return logger



def running(sample_index, log_file, cut_file_name, train_data_path):

    process_name = multiprocessing.current_process().name

    # 确保目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)


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
        train_data_path
    )
    logger.info(f"sample {sample_index+1}:")
    scenario_dir = os.path.join(train_data_path, "scenarios", f"{sample_index+1}_scenario.csv")
    # 重新初始化参数problem_params、存储字典等
    algo.init(scenario_dir)

    algo.run(cut_file_name)


    # try:
    #     algo.run(cut_file_name)
    # except KeyboardInterrupt:
    #     logger.warning("Shutdown request ... exiting")


def main(train_data_path):

    path = os.path.join(train_data_path, "cut_csv")
    index_list = []
    for i in range(0, 3000):
        file = os.path.join(path, f"{i + 1}_cuts.csv")
        if not os.path.exists(file):
            index_list.append(i)
        # else:
        #     print(f"{file}已存在")
    print(f"index_list: {index_list}")
    # for sample_index in index_list:
    #     log_file = os.path.join(train_data_path, "log", f"{sample_index + 1}_logs.txt")
    #     running(sample_index, log_file, f"{sample_index + 1}_cuts", train_data_path)

    #     print(f"index: {sample_index + 1}执行完毕")

    # test
    # sample_index = 0
    # log_file = os.path.join(train_data_path, "log", f"{sample_index + 1}_logs.txt")
    # running(sample_index, log_file, f"{sample_index + 1}_cuts", train_data_path)

    pool = Pool(10)
    try:
        for sample_index in index_list:
            log_file = os.path.join(train_data_path, "log", f"{sample_index + 1}_logs.txt")
            pool.apply_async(running, args=(sample_index, log_file, f"{sample_index + 1}_cuts", train_data_path))
    except Exception as e:
        import time
        print(time.time())
        print(e)
    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":

    train_data_path = r"./data_gen_24_bus6_CV/train_data"
    print(train_data_path)
    main(train_data_path)

    # train_data_gen_b6_t24_n6_CV

