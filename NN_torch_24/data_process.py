import os
import pickle
import pandas as pd
import numpy as np
import torch

# num_data = 10000
num_cuts_stage = 24 - 1

def collect_data_update(train_data_path, tensor_path, num_data, num_cuts, additional_data=None):
    print("train_data_path", train_data_path)
    print("tensor_path", tensor_path)
    if additional_data is not None:
        cuts_processed_path = os.path.join(train_data_path, f"cut_processed_pkl_{num_cuts}_{additional_data}")
    else:
        cuts_processed_path = os.path.join(train_data_path, f"cut_processed_pkl_{num_cuts}")
    scenarios_path = os.path.join(train_data_path, "scenarios")

    if additional_data is not None:
        feat_tensor_path = os.path.join(tensor_path, f"feat_tensor_{num_data}_{num_cuts}_{additional_data}.pkl")
        scenario_tensor_path = os.path.join(tensor_path, f"scenario_tensor_{num_data}_{num_cuts}_{additional_data}.pkl")
        label_tensor_path = os.path.join(tensor_path, f"label_tensor_{num_data}_{num_cuts}_{additional_data}.pkl")
        x_tensor_path = os.path.join(tensor_path, f"x_tensor_{num_data}_{num_cuts}_{additional_data}.pkl")
    else:
        feat_tensor_path = os.path.join(tensor_path, f"feat_tensor_{num_data}_{num_cuts}.pkl")
        scenario_tensor_path = os.path.join(tensor_path, f"scenario_tensor_{num_data}_{num_cuts}.pkl")
        label_tensor_path = os.path.join(tensor_path, f"label_tensor_{num_data}_{num_cuts}.pkl")
        x_tensor_path = os.path.join(tensor_path, f"x_tensor_{num_data}_{num_cuts}.pkl")

    if os.path.exists(feat_tensor_path):
        raise Exception(f"file already exists: {feat_tensor_path}")

    feat_list = []
    label_list = []
    scenarios_list = []
    x_list = []
    for n in range(num_data):
        cur_cuts_path = os.path.join(cuts_processed_path, f"{n + 1}_cuts.pkl")
        cur_x_path = os.path.join(cuts_processed_path, f"{n + 1}_x.pkl")
        cur_parameter_path = os.path.join(scenarios_path, f"{n + 1}_parameter.csv")
        cur_scenario_path = os.path.join(scenarios_path, f"{n + 1}_scenario.csv")
        print(f"cur: {n}")
        # 当数据不存在时 continue
        if not os.path.exists(cur_cuts_path):
            print(f"文件不存在：{cur_cuts_path}")
            continue

        # pickle 保存能保证array类型不改变
        with open(cur_cuts_path, "rb") as file:
            cuts_df = pickle.load(file)
        with open(cur_x_path, "rb") as file:
            x_df = pickle.load(file)
        parameters_df = pd.read_csv(cur_parameter_path, sep="\t")
        scenarios_df = pd.read_csv(cur_scenario_path, sep="\t")
        feat_instance_list = []
        label_instance_list = []
        scenarios_instance_list = []
        x_instance_list = []
        for t in range(num_cuts_stage):
            # key: t mu sigma
            key = parameters_df[parameters_df["t"] == t + 2].values.reshape(-1).tolist()  # parameter中t从2开始  2~6
            scenario = scenarios_df[scenarios_df["t"] == t + 2].iloc[:, 3:].values.tolist()


            # value: cuts  <class 'numpy.ndarray'>
            cut_value = cuts_df[cuts_df["t"] == t]["value"].iloc[0]
            x_value = x_df[x_df["t"] == t]["value"].iloc[0]

            cut_value = np.round(cut_value, 4)  # <<< 保留四位小数
            x_value = np.round(x_value, 4)

            # 拼接上instance 索引
            key = [n + 1] + key
            feat_instance_list.append(key)
            scenarios_instance_list.append(scenario)
            label_instance_list.append(cut_value)
            x_instance_list.append(x_value)
            # print("key", key)
            # print("label", value)

        feat_list.append(feat_instance_list)
        scenarios_list.append(scenarios_instance_list)
        label_list.append(label_instance_list)
        x_list.append(x_instance_list)


    feat_tensor = torch.tensor(
        np.array(feat_list),  # 自动推断维度 [num_data, 5, 4]
        dtype=torch.float32
    )
    scenario_tensor = torch.tensor(
        np.array(scenarios_list),  # [num_data, 5, 6, 12]  num_realizations = 6
        dtype=torch.float32
    )
    x_tensor = torch.tensor(
        np.array(x_list),  # [num_data, 5, 5, 13]
        dtype=torch.float32
    )
    label_tensor = torch.tensor(
        np.array(label_list),  # 堆叠多维数组 [num_data, 5, 5, 14]
        dtype=torch.float32
    )
    print("feat_tensor.shape: {}".format(feat_tensor.shape))
    print("scenario_tensor.shape: {}".format(scenario_tensor.shape))
    print("label_tensor.shape: {}".format(label_tensor.shape))
    print("x_tensor.shape: {}".format(x_tensor.shape))


    # 6bus CV 1000 num_data
    # feat_tensor.shape: torch.Size([1000, 23, 14])
    # scenario_tensor.shape: torch.Size([1000, 23, 6, 12])
    # label_tensor.shape: torch.Size([1000, 23, 15, 14])
    # x_tensor.shape: torch.Size([1000, 23, 15, 13])

    # 118bus  4060 num_data
    # feat_tensor.shape: torch.Size([4053, 23, 238])
    # scenario_tensor.shape: torch.Size([4053, 23, 6, 236])
    # label_tensor.shape: torch.Size([4053, 23, 15, 379])
    # x_tensor.shape: torch.Size([4053, 23, 15, 378])



    with open(feat_tensor_path, "wb") as f:
        pickle.dump(feat_tensor, f)
    with open(scenario_tensor_path, "wb") as f:
        pickle.dump(scenario_tensor, f)
    with open(label_tensor_path, "wb") as f:
        pickle.dump(label_tensor, f)
    with open(x_tensor_path, "wb") as f:
        pickle.dump(x_tensor, f)

if __name__ == '__main__':
    train_data_path = r"../data_gen_24_bus6_CV\train_data"
    tensor_path = r"tensor_6_CV"
    num_data = 1000
    collect_data_update(train_data_path, tensor_path, num_data, 15, additional_data="prefix")
