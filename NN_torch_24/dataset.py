import torch
from torch.utils.data import Dataset
import pickle
import os
class CutDataset(Dataset):

    def __init__(self, tensor_path, num_data, num_cuts, additional_data=None):
        if additional_data is None:
            feat_path = os.path.join(tensor_path, f"feat_tensor_{num_data}_{num_cuts}.pkl")
            scenario_path = os.path.join(tensor_path, f"scenario_tensor_{num_data}_{num_cuts}.pkl")
            cut_path = os.path.join(tensor_path, f"label_tensor_{num_data}_{num_cuts}.pkl")
            x_path = os.path.join(tensor_path, f"x_tensor_{num_data}_{num_cuts}.pkl")
        else:
            feat_path = os.path.join(tensor_path, f"feat_tensor_{num_data}_{num_cuts}_{additional_data}.pkl")
            scenario_path = os.path.join(tensor_path, f"scenario_tensor_{num_data}_{num_cuts}_{additional_data}.pkl")
            cut_path = os.path.join(tensor_path, f"label_tensor_{num_data}_{num_cuts}_{additional_data}.pkl")
            x_path = os.path.join(tensor_path, f"x_tensor_{num_data}_{num_cuts}_{additional_data}.pkl")
        print(f"feat_path: {feat_path}")
        with open(feat_path, "rb") as f:
            self.feat_tensor = pickle.load(f)
        with open(scenario_path, "rb") as f:
            self.scenario_tensor = pickle.load(f)
        with open(x_path, "rb") as f:
            self.x_tensor = pickle.load(f)
        with open(cut_path, "rb") as f:
            self.cut_tensor = pickle.load(f)

    def __getitem__(self, idx):
        '''
        data_set.feat_tensor.shape torch.Size([5000, 23, 14])  [num_data, num_stage, -]
        data_set.cut_tensor.shape torch.Size([5000, 23, 15, 14])
        data_set.scenario_tensor.shape torch.Size([5000, 23, 6, 12]) [num_data, num_stage, num_realizations, -]
        data_set.x_tensor.shape torch.Size([5000, 23, 15, 13])
        '''
        return self.feat_tensor[idx], self.scenario_tensor[idx], self.cut_tensor[idx], self.x_tensor[idx]  # 返回一个instance的数据

    def __len__(self):
        return self.feat_tensor.shape[0]

class CutDatasetNormalized(Dataset):
    def __init__(self, feat_tensor, scenario_tensor, x_tensor, cut_tensor):
        self.feat_tensor = feat_tensor
        self.scenario_tensor = scenario_tensor
        self.x_tensor = x_tensor
        self.cut_tensor = cut_tensor

    def __getitem__(self, idx):
        return self.feat_tensor[idx], self.scenario_tensor[idx], self.cut_tensor[idx], self.x_tensor[idx]

    def __len__(self):
        return self.feat_tensor.shape[0]


# class StandardScaler:
#     def __init__(self):
#         self.mean = None
#         self.std = None
#
#     def fit(self, data):
#         # print("fit:", data.shape)
#         # 前13列统一处理
#         flattened = data.flatten()
#
#         flattened = torch.abs(flattened)
#
#         self.mean = torch.mean(flattened)
#         self.std = torch.std(flattened)
#         # 前13列也分别标准化
#         # data = torch.reshape(data, [-1, data.shape[-1]])
#         # self.mean = torch.mean(data, dim=0)
#         # self.std = torch.std(data, dim=0)
#         # 防止除零错误
#         self.std[self.std == 0] = 1.0
#
#     def transform(self, data):
#         return (data - self.mean) / self.std
#
#     def inverse_transform(self, data):
#         return (data * self.std) + self.mean

# ===== 自定义标准化器 =====
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        self.dim = None

    def fit(self, data: torch.Tensor, dim: int):
        # data shape: (N, ..., D) → reshape to (-1, D)
        self.dim = dim
        data = data[..., :self.dim]
        data = data.reshape(-1, data.shape[-1])
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0, unbiased=False)
        # 防止除零
        self.std[self.std == 0] = 1.0
        print("mean: ", self.mean)
        print("std: ", self.std)
        print("mean.shape: ", self.mean.shape)  # torch.Size([14])
        print("std.shape: ", self.std.shape)  # torch.Size([14])


    def transform(self, data: torch.Tensor):
        # 拆分最后一维
        data_front = data[..., :self.dim]
        data_rest = data[..., self.dim:]
        # 标准化前 dim 部分
        data_front = (data_front - self.mean) / self.std
        # 拼接处理后与未处理的部分
        return torch.cat([data_front, data_rest], dim=-1)

    def inverse_transform(self, data: torch.Tensor):
        if data.shape[-1] > self.dim:
            # 拆分最后一维
            data_front = data[..., :self.dim]
            data_rest = data[..., self.dim:]
            # 标准化前 dim 部分
            data_front = data_front * self.std + self.mean
            # 拼接处理后与未处理的部分
            return torch.cat([data_front, data_rest], dim=-1)
        return data * self.std + self.mean

# ===== 包装 Dataset，标准化 label =====
class ProcessedDataset(Dataset):
    def __init__(self, base_dataset, scaler):
        self.dataset = base_dataset
        self.scaler = scaler


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        y_scaled = self.scaler.transform(y)
        return x, y_scaled


# class ProcessedDataset(Dataset):
#     def __init__(self, dataset, scaler_part1, scaler_part2):
#         self.dataset = dataset
#         self.scaler_part1 = scaler_part1  # 前13列的标准化器
#         self.scaler_part2 = scaler_part2  # 最后一列的标准化器
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         feat, label = self.dataset[idx]
#         # 分割标签
#         part1 = label[..., :13]  # 前13列
#         part2 = label[..., -1:]  # 最后一列
#         # 应用标准化
#         part1 = self.scaler_part1.transform(part1)
#         part2 = self.scaler_part2.transform(part2)
#         # print("part1", part1.shape)
#         # print("part2", part2.shape)
#         # 合并处理后的标签
#         processed_label = torch.cat([part1, part2], dim=-1)
#         # print("processed_label", processed_label.shape)
#         return feat, processed_label

# if __name__ == '__main__':
#     data = CutDataset(20000)
#     feat, label = data.__getitem__(1)
#     print("feat:", feat[0])
#     print("label:", label[0][0][4].item())
#     print("label:", label[0][0])