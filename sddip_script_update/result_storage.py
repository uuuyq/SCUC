import os.path
import numpy as np
import pandas as pd

class result_Dict:
    def __init__(self, name):
        """初始化一个新的字典对象"""
        self.data = {}  # 用于存储 (i, k, t) -> list
        self.name = name

    def is_empty(self):
        return self.data.__len__() == 0

    def get_keys(self):
        return self.data.keys()

    def add(self, i, k, t, value):
        """向字典中添加一个新的 key-value 对"""
        key = (i, k, t)
        if key not in self.data:
            self.data[key] = []
        self.data[key] = value

    def get_values(self, i, k, t):
        """获取指定 key (i, k, t) 的值列表"""
        return self.data.get((i, k, t), [])

    def has_key(self, i, k, t):
        """检查是否存在 key (i, k, t)"""
        return (i, k, t) in self.data

    def __repr__(self):
        """返回字典的字符串表示"""
        return repr(self.data)

    def get_values_by_t(self, t):
        """获取所有 t 值相同的元素，并合并成一个列表"""
        merged_list = []
        for (i, k, t_key), values in self.data.items():
            if t_key == t:
                merged_list.append(values)  # 合并列表
        return merged_list

    # 获取后k个Benders cut
    def get_last_k_by_t(self, t, k):
        """获取指定t的后k个数据"""
        merged_list = self.get_values_by_t(t)
        return merged_list[-k:] if len(merged_list) >= k else merged_list

    def save_cut(self, file_name, train_data_path):

        """将数据转换为 Pandas DataFrame"""
        records = []
        for (i, k, t), values in self.data.items():
            values_arr = np.array(values)  # 转换为 数组 格式
            records.append((i, k, t, values_arr))
        df = pd.DataFrame(records, columns=["i", "k", "t", "value"])
        # 按"t", "i", "k"顺序从小到大排序
        df = df.sort_values(by=["t", "i", "k"], ascending=True)


        """保存数据"""
        df.to_pickle(os.path.join(train_data_path, "cut_pkl", f"{file_name}.pkl"))
        df.to_csv(os.path.join(train_data_path, "cut_csv", f"{file_name}.csv"), index=False)




