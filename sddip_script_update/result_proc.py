import os.path

import numpy as np
import pandas as pd
import pathlib
import pickle

def data_process_x(df, x_df, num_cuts):
    # 0. 确保两个DataFrame基于相同顺序 (i, k, t)
    df = df.sort_values(by=["i", "k", "t"]).reset_index(drop=True)
    x_df = x_df.sort_values(by=["i", "k", "t"]).reset_index(drop=True)

    # 验证行数相同
    assert len(df) == len(x_df), "df and x_df must have same number of rows after sorting"

    # 1. 保留5位小数
    df["value"] = df["value"].apply(lambda arr: np.round(arr, 5))
    x_df["value"] = x_df["value"].apply(lambda arr: np.round(arr, 5))

    # 2. 去掉i,k列
    df = df.drop(columns=["i", "k"])
    x_df = x_df.drop(columns=["i", "k"])

    # 3. 定义MSE函数
    def mse(a, b):
        return np.mean((a - b) ** 2) if len(a) == len(b) else float('inf')

    # 4. 创建合并DataFrame处理同步去重
    combined = pd.concat([df, x_df], axis=1, keys=['df', 'x'])
    combined.columns = combined.columns.map('_'.join)

    # 5. 分组处理函数
    def process_group(group):
        # 分离df和x数据
        df_values = group['df_value'].tolist()
        x_values = group['x_value'].tolist()

        # 基于df value去重
        unique_df = []
        unique_x = []
        for d, x in zip(df_values, x_values):
            if all(mse(d, existing) > 1e-6 for existing in unique_df):
                unique_df.append(d)
                unique_x.append(x)

        return pd.DataFrame({
            't': [group['df_t'].iloc[0]],
            'df_value': [unique_df],
            'x_value': [unique_x]
        })

    # 6. 分组处理
    processed = combined.groupby('df_t', group_keys=False).apply(process_group)

    # 7. 分离结果
    df_processed = processed[['t', 'df_value']].rename(columns={'df_value': 'value'})
    x_processed = processed[['t', 'x_value']].rename(columns={'x_value': 'value'})

    # 8. 填充/截断函数
    # 取后num_cuts个
    # def pad_or_trim(arr_list):
    #     if len(arr_list) >= num_cuts:
    #         return np.array(arr_list[-num_cuts:])
    #     else:
    #         last = arr_list[-1] if arr_list else np.zeros(1)  # 处理空列表情况
    #         return np.array(arr_list + [last] * (num_cuts - len(arr_list)))
    # 取前num_cuts个
    def pad_or_trim(arr_list):
        if len(arr_list) >= num_cuts:
            # 取前 num_cuts 个
            return np.array(arr_list[:num_cuts])
        else:
            first = arr_list[0] if arr_list else 0.0  # 处理空列表
            pad_len = num_cuts - len(arr_list)
            # 在前面补足
            return np.array([first] * pad_len + arr_list)

    # 9. 应用填充/截断
    df_processed['value'] = df_processed['value'].apply(pad_or_trim)
    x_processed['value'] = x_processed['value'].apply(pad_or_trim)

    return df_processed.reset_index(drop=True), x_processed.reset_index(drop=True)


def proc(num_cuts, train_data_path, force_recalculate_flag=False):

    path = train_data_path / "cut_pkl"
    pkl_file_name = f"cut_processed_pkl_15_prefix"
    csv_file_name = f"cut_processed_csv_15_prefix"

    for i in range(2800, 4060):
        cut_file = path / f"{i + 1}_cuts.pkl"
        x_file = path / f"{i + 1}_x.pkl"

        if not pathlib.Path.exists(cut_file) or not pathlib.Path.exists(x_file):
            print(f"文件不存在: {cut_file} or {x_file}")
            continue

        # 检查文件是否存在及权限
        # print("存在:", os.path.exists(cut_file))
        # print("可读:", os.access(cut_file, os.R_OK))
        # print("可写:", os.access(cut_file, os.W_OK))
        # print("文件大小:", os.path.getsize(cut_file) if os.path.exists(cut_file) else "不存在")

        # with open(cut_file, "br") as f:
        #     cut_df = pickle.load(f)
        # with open(x_file, "br") as f:
        #     x_df = pickle.load(f)
        try:
            with open(cut_file, "rb") as f:
                cut_df = pickle.load(f)
        except PermissionError:
            print(f"[警告] 文件被占用或无权限: {cut_file}")
            continue
        except Exception as e:
            print(f"[错误] 无法读取 {cut_file}: {e}")
            continue
        try:
            with open(x_file, "br") as f:
                x_df = pickle.load(f)
        except PermissionError:
            print(f"[警告] 文件被占用或无权限: {x_file}")
            continue
        except Exception as e:
            print(f"[错误] 无法读取 {x_file}: {e}")
            continue



        if pathlib.Path.exists(train_data_path / pkl_file_name / f"{i+1}_cuts.pkl") and not force_recalculate_flag:
            print(f"文件已处理: {train_data_path / pkl_file_name / f'{i+1}_cuts.pkl'}")
            continue
        else:
            cut_df_processed, x_df_preocessed = data_process_x(cut_df, x_df, num_cuts)

            cut_df_processed.to_pickle(train_data_path / pkl_file_name / f"{i+1}_cuts.pkl")
            cut_df_processed.to_csv(train_data_path / csv_file_name / f"{i+1}_cuts.csv", index=False)
            x_df_preocessed.to_pickle(train_data_path / pkl_file_name / f"{i+1}_x.pkl")
            x_df_preocessed.to_csv(train_data_path / csv_file_name / f"{i+1}_x.csv", index=False)

            # finally:
            #     print(f"file error: {train_data_path / pkl_file_name / f'{i + 1}_cuts.pkl'}")



if __name__ == "__main__":
    """
    收集指定数量的cuts和x
    """
    num_cuts = 15  # 指定cuts数量
    train_data_path = pathlib.Path(r"../data_gen_24_bus118/train_data")  # 原始cut地址

    proc(num_cuts, train_data_path)






