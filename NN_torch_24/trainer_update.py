import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
import time
from torch.utils.data import Subset
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from NN_torch_24.config import Config
from NN_torch_24.dataset import CutDataset, CutDatasetNormalized
from NN_torch_24.model import NN_Model
from NN_torch_24.infer import Infer
from NN_torch_24.constant import CompareConstant
import logging

class TrainerUpdate:
    """
    输入： x + 分布参数 / scenarios
    输出：cuts
    loss：mse(Q) + gamma*mse(cuts)
    """
    def __init__(self, config: Config,
                 ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params_name = f"{self.config.num_data}_{self.config.n_pieces}_lr-{self.config.LEARNING_RATE}_wd-{self.config.weight_decay}_gamma-{self.config.gamma}_dim-{self.config.hidden_arr}_standard-{self.config.standard_flag}"


    def load_dataset(self):
        print("load_dataset ...")
        # 数据集
        data_set = CutDataset(self.config.tensor_data_path, self.config.num_data, self.config.n_pieces, additional_data=self.config.additional_data)

        print("data_set.feat_tensor.shape", data_set.feat_tensor.shape)
        print("data_set.cut_tensor.shape", data_set.cut_tensor.shape)
        print("data_set.scenario_tensor.shape", data_set.scenario_tensor.shape)
        print("data_set.x_tensor.shape", data_set.x_tensor.shape)

        train_size = int(0.7 * len(data_set))
        val_size = int(0.15 * len(data_set))
        test_size = len(data_set) - train_size - val_size

        # 使用 random_split 方法进行划分
        generator1 = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(data_set, [train_size, val_size, test_size], generator=generator1)

        if self.config.standard_flag:
            #  label_tensor.shape: torch.Size([4053, 23, 15, 379])
            train_indices = self.train_dataset.indices  # Subset 的索引
            train_cut = data_set.cut_tensor[train_indices]  # [num_train, 23, 15, 379]

            # 计算均值和标准差，只按最后一维计算
            # shape = [1,1,1,379] 用于广播
            self.cut_mean = train_cut.mean(dim=(0, 1, 2))
            self.cut_std = train_cut.std(dim=(0, 1, 2)) + 1e-8  # 防止除零

            # 标准化训练集
            train_cut_norm = (train_cut - self.cut_mean) / self.cut_std
            # 标准化验证集
            val_cut = data_set.cut_tensor[self.val_dataset.indices]
            val_cut_norm = (val_cut - self.cut_mean) / self.cut_std

            # 封装为 Dataset 对象
            self.train_dataset = CutDatasetNormalized(
                data_set.feat_tensor[self.train_dataset.indices],
                data_set.scenario_tensor[self.train_dataset.indices],
                data_set.x_tensor[self.train_dataset.indices],
                train_cut_norm
            )

            self.val_dataset = CutDatasetNormalized(
                data_set.feat_tensor[self.val_dataset.indices],
                data_set.scenario_tensor[self.val_dataset.indices],
                data_set.x_tensor[self.val_dataset.indices],
                val_cut_norm
            )
        print(f"load_dataset done, standard_flag: {self.config.standard_flag}")
        if self.config.standard_flag:
            print("cut_mean: ", self.cut_mean)
            print("cut_std: ", self.cut_std)


        # print(len(data_set))
        # print(len(self.train_dataset))
        # print(len(self.val_dataset))
        # print(len(self.test_dataset))

    def train(self):


        loss_path = os.path.join(self.config.result_path, "model", f"loss_{self.params_name}.pkl")
        model_path = os.path.join(self.config.result_path, "model", f"model_{self.params_name}.pth")
        if os.path.exists(loss_path):
            print(f"model 存在，load model: {model_path}")
            self._load_model()
            return

        model = NN_Model(self.config.num_stage, self.config.hidden_arr, self.config.n_vars, self.config.n_pieces)
        model = model.to(self.device)
        train_data_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0, drop_last=False)
        val_data_loader = DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0, drop_last=False)
        # 损失函数
        # criterion = ApproxEMDLoss()
        # MSE
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.weight_decay  # L2 正则化系数
        )
        train_loss_history = []
        val_loss_history = []
        train_loss_Q_list = []
        train_loss_cuts_list = []
        best_val_loss = float('inf')

        if self.config.standard_flag:
            self.cut_std = self.cut_std.to(self.device)
            self.cut_mean = self.cut_mean.to(self.device)

        for epoch in range(self.config.N_EPOCHS):

            # 训练模式
            model.train()
            train_loss = 0.0
            train_loss_Q = 0.0
            train_loss_cuts = 0.0
            for feats, scenarios, cuts, x in train_data_loader:  # feats.shape[batch_size, 5, 3] labels.shape[batch_size, 5, 5, 14]

                feats = torch.reshape(feats, [-1, self.config.feat_dim])
                # scenarios = torch.reshape(scenarios, [-1, self.num_scenarios, 12])
                cuts = torch.reshape(cuts, [-1, self.config.n_pieces, self.config.single_cut_dim])
                x = torch.reshape(x, [-1, self.config.n_pieces, self.config.n_vars])

                # 数据转移到设备
                feats = feats.to(self.device)
                cuts = cuts.to(self.device)
                scenarios = scenarios.to(self.device)
                x = x.to(self.device)


                pred_cuts = model(feats, x, x_input_flag=self.config.x_input_flag)
                loss_cuts = criterion(pred_cuts, cuts)

                if self.config.standard_flag:
                    cuts = cuts * self.cut_std + self.cut_mean
                    pred_cuts = pred_cuts * self.cut_std + self.cut_mean

                labels = (x * cuts[:, :, :x.size(-1)]).sum(dim=-1, keepdim=True) + cuts[:, :, -1:]
                Q = (x * pred_cuts[:, :, :x.size(-1)]).sum(dim=-1, keepdims=True) + pred_cuts[:, :, -1:]
                loss_Q = criterion(Q, labels)

                loss = loss_Q + self.config.gamma * loss_cuts

                train_loss_Q += loss_Q.item() * feats.size(0)
                train_loss_cuts += loss_cuts.item() * feats.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * feats.size(0)
            train_loss = train_loss / len(self.train_dataset)
            train_loss_Q = train_loss_Q / len(self.train_dataset)
            train_loss_cuts = train_loss_cuts / len(self.train_dataset)
            train_loss_history.append(train_loss)
            train_loss_Q_list.append(train_loss_Q)
            train_loss_cuts_list.append(train_loss_cuts)
            # 验证模式
            model.eval()
            val_loss = 0.0
            val_loss_Q = 0.0
            val_loss_cuts = 0.0
            with torch.no_grad():
                for feats, scenarios, cuts, x in val_data_loader:

                    feats = torch.reshape(feats, [-1, self.config.feat_dim])
                    # 暂时不考虑使用scenarios
                    # scenarios = torch.reshape(scenarios, [-1, self.num_scenarios, 12])
                    cuts = torch.reshape(cuts, [-1, self.config.n_pieces, self.config.single_cut_dim])
                    x = torch.reshape(x, [-1, self.config.n_pieces, self.config.n_vars])

                    # 数据转移到设备
                    feats = feats.to(self.device)
                    cuts = cuts.to(self.device)
                    scenarios = scenarios.to(self.device)
                    x = x.to(self.device)

                    pred_cuts = model(feats, x, x_input_flag=self.config.x_input_flag)
                    loss_cuts = criterion(pred_cuts, cuts)

                    if self.config.standard_flag:
                        cuts = cuts * self.cut_std + self.cut_mean
                        pred_cuts = pred_cuts * self.cut_std + self.cut_mean

                    labels = (x * cuts[:, :, :x.size(-1)]).sum(dim=-1, keepdim=True) + cuts[:, :, -1:]
                    Q = (x * pred_cuts[:, :, :x.size(-1)]).sum(dim=-1, keepdims=True) + pred_cuts[:, :, -1:]
                    loss_Q = criterion(Q, labels)

                    loss = loss_Q + self.config.gamma * loss_cuts

                    val_loss_Q += loss_Q.item() * feats.size(0)
                    val_loss_cuts += loss_cuts.item() * feats.size(0)
                    val_loss += loss.item() * feats.size(0)

            val_loss = val_loss / len(self.val_dataset)
            val_loss_history.append(val_loss)

            val_loss_Q = val_loss_Q / len(self.val_dataset)
            val_loss_cuts = val_loss_cuts / len(self.val_dataset)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(self.config.result_path, "model", f"model_{self.params_name}.pth"))

            # 打印进度
            print(f"Epoch [{epoch + 1}/{self.config.N_EPOCHS}] | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            print(f"loss_Q: {train_loss_Q:.4f}  loss_cuts: {train_loss_cuts:.4f}")
            print(f"loss_Q: {val_loss_Q:.4f}  loss_cuts: {val_loss_cuts:.4f}")

        # 绘图 loss
        plt.figure(figsize=(8, 5))
        plt.plot(train_loss_history, label='Train Loss', color='blue')
        plt.plot(val_loss_history, label='Validation Loss', color='orange')
        # plt.plot(train_loss_Q_list, label='train_loss_Q_list', color='g')
        # plt.plot(train_loss_cuts_list, label='train_loss_cuts_list', color='y')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Train Validation Loss with gamma={self.config.gamma}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        save_path = os.path.join(self.config.result_path, "model")
        filepath = os.path.join(save_path, f"loss_{self.params_name}.png")
        plt.savefig(filepath)  # 保存为文件
        plt.close()  # 关闭窗口，避免重复显示（尤其在循环中）

        # 保存为 CSV
        loss_path = os.path.join(save_path, f"loss_{self.params_name}.pkl")
        loss_df = {
            "train_loss": train_loss_history,
            "val_loss": val_loss_history
        }
        torch.save(loss_df, loss_path)
        print(f"Train/Val loss saved to pkl: {loss_path}")

        print(f"Plot saved to: {filepath}")


    def _load_model(self):
        model = NN_Model(self.config.num_stage, self.config.hidden_arr, self.config.n_vars, self.config.n_pieces)
        model_path = os.path.join(self.config.result_path, "model", f"model_{self.params_name}.pth")
        print(f"Load model from: {model_path}")
        model.load_state_dict(torch.load(model_path))
        model = model.to(self.device)
        return model

    def _predict_single(self, model, inputs):
        """
        预测单个样本
        :param model: NN_model
        :param inputs: (feat, x)
        :return:  pred_cuts: [num_stages-1, n_pieces, N_VARS+1]
        """
        model.eval()
        with torch.no_grad():

            feats, x = inputs
            # 转换成tensor才能进行预测
            feats = torch.tensor(feats, dtype=torch.float32)
            x = torch.tensor(x, dtype=torch.float32)

            feats = torch.reshape(feats, [-1, self.config.feat_dim])
            x = torch.reshape(x, [-1, self.config.n_pieces, self.config.n_vars])
            # 数据转移到设备
            feats = feats.to(self.device)
            x = x.to(self.device)
            self.cut_std = self.cut_std.to(self.device)
            self.cut_mean = self.cut_mean.to(self.device)


            pred_cuts = model(feats, x, x_input_flag=self.config.x_input_flag)

            if self.config.standard_flag:
                pred_cuts = pred_cuts * self.cut_std + self.cut_mean

        return pred_cuts

    def _calculate_x(self, feat):
        '''
        :param feat: [n_stages-1, n_feats]
        :return: x_array [n_stages-1, n_pieces, N_VARS+1]
        '''

        inference_sddip = Infer(n_stages=self.config.num_stage,
                                n_realizations=self.config.n_realizations,
                                N_VARS=self.config.n_vars,
                                train_data_path=self.config.train_data_path,
                                result_path=self.config.result_path
                                )
        x_array = inference_sddip.calculate_x(feat, self.config.n_pieces)
        return x_array

    def sample_test_dataset(self, num_instances):
        """
        选取test_data中的部分数据
        """

        save_path = os.path.join(self.config.compare_path, f"num-{num_instances}_sampled_fixed.pkl")
        if os.path.exists(save_path):
            print(f"{save_path} 已存在，直接返回")
            with open(save_path, "rb") as f:
                data_sampled = torch.load(f)
            return data_sampled

        print("sample_test_dataset start...")
        # 随机选取dataset中的部分数据
        import random
        random.seed(43)

        test_dataset = self.test_dataset
        total = len(test_dataset)

        # 找到 instance_index < 3000 的所有可选索引  下面限制的数据才有训练时的obj和time
        eligible_indices = []
        for idx in range(total):
            feat, scenario, cut, x = test_dataset[idx]
            instance_index = feat[0][0].item() if isinstance(feat[0][0], torch.Tensor) else feat[0][0]
            if instance_index <= 1000 or instance_index > 1500 and instance_index <= 3000:
                eligible_indices.append(idx)


        num_samples = min(num_instances, len(eligible_indices))

        # 抽取
        sampled_indices = random.sample(eligible_indices, num_samples)
        test_dataset_sampled = Subset(test_dataset, sampled_indices)
        print("test_dataset_sampled", len(test_dataset_sampled))
        # 将数据合并到一个dict中

        data_sampled = []  # ← 从 dict[list] 改为 list[dict]

        for i in range(len(test_dataset_sampled)):
            feat, scenario, cut, x = test_dataset_sampled[i]

            # 保证 在 CPU，可以安全传多进程
            feat = feat.detach().cpu().numpy().copy()
            scenario = scenario.detach().cpu().numpy().copy()
            cut = cut.detach().cpu().numpy().copy()
            x = x.detach().cpu().numpy().copy()

            instance_index = feat[0][0]

            # 组装一个 dict，加入 list
            data_sampled.append({
                CompareConstant.instance_index: instance_index,
                CompareConstant.feat: feat,
                CompareConstant.scenario: scenario,
                CompareConstant.cuts_train: cut,
                CompareConstant.x_train: x,
            })

        torch.save(data_sampled, save_path)

        return data_sampled

    def sampled_read(self, data_sampled):
        """
        sddip太慢了，简单读取生成数据时的obj和time看一下
        :return:
        """
        num_instances = len(data_sampled)
        save_path = os.path.join(self.config.compare_path, f"num-{num_instances}_sampled_read.pkl")
        if os.path.exists(save_path):
            print(f"{save_path} 已存在，直接返回")
            with open(save_path, "rb") as f:
                data_sampled = torch.load(f)
            return data_sampled

        # 读取生成数据是的obj和time
        time_path = r"D:\tools\workspace_pycharm\sddip-SCUC-6-24\data_gen_24_bus118\train_data\time.txt"
        obj_path_root = r"D:\tools\workspace_pycharm\sddip-SCUC-6-24\data_gen_24_bus118\train_data\LB_obj_list"

        # 读取 time 文件，存到字典
        time_dict = {}
        with open(time_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # line 格式: "1_cuts : 10329.487878084183 seconds"
                parts = line.split(":")
                index_str = parts[0].replace("_cuts", "").strip()
                time_val = float(parts[1].replace("seconds", "").strip())
                time_dict[int(index_str)] = time_val

        # 存储结果
        for data_instance in data_sampled:
            feat = data_instance[CompareConstant.feat]
            instance_index = int(feat[0][0])

            # 读取 obj 文件
            obj_path = os.path.join(obj_path_root, f"{instance_index}_cuts_obj.txt")
            with open(obj_path, "r") as f_obj:
                lines = f_obj.readlines()
                last_obj = float(lines[-1].strip())  # 取最后一个数

            data_instance[CompareConstant.obj_sddip_read] = last_obj

            # 取对应时间
            if instance_index in time_dict:
                data_instance[CompareConstant.time_sddip_read] = time_dict[instance_index]
            else:
                data_instance[CompareConstant.time_sddip_read] = None  # 若 time 文件中没有对应 index

        torch.save(data_sampled, save_path)

        return data_sampled

    def _get_pred_cuts(self, data_sampled):
        """
        获取pred_cuts，记录推理耗时
        :return:
        """
        num_instances = len(data_sampled)

        save_path = os.path.join(self.config.compare_path,
                                 f"num-{num_instances}_data_sampled_sddip_pred.pkl")
        if os.path.exists(save_path):
            print(f"{save_path} 已存在，直接返回")
            with open(save_path, "rb") as f:
                data_sampled = torch.load(f)
            return data_sampled


        model = self._load_model()
        print(f" predict start  device: {self.device}")
        for i in tqdm(range(num_instances), desc="pred cuts processing"):
            sample = data_sampled[i]

            feat = sample[CompareConstant.feat]

            start = time.time()
            # 没有cut计算x
            x_array = self._calculate_x(feat)
            # 预测 cuts
            pred_cuts = self._predict_single(model, (feat, x_array))
            end = time.time()

            # ✔ 写回当前 dict
            sample[CompareConstant.time_pred] = end - start
            sample[CompareConstant.cuts_pred] = pred_cuts.detach().cpu().numpy().copy()  # 转换成array
            sample[CompareConstant.x_pred] = x_array

        torch.save(data_sampled, save_path)
        return data_sampled


    def sampled_sddip(self, num_instances, sddip_fw_n_samples, max_iterations, sddip_timeout_sec, num_threads):
        # 创建保存结果的文件夹
        if self.config.compare_path is None:
            raise Exception("compare_path is None 需要设置compare_path为测试结果保存的位置")
        os.makedirs(self.config.compare_path, exist_ok=True)

        save_path = os.path.join(self.config.compare_path, f"num-{num_instances}_data_sampled_sddip_fw-{sddip_fw_n_samples}_iter-{max_iterations}.pkl")
        if os.path.exists(save_path):
            print(f"{save_path} 已存在，直接返回")
            with open(save_path, "rb") as f:
                data_sampled = torch.load(f)
            return data_sampled

        # 内部判断是否已经存在
        data_sampled = self.sample_test_dataset(num_instances)

        num_workers = min(mp.cpu_count(), num_threads)  # 限制最大并行数，防止CPU爆满
        # sddip
        print(f"sddip: using {num_workers} processes for parallel...")
        func = partial(_process_sddip,
                       sddip_fw_n_samples=sddip_fw_n_samples,
                       config=self.config,
                       max_iterations=max_iterations,
                       log_path=self.config.compare_path
                       )
        data_sampled = multi_process(func, num_workers, data_sampled, sddip_timeout_sec)
        print("Parallel sddip complete.")
        # 保存原始结果
        torch.save(data_sampled, save_path)

    def compare_obj_multiprocess(self, data_sampled_sddip, obj_fw_n_samples, max_lag, compare_timeout_sec, num_threads):
        """
        使用多进程方式比较时间和obj。
        注意：如果CPU紧张，pred或re的耗时可能会略有上升。
        pred
        开始多进程的方式
        re
        obj
        """
        data_sampled = self._get_pred_cuts(data_sampled_sddip)
        print("#########pred finish#########")

        num_workers = min(mp.cpu_count(), num_threads)  # 限制最大并行数，防止CPU爆满

        print(f"compare: using {num_workers} processes for parallel...")
        func = partial(_process_obj,
                       obj_fw_n_samples=obj_fw_n_samples,
                       max_lag=max_lag,
                       config=self.config,
                       log_path=self.config.compare_path)
        compare_obj_result = multi_process(func, num_workers, data_sampled, compare_timeout_sec)

        # 保存原始结果
        torch.save(compare_obj_result, os.path.join(self.config.compare_path,
                                        f"num-{len(data_sampled_sddip)}_compare_obj_result_fw-{obj_fw_n_samples}_max_lag-{max_lag}.pkl"))
        print("Parallel comparison complete.")
        return compare_obj_result

    def compare_LB_multiprocess(self, data_sampled_sddip, num_instances, sddip_fw_n_samples, max_iterations, compare_timeout_sec, num_threads):
        """
        比较LB
        """
        num_workers = min(mp.cpu_count(), num_threads)  # 限制最大并行数，防止CPU爆满
        data_sampled_sddip = data_sampled_sddip[:num_instances]
        print(f"compare: using {num_workers} processes for parallel...")
        func = partial(_process_LB,
                       sddip_fw_n_samples=sddip_fw_n_samples,
                       max_iterations=max_iterations,
                       config=self.config,
                       log_path=self.config.compare_path)

        compare_LB_result = multi_process(func, num_workers, data_sampled_sddip, compare_timeout_sec)
        # 保存原始结果
        torch.save(compare_LB_result, os.path.join(self.config.compare_path,
                                                    f"num-{len(data_sampled_sddip)}_compare_LB_result_fw-{sddip_fw_n_samples}_max_iter-{max_iterations}.pkl"))
        print("Parallel comparison complete.")
        return compare_LB_result

    def compare_quick(self, num_instances, fw_n_samples, max_lag,
                         compare_timeout_sec, num_threads):
        """
        使用多进程方式比较时间和obj。
        注意：如果CPU紧张，pred或re的耗时可能会略有上升。
        """
        # 创建保存结果的文件夹
        if self.config.compare_path is None:
            raise Exception("compare_path is None 需要设置compare_path为测试结果保存的位置")

        os.makedirs(self.config.compare_path, exist_ok=True)

        # 内部判断是否已经存在
        data_sampled = self.sample_test_dataset(num_instances)
        data_sampled = self.sampled_read(data_sampled)
        data_sampled = self._get_pred_cuts(data_sampled)

        print("输出data_sampled：")
        print(data_sampled)


        num_workers = min(mp.cpu_count(), num_threads)  # 限制最大并行数，防止CPU爆满
        print(f"compare: using {num_workers} processes for parallel...")
        func = partial(_process_instance_quick,
                       fw_n_samples=fw_n_samples,
                       max_lag=max_lag,
                       config=self.config,
                       log_path=self.config.compare_path)

        compare_result = multi_process(func, num_workers, data_sampled, compare_timeout_sec)

        # 保存原始结果
        torch.save(compare_result, os.path.join(self.config.compare_path,
                                                f"num-{num_instances}_compare_result_fw-{fw_n_samples}.pkl"))
        print("Parallel comparison complete.")
        return compare_result



def multi_process(func, num_workers, data_sampled, timeout_sec=None):

    result = []
    with mp.Pool(processes=num_workers) as pool:
        async_results = [pool.apply_async(func, args=(i, data_sampled[i])) for i in range(len(data_sampled))]

        for i, async_result in enumerate(async_results):
            try:
                if timeout_sec is None:
                    res = async_result.get()
                else:
                    res = async_result.get(timeout=timeout_sec)
                if res is not None:  # 只保存有效结果
                    result.append(res)
            except mp.TimeoutError:
                print(f"⚠️ Instance {i} timed out after {timeout_sec} seconds!")
            except Exception as e:
                print(f"❌ Instance {i} failed with error: {e}")
    return result



def _process_sddip(instance_index, instance_dict, sddip_fw_n_samples, config, max_iterations, log_path):
    # === 每个进程单独日志 ===
    log_dir = os.path.join(log_path, "sddip_logs")  # 日志保存位置
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"process_{instance_index}.log")
    # 清空旧日志处理器
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s][Process {instance_index}][%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            # logging.StreamHandler()  # 若希望仍打印到控制台可加这一行
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Process {instance_index} start")

    # === 正式计算 ===
    inference_sddip = Infer(
        n_stages=config.num_stage,
        n_realizations=config.n_realizations,
        N_VARS=config.n_vars,
        train_data_path=config.train_data_path,
        result_path=config.result_path
    )

    feat = instance_dict[CompareConstant.feat]
    logger.info(f"feat: {feat}")
    time_list, obj_list, LB_list = calculate_obj_with_sddip_iteration(
        inference_sddip=inference_sddip,
        feat=feat,
        cuts=None,
        fw_n_samples=sddip_fw_n_samples,
        additional_time=0,  # sddip没有额外的时间
        max_iterations=max_iterations,
        logger=logger,
    )

    instance_dict[CompareConstant.time_sddip] = time_list
    instance_dict[CompareConstant.obj_sddip] = obj_list
    instance_dict[CompareConstant.LB_sddip] = LB_list

    save_path = os.path.join(log_path, "sddip_result") 
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{instance_index}_result.pkl")
    torch.save(instance_dict, save_file)  # 分开保存

    return instance_dict



def _process_obj(instance_index, instance_dict, obj_fw_n_samples, max_lag, config, log_path):
    """
    单个样本的处理逻辑
    计算这几种的obj
    nocut pred pred_re pred_sddip pred_re_sddip sddip收敛
    """

    # === 每个进程单独日志 ===
    log_dir = os.path.join(log_path, "compare_obj_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"process_{instance_index}.log")
    # 清空旧日志处理器
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s][Process {instance_index}][%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            # logging.StreamHandler()  # 若希望仍打印到控制台可加这一行
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Process {instance_index} start")

    # === 正式计算 ===
    inference_sddip = Infer(
        n_stages=config.num_stage,
        n_realizations=config.n_realizations,
        N_VARS=config.n_vars,
        train_data_path=config.train_data_path,
        result_path=config.result_path
    )

    feat = instance_dict[CompareConstant.feat]
    logger.info(f"feat: {feat}")
    cuts_pred = instance_dict[CompareConstant.cuts_pred]
    x_pred = instance_dict[CompareConstant.x_pred]
    logger.info(f"cuts_pred.shape: {cuts_pred.shape}")

    logger.info("############Start recalculate cuts################")
    logger.info("re需要的参数：")
    logger.info(f"feat.device: {feat.device}")
    logger.info(f"cuts_pred.device: {cuts_pred.device}")
    logger.info(f"x_pred.device: {x_pred.device}")
    # 直接在dual model求解中限制求解时间和精度
    cuts_pred_re, recalculate_time = _recalculate_cuts(feat, cuts_pred, x_pred, inference_sddip, logger)

    # re结果保存
    logger.info("###########re正常完成，保存到dict中##############")
    instance_dict[CompareConstant.recalculate_time] = recalculate_time
    instance_dict[CompareConstant.time_pred_re] = instance_dict[CompareConstant.time_pred] + recalculate_time
    instance_dict[CompareConstant.cuts_pred_re] = cuts_pred_re


    logger.info("############Start calculate obj################")

    cuts_sddip = instance_dict[CompareConstant.cuts_sddip]

    # nocut pred pred_re 对应的obj
    samples = inference_sddip.get_fw_samples(feat, obj_fw_n_samples)
    obj_list_nocut, obj_list_pred, obj_list_pred_re = _calculate_obj(
        feat, cuts_pred, cuts_pred_re, inference_sddip, samples, logger
    )
    # sddip收敛对应的obj
    obj_list_sddip = inference_sddip.forward_obj_calculate(feat, samples, cuts_sddip)
    logger.info("################calculate pred_re obj complete#################")

    # obj结果保存
    instance_dict[CompareConstant.obj_nocut] = obj_list_nocut
    instance_dict[CompareConstant.obj_pred] = obj_list_pred
    instance_dict[CompareConstant.obj_pred_re] = obj_list_pred_re
    instance_dict[CompareConstant.obj_sddip] = obj_list_sddip


    if max_lag > 0:
        logger.info("###############进一步执行sddip##################")
        lag_time_list, lag_obj_list, lag_cuts_list, lag_time_list_re, lag_obj_list_re, lag_cuts_list_re = (
            _pred_sddip(inference_sddip, feat, cuts_pred, cuts_pred_re, max_lag, logger))
        logger.info("###############sddip完成##################")
        # 计算对应的obj
        logger.info("###############pred_sddip和pred_re_sddip计算obj ##################")
        for i in range(max_lag):
            logger.info(f"Start calculate obj with additional_sddip_cuts: {i} / {max_lag}")
            obj_list_nocut, obj_list_pred, obj_list_pred_re = _calculate_obj(
                feat, lag_cuts_list[i], lag_cuts_list_re[i], inference_sddip, samples, logger
            )
            instance_dict[CompareConstant.time_pred_sddip_list[i]] = (lag_time_list[i]
                                                                      + instance_dict[CompareConstant.time_pred])
            instance_dict[CompareConstant.time_pred_re_sddip_list[i]] = (lag_time_list_re[i]
                                                                         + instance_dict[CompareConstant.time_pred_re])
            instance_dict[CompareConstant.obj_pred_sddip_list[i]] = obj_list_pred
            instance_dict[CompareConstant.obj_pred_re_sddip_list[i]] = obj_list_pred_re


    logger.info(f"Process {instance_index} done.")
    return instance_dict

def _process_LB(instance_index, instance_dict, sddip_fw_n_samples, max_iterations, config, log_path):
    """
        单个样本的处理逻辑
        计算这几种的obj
        nocut pred pred_re pred_sddip pred_re_sddip sddip收敛
        """

    # === 每个进程单独日志 ===
    log_dir = os.path.join(log_path, "compare_LB_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"process_{instance_index}.log")
    # 清空旧日志处理器
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s][Process {instance_index}][%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            # logging.StreamHandler()  # 若希望仍打印到控制台可加这一行
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Process {instance_index} start")

    # === 正式计算 ===
    inference_sddip = Infer(
        n_stages=config.num_stage,
        n_realizations=config.n_realizations,
        N_VARS=config.n_vars,
        train_data_path=config.train_data_path,
        result_path=config.result_path
    )

    feat = instance_dict[CompareConstant.feat]

    # pred
    logger.info("##########pred sddip start##############")
    cuts_pred = instance_dict[CompareConstant.cuts_pred]
    time_pred = instance_dict[CompareConstant.time_pred]
    time_pred_list, obj_pred_list, LB_pred_list = calculate_obj_with_sddip_iteration(inference_sddip, feat, cuts_pred, sddip_fw_n_samples, time_pred, max_iterations,
                                       logger)
    logger.info("##########pred re sddip start##############")
    cuts_pred_re = instance_dict[CompareConstant.cuts_pred_re]
    time_pred_re = instance_dict[CompareConstant.time_pred_re]
    time_pred_re, obj_pred_re_list, LB_pred_re_list = calculate_obj_with_sddip_iteration(inference_sddip, feat, cuts_pred_re, sddip_fw_n_samples, time_pred_re, max_iterations,
                                       logger)

    instance_dict[CompareConstant.time_pred_sddip] = time_pred_list
    instance_dict[CompareConstant.obj_pred_sddip] = obj_pred_list
    instance_dict[CompareConstant.LB_pred_sddip] = LB_pred_list
    instance_dict[CompareConstant.time_pred_re_sddip] = time_pred_re
    instance_dict[CompareConstant.obj_pred_re_sddip] = obj_pred_re_list
    instance_dict[CompareConstant.LB_pred_re_sddip] = LB_pred_re_list

    return instance_dict




def _process_instance_quick(instance_index, instance_dict, fw_n_samples, max_lag, config, log_path):
    # === 每个进程单独日志 ===
    log_dir = os.path.join(log_path, "compare_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"process_{instance_index}.log")
    # 清空旧日志处理器
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s][Process {instance_index}][%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            # logging.StreamHandler()  # 若希望仍打印到控制台可加这一行
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Process {instance_index} start")

    # === 正式计算 ===
    inference_sddip = Infer(
        n_stages=config.num_stage,
        n_realizations=config.n_realizations,
        N_VARS=config.n_vars,
        train_data_path=config.train_data_path,
        result_path=config.result_path
    )

    feat = instance_dict[CompareConstant.feat]
    cuts_pred = instance_dict[CompareConstant.cuts_pred]
    x_pred = instance_dict[CompareConstant.x_pred]
    logger.info(f"cuts_pred.shape: {cuts_pred.shape}")
    logger.info(cuts_pred)

    """不知道什么原因，recalculate在有些样本中耗时很长，控制运行时长，超时直接舍弃该样本返回"""
    logger.info("Start recalculate cuts...")
    logger.info("re需要的参数：")
    logger.info(f"feat.device: {feat.device}")
    logger.info(f"cuts_pred.device: {cuts_pred.device}")
    logger.info(f"x_pred.device: {x_pred.device}")
    # 直接在dual model求解中限制求解时间和精度
    cuts_pred_re, recalculate_time = _recalculate_cuts(feat, cuts_pred, x_pred, inference_sddip, logger)

    # re结果保存
    logger.info("re正常完成，保存到dict中...")
    instance_dict[CompareConstant.recalculate_time] = recalculate_time
    instance_dict[CompareConstant.time_pred_re] = instance_dict[CompareConstant.time_pred] + recalculate_time
    instance_dict[CompareConstant.cuts_pred_re] = cuts_pred_re

    logger.info("Start calculate obj with (no_cuts, cuts_pred, cuts_pred_re)")
    samples = inference_sddip.get_fw_samples(feat, fw_n_samples)
    obj_list_nocut, obj_list_pred, obj_list_pred_re = _calculate_obj(
        feat, cuts_pred, cuts_pred_re, inference_sddip, samples, logger
    )
    # obj结果保存
    logger.info("calculate obj完成，保存到dict中...")
    instance_dict[CompareConstant.obj_nocut] = obj_list_nocut
    instance_dict[CompareConstant.obj_pred] = obj_list_pred
    instance_dict[CompareConstant.obj_pred_re] = obj_list_pred_re

    # 进一步执行sddip
    if max_lag > 0:
        lag_time_list, lag_obj_list, lag_cuts_list, lag_time_list_re, lag_obj_list_re, lag_cuts_list_re = (
            _pred_sddip(inference_sddip, feat, cuts_pred, cuts_pred_re, max_lag, logger))
        # 计算对应的obj
        for i in range(max_lag):
            logger.info(f"Start calculate obj with additional_sddip_cuts: {i} max_lag: {max_lag}")
            samples = inference_sddip.get_fw_samples(feat, fw_n_samples)
            obj_list_nocut, obj_list_pred, obj_list_pred_re = _calculate_obj(
                feat, lag_cuts_list[i], lag_cuts_list_re[i], inference_sddip, samples, logger
            )
            instance_dict[CompareConstant.time_pred_sddip_list[i]] = (lag_time_list[i]
                                                                      + instance_dict[CompareConstant.time_pred])
            instance_dict[CompareConstant.time_pred_re_sddip_list[i]] = (lag_time_list_re[i]
                                                                    + instance_dict[CompareConstant.time_pred_re])
            instance_dict[CompareConstant.obj_pred_sddip_list[i]] = obj_list_pred
            instance_dict[CompareConstant.obj_pred_re_sddip_list[i]] = obj_list_pred_re

    logger.info(f"Process {instance_index} done.")
    return instance_dict

def _recalculate_cuts(feat, cuts_pred, x_array, inference_sddip, logger):
    """
    处理一个instance
    重新计算intercept
    """
    recalculate_time, cuts_predicted_re = inference_sddip.intercept_recalculate(feat, cuts_pred, x_array, logger)
    return cuts_predicted_re, recalculate_time

def _calculate_obj(feat, cuts_pred, cuts_pred_re, inference_sddip, samples, logger):
    """
    针对一个instance，由于要使用相同的sample，只能将nocut、pred、pred_re放在一起
    只是计算不同cuts对应的obj
    """
    # nocut
    obj_list_nocut = inference_sddip.forward_obj_calculate(feat, samples, cuts=None)
    logger.info("calculate nocut obj complete")
    # pred
    obj_list_pred = inference_sddip.forward_obj_calculate(feat, samples, cuts_pred)
    logger.info("calculate pred obj complete")
    # pred_re
    obj_list_pred_re = inference_sddip.forward_obj_calculate(feat, samples, cuts_pred_re)
    logger.info("calculate pred_re obj complete")
    return obj_list_nocut, obj_list_pred, obj_list_pred_re


def _pred_sddip(inference_sddip, feat, cuts_pred, cuts_pred_re, max_lag, logger):
    """
    使用预测的cuts进行n次sddip，记录增加后的所有cuts，也顺便把obj也记录了
        pred + 1 lag
        pred_re + 2 lag
        ...
        pred + n lag
        pred_re + n lag
    """

    lag_time_list, lag_obj_list, lag_cuts_list = inference_sddip.sddip_n_lag(feat, cuts_pred, max_lag, logger)
    lag_time_list_re, lag_obj_list_re, lag_cuts_list_re = inference_sddip.sddip_n_lag(feat, cuts_pred_re, max_lag, logger)

    return lag_time_list, lag_obj_list, lag_cuts_list, lag_time_list_re, lag_obj_list_re, lag_cuts_list_re






def calculate_obj_with_sddip_iteration(inference_sddip, feat, cuts, fw_n_samples, additional_time, max_iterations, logger):
    """
    在给定的cut基础上迭代sddip，记录每次迭代的obj（用于计算相对误差）和时间，用于绘制进一步迭代的比较图
    需要计算两种：没有cut的sddip收敛，以及在预测cut的基础上的结果
    :param additional_time: pred或re的额外时间
    """
    time_list, obj_list, LB_list = inference_sddip.sddip_fw_n_samples(feat, cuts, fw_n_samples, max_iterations, logger)
    time_list = [additional_time + time for time in time_list]
    return time_list, obj_list, LB_list