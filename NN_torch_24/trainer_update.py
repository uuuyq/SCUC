import copy
import logging

import torch
from torch.utils.data import DataLoader, random_split
from config import Config
from dataset import CutDataset, CutDatasetNormalized
from model import NN_Model
import matplotlib.pyplot as plt
import os
from infer import Infer
from torch.utils.data import Subset
from datetime import datetime
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

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
        self.params_name = f"model_{self.config.num_data}_{self.config.n_pieces}_lr-{self.config.LEARNING_RATE}_wd-{self.config.weight_decay}_gamma-{self.config.gamma}_dim-{self.config.hidden_arr}_standard-{self.config.standard_flag}"


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

    def data_process(self):
        pass


    def train(self):


        model_path = os.path.join(self.config.result_path, "model", f"{self.params_name}.pth")

        if os.path.exists(model_path):
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
                self.cut_std = self.cut_std.to(self.device)
                self.cut_mean = self.cut_mean.to(self.device)

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

                    val_loss += loss.item() * feats.size(0)

            val_loss = val_loss / len(self.val_dataset)
            val_loss_history.append(val_loss)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(self.config.result_path, "model", f"{self.params_name}.pth"))

            # 打印进度
            print(f"Epoch [{epoch + 1}/{self.config.N_EPOCHS}] | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            print(f"loss_Q: {train_loss_Q:.4f}  loss_cuts: {train_loss_cuts:.4f}")

        # 绘图
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
        filename = f"loss_plot_gamma_{self.config.gamma}.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath)  # 保存为文件
        plt.close()  # 关闭窗口，避免重复显示（尤其在循环中）

        # 保存为 CSV
        loss_filename = f"loss_gamma_{self.config.gamma}.pkl"
        loss_path = os.path.join(save_path, loss_filename)
        loss_df = {
            "train_loss": train_loss_history,
            "val_loss": val_loss_history
        }
        torch.save(loss_df, loss_path)
        print(f"Train/Val loss saved to CSV: {loss_path}")

        print(f"Plot saved to: {filepath}")


    def _load_model(self):
        model = NN_Model(self.config.num_stage, self.config.hidden_arr, self.config.n_vars, self.config.n_pieces)
        model_path = os.path.join(self.config.result_path, "model", f"{self.params_name}.pth")
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
        :return: x_tensor [n_stages-1, n_pieces, N_VARS+1]
        '''

        inference_sddip = Infer(n_stages=self.config.num_stage,
                                n_realizations=self.config.n_realizations,
                                N_VARS=self.config.n_vars,
                                train_data_path=self.config.train_data_path,
                                result_path=self.config.result_path
                                )
        x_tensor = inference_sddip.calculate_x(feat, self.config.n_pieces)
        # print("x_tensor.shape", x_tensor.shape)  # [23, num_X, N_VARS]
        return x_tensor

    def sample_test_dataset(self, num_instances, save_flag):

        '''
        1. 选取test_data中的部分数据，计算x，预测cuts
        :param num_instances:
        :param save_flag:
        :return:
        '''
        '''
        data_set.feat_tensor.shape torch.Size([5000, 23, 14])  [num_data, num_stage-1, n_feats]
        data_set.cut_tensor.shape torch.Size([5000, 23, 15, 14])
        data_set.scenario_tensor.shape torch.Size([5000, 23, 6, 12]) [num_data, num_stage-1, num_realizations, -]
        data_set.x_tensor.shape torch.Size([5000, 23, 15, 13])
        '''

        # 随机选取dataset中的部分数据
        import random
        random.seed(43)

        test_dataset = self.test_dataset
        total = len(test_dataset)

        # 找到 instance_index < 3000 的所有可选索引
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
        data_sampled = {
            "instance_index": [],
            "feat": [],
            "scenario": [],
            "cuts": [],
            "x_cuts": []
        }

        for i in range(len(test_dataset_sampled)):
            feat, scenario, cut, x = test_dataset_sampled[i]
            feat = feat.detach().cpu().clone()
            scenario = scenario.detach().cpu().clone()
            cut = cut.detach().cpu().clone()
            x = x.detach().cpu().clone()

            instance_index = feat[0][0]
            data_sampled["instance_index"].append(instance_index)
            data_sampled["feat"].append(feat)
            data_sampled["scenario"].append(scenario)
            data_sampled["cuts"].append(cut)
            data_sampled["x_cuts"].append(x)
        if save_flag:
            # t = datetime.now().strftime("%m-%d-%H")
            save_path = os.path.join(self.config.result_path, "compare", f"num-{num_instances}_sampled_fixed.pkl")
            # with open(save_path, "wb") as f:
            #     pickle.dump(data_sampled, f)
            torch.save(data_sampled, save_path)

        print("sampling test dataset end")

        return data_sampled

    def sampled_sddip(self, data_sampled, num_instances=None):
        """
        使用采样的数据进行sddip, 执行一次之后尽可能不再更改，使用的sampled的数据也固定
        :return:
        """

        # 使用采样的数据进行sddip
        from tqdm import tqdm
        feat_sampled = data_sampled["feat"]
        if num_instances is None:
            num_instances = len(feat_sampled)

        inference_sddip = Infer(n_stages=self.config.num_stage,
                                n_realizations=self.config.n_realizations,
                                N_VARS=self.config.n_vars,
                                train_data_path=self.config.train_data_path,
                                result_path=self.config.result_path
                                )
        data_sampled["time_sddip"] = []
        data_sampled["obj_sddip_only_one"] = []
        data_sampled["cuts_sddip"] = []
        # start sddip with no cut
        for index in tqdm(range(num_instances), desc="sampled data sddip time"):
            print(f"sampled data sddip step: {index} / {num_instances}")
            sddip_time, obj, cuts_tensor = inference_sddip.sddip(feat_sampled[index], cuts=None, return_cuts=True)
            data_sampled["time_sddip"].append(sddip_time)
            data_sampled["obj_sddip_only_one"].append(obj)
            data_sampled["cuts_sddip"].append(cuts_tensor)

        time = datetime.now().strftime("%m-%d-%H-%M")
        save_path = os.path.join(self.config.result_path, "compare", f"num-{num_instances}_sampled_sddip_{time}.pkl")
        torch.save(data_sampled, save_path)

        return data_sampled

    def sampled_read(self, data_sampled):
        """
        sddip太慢了，简单读取生成数据时的obj和time看一下
        :return:
        """

        # 读取生成数据是的obj和time
        feat_sampled = data_sampled["feat"]
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
        objs = []
        times = []
        num_instances = len(feat_sampled)

        for feat in feat_sampled:
            instance_index = int(feat[0][0])

            # 读取 obj 文件
            obj_path = os.path.join(obj_path_root, f"{instance_index}_cuts_obj.txt")
            with open(obj_path, "r") as f_obj:
                lines = f_obj.readlines()
                last_obj = float(lines[-1].strip())  # 取最后一个数

            objs.append(last_obj)

            # 取对应时间
            if instance_index in time_dict:
                times.append(time_dict[instance_index])
            else:
                times.append(None)  # 若 time 文件中没有对应 index

        # objs 和 times 中存储每个样本对应的最后 obj 和运行时间
        print(objs)
        print(times)
        data_sampled["time_sddip"] = times
        data_sampled["obj_sddip_only_one"] = objs

        time = datetime.now().strftime("%m-%d-%H-%M")
        save_path = os.path.join(self.config.result_path, "compare", f"num-{num_instances}_sampled_sddip_{time}.pkl")
        torch.save(data_sampled, save_path)

        return data_sampled




    # def _get_pred_cuts(self, feat):
    #     """
    #     预测一个instance
    #     获取pred_cuts，记录推理耗时
    #     :return:
    #     """
    #     import time
    #
    #     model = self._load_model()
    #     # 记录开始时间
    #     start = time.time()
    #     # print("feat.shape", feat.shape)
    #     x_tensor = self._calculate_x(feat)
    #     # print("x_tensor", x_tensor.shape)
    #     cuts_pred = self._predict_single(model, (feat, x_tensor))
    #     # print("pred_cuts", pred_cuts.shape)
    #     pred_time = time.time() - start
    #
    #     return cuts_pred, x_tensor, pred_time




    def integrate_result(self, result):
        """整合为原来的形式，外部是dict，元素为list"""
        combined_dict = {}

        for d in result:
            for key, value in d.items():
                if key not in combined_dict:
                    combined_dict[key] = []
                combined_dict[key].append(value)

        return combined_dict

    def _get_pred_cuts(self, data_sampled, file_name=None):
        '''
        获取pred_cuts，记录推理耗时
        :return:
        '''
        import time


        model = self._load_model()
        # copy data_sampled
        data_sampled = copy.deepcopy(data_sampled)
        num_instances = len(data_sampled["feat"])

        data_sampled["time_pred"] = []
        data_sampled["cuts_pred"] = []
        data_sampled["x_calculated"] = []


        for i in tqdm(range(num_instances), desc="pred cuts processing"):

            feat = data_sampled["feat"][i]
            # 记录开始时间
            start = time.time()
            # print("feat.shape", feat.shape)
            x_tensor = self._calculate_x(feat)
            # print("x_tensor", x_tensor.shape)
            pred_cuts = self._predict_single(model, (feat, x_tensor))
            # print("pred_cuts", pred_cuts.shape)
            end = time.time()

            data_sampled["time_pred"].append(end - start)
            data_sampled["cuts_pred"].append(pred_cuts)
            data_sampled["x_calculated"].append(x_tensor)
        torch.save(data_sampled, os.path.join(self.config.result_path, "compare", f"{file_name}",
                                        f"num-{num_instances}_data_sampled_pred.pkl"))
        return data_sampled

    def compare_multiprocess(self, num_instances, fw_n_samples, file_name):
        """
        使用多进程方式比较时间和obj。
        注意：如果CPU紧张，pred或re的耗时可能会略有上升。
        """
        if not os.path.exists(os.path.join(self.config.result_path, "compare", f"{file_name}",
                                        f"num-{num_instances}_data_sampled_pred.pkl")):
            # 创建文件夹
            if not os.path.exists(os.path.join(self.config.result_path, "compare", f"{file_name}")):
                os.makedirs(os.path.join(self.config.result_path, "compare", f"{file_name}"))

            data_sampled = self.sample_test_dataset(num_instances, save_flag=True)

            data_sampled = self.sampled_read(data_sampled)

            data_sampled = self._get_pred_cuts(data_sampled, file_name)
        else:
            data_sampled = torch.load(os.path.join(self.config.result_path, "compare", f"{file_name}",
                                        f"num-{num_instances}_data_sampled_pred.pkl"))

        def dict_tensorlist_to_arraylist(d):
            """
            将字典中所有 value 为 list 且内部为 tensor 的项，
            转换为 list 且内部为 numpy.ndarray。
            其他类型保持不变。
            """
            result = {}
            for k, v in d.items():
                if isinstance(v, list) and all(isinstance(x, torch.Tensor) for x in v):
                    result[k] = [x.detach().cpu().numpy() for x in v]
                else:
                    result[k] = v
            return result

        data_sampled = dict_tensorlist_to_arraylist(data_sampled)

        print("输出data_sampled：")
        print(data_sampled)


        # result = []
        # for instance_index in range(num_instances):
        #     result.append(_process_instance(instance_index, data_sampled, fw_n_samples, self.config))

        num_workers = min(mp.cpu_count(), 5)  # 限制最大并行数，防止CPU爆满
        print(f"Using {num_workers} processes for parallel comparison...")

        # with mp.Pool(processes=num_workers) as pool:
        #     func = partial(_process_instance, data_sampled=data_sampled, fw_n_samples=fw_n_samples, config=self.config)
        #     result = pool.map(func, range(num_instances))

        func = partial(_process_instance,
                       data_sampled=data_sampled,
                       fw_n_samples=fw_n_samples,
                       config=self.config)
        result = []
        timeout_sec = 3600  # 每个进程最多运行 3600 秒

        with mp.Pool(processes=num_workers) as pool:
            async_results = [pool.apply_async(func, args=(i,)) for i in range(num_instances)]

            for i, async_result in enumerate(async_results):
                try:
                    res = async_result.get(timeout=timeout_sec)
                    result.append(res)
                except mp.TimeoutError:
                    print(f"⚠️ Instance {i} timed out after {timeout_sec} seconds!")
                    # results.append(None)  # 可用占位符
                except Exception as e:
                    print(f"❌ Instance {i} failed with error: {e}")
                    # results.append(None)


        # 保存原始结果
        torch.save(result, os.path.join(self.config.result_path, "compare", f"{file_name}",
                                        f"num-{num_instances}_compare_result_fw-{fw_n_samples}.pkl"))

        # 聚合结果
        integrated_result = self.integrate_result(result)

        torch.save(integrated_result, os.path.join(self.config.result_path, "compare", f"{file_name}",
                                                   f"num-{num_instances}_compare_result_fw-{fw_n_samples}_integrated.pkl"))

        print("Parallel comparison complete.")
        return result, integrated_result


def _process_instance(instance_index, data_sampled, fw_n_samples, config):
    """单个样本的处理逻辑，供多进程调用"""

    # === 每个进程单独日志 ===
    log_dir = os.path.join(config.result_path, "compare_logs")
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

    feat = data_sampled["feat"][instance_index]
    cuts_pred = data_sampled["cuts_pred"][instance_index]

    logger.info(cuts_pred.shape)

    logger.info(cuts_pred)

    logger.info("Start recalculate cuts")
    cuts_pred_re, recalculate_time = _recalculate_cuts(feat, cuts_pred, inference_sddip, logger)
    logger.info("Recalculate complete")

    logger.info("Start calculate obj")
    obj_list_nocut, obj_list_pred, obj_list_pred_re = calculate_obj(
        feat, cuts_pred, cuts_pred_re, inference_sddip, fw_n_samples, logger
    )
    logger.info("Calculate obj complete")

    instance_dict = {
        "instance_index": instance_index,
        "feat": feat,
        "scenario": data_sampled["scenario"][instance_index],
        "cuts": data_sampled["cuts"][instance_index],
        "x_cuts": data_sampled["x_cuts"][instance_index],
        "time_sddip": data_sampled["time_sddip"][instance_index],
        "obj_sddip_only_one": data_sampled["obj_sddip_only_one"][instance_index],
        "time_pred": data_sampled["time_pred"][instance_index],
        "cuts_pred": cuts_pred,
        "x_calculated": data_sampled["x_calculated"][instance_index],
        "recalculate_time": recalculate_time,
        "time_pred_re": data_sampled["time_pred"][instance_index] + recalculate_time,
        "obj_nocut": obj_list_nocut,
        "obj_pred": obj_list_pred,
        "obj_pred_re": obj_list_pred_re
    }

    logger.info(f"Process {instance_index} done.")
    return instance_dict

def _recalculate_cuts(feat, cuts_pred, inference_sddip, logger):
    """
    处理一个instance
    重新计算intercept
    """

    # inference_sddip = Infer(n_stages=self.config.num_stage,
    #                         n_realizations=self.config.n_realizations,
    #                         N_VARS=self.config.n_vars,
    #                         train_data_path=self.config.train_data_path,
    #                         result_path=self.config.result_path
    #                         )
    recalculate_time, cuts_predicted_re = inference_sddip.intercept_recalculate(feat, cuts_pred, logger)
    return cuts_predicted_re, recalculate_time


def calculate_obj(feat, cuts_pred, cuts_pred_re, inference_sddip, fw_n_samples, logger):
    """
    针对一个instance，由于要使用相同的sample，只能将nocut、pred、pred_re放在一起
    只是计算不同cuts对应的obj
    """

    samples = inference_sddip.get_fw_samples(feat, fw_n_samples)
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