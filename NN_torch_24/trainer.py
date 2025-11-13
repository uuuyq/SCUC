import copy
import pickle

import pandas
import torch
from torch.utils.data import DataLoader, random_split
from config import Config
from dataset import CutDataset
from model import NN_Model
import matplotlib.pyplot as plt
import os
from infer import Infer
from torch.utils.data import Subset
from datetime import datetime
from tqdm import tqdm
import multiprocessing

class Trainer:
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

        # TODO: 标准化
        # if self.standardize_flag:
        #     # 整个数据集标准化
        #     labels = torch.cat([label for _, label in data_set], dim=0)
        #     self.scaler_part1 = StandardScaler()  # 前13列
        #     self.scaler_part2 = StandardScaler()  # 最后一列
        #     # 按最后一个维度划分，分别求均值和方差
        #     self.scaler_part1.fit(labels[..., :13])
        #     self.scaler_part2.fit(labels[..., -1:])
        #     # 处理后的数据集
        #     data_set = ProcessedDataset(data_set, self.scaler_part1, self.scaler_part2)

        # 使用 random_split 方法进行划分
        generator1 = torch.Generator().manual_seed(42)
        train_dataset, val_dataset, test_dataset = random_split(data_set, [train_size, val_size, test_size], generator=generator1)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        # print(len(data_set))
        # print(len(self.train_dataset))
        # print(len(self.val_dataset))
        # print(len(self.test_dataset))

    def data_process(self):
        pass


    def train(self):
        model = NN_Model(self.config.num_stage, self.config.hidden_arr, self.config.n_vars, self.config.n_pieces).to(self.device)


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

                labels = (x * cuts[:, :, :x.size(-1)]).sum(dim=-1, keepdim=True) + cuts[:, :, -1:]
                if self.config.scenario_flag:
                    Q, pred_cuts = model(feats, x, scenarios, return_output=True, x_input_flag=self.config.x_input_flag)
                else:
                    Q, pred_cuts = model(feats, x, return_output=True, x_input_flag=self.config.x_input_flag)

                loss_Q = criterion(Q, labels)
                loss_cuts = criterion(pred_cuts, cuts)
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

                    labels = (x * cuts[:, :, :x.size(-1)]).sum(dim=-1, keepdim=True) + cuts[:, :, -1:]

                    if self.config.scenario_flag:
                        Q, pred_cuts = model(feats, x, scenarios, return_output=True, x_input_flag=self.config.x_input_flag)
                    else:
                        Q, pred_cuts = model(feats, x, return_output=True, x_input_flag=self.config.x_input_flag)

                    loss_Q = criterion(Q, labels)
                    loss_cuts = criterion(pred_cuts, cuts)
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
        model.load_state_dict(torch.load(os.path.join(self.config.result_path, "model", f"{self.params_name}.pth")))
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
            Q, pred_cuts = model(feats, x, return_output=True, x_input_flag=self.config.x_input_flag)

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
        random.seed(42)

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
        print("num_samples", num_samples)

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

    def _get_pred_cuts(self, data_sampled):
        '''
        获取pred_cuts，记录推理耗时
        :return:
        '''
        import time


        model = self._load_model()
        # copy data_sampled
        data_sampled = copy.deepcopy(data_sampled)
        num_instances = len(data_sampled["feat"])

        data_sampled["inference_calculate_X_time"] = []
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

            data_sampled["inference_calculate_X_time"].append(end - start)
            data_sampled["cuts_pred"].append(pred_cuts)
            data_sampled["x_calculated"].append(x_tensor)
        return data_sampled

    def _recalculate_cuts(self, data_sampled):
        """
        重新计算intercept
        :param data_sampled:
        :return:
        """
        feat_sampled = data_sampled["feat"]
        cuts_pred = data_sampled["cuts_pred"]
        num_instances = len(feat_sampled)

        data_sampled["cuts_pred_re"] = []
        data_sampled["recalculate_time"] = []

        inference_sddip = Infer(n_stages=self.config.num_stage,
                                n_realizations=self.config.n_realizations,
                                N_VARS=self.config.n_vars,
                                train_data_path=self.config.train_data_path,
                                result_path=self.config.result_path
                                )
        # no cut sddip
        for index in tqdm(range(num_instances), desc="recalculate"):
            print(f"recalculate step: {index} / {num_instances}")
            recalculate_time, cuts_predicted_re = inference_sddip.intercept_recalculate(feat_sampled[index], cuts_pred[index])
            data_sampled["cuts_pred_re"].append(cuts_predicted_re)
            data_sampled["recalculate_time"].append(recalculate_time)
        return data_sampled


    def pred_and_re(self, data_sampled, save_flag):
        data_sampled = self._get_pred_cuts(data_sampled)
        data_sampled = self._recalculate_cuts(data_sampled)
        if save_flag:
            t = datetime.now().strftime("%m-%d-%H-%M")
            save_path = os.path.join(self.config.result_path, "compare", f"num-{len(data_sampled['feat'])}_sampled_sddip_pred_and_re_{t}.pkl")
            torch.save(data_sampled, save_path)
        return data_sampled



    def pred_sddip(self, data_sampled, max_lag=3):
        """
        使用预测的cuts进行n次sddip，记录增加后的所有cuts，也顺便把obj也记录了
            inference + 1 lag
            inference + 2 lag
            ...
            inference + n lag
        :param data_sampled: feat cuts_pred cuts_pred_re (data_sampled, get_pred_cuts, recalculate_cuts)
        :return:
        """


        inference_sddip = Infer(n_stages=self.config.num_stage,
                                n_realizations=self.config.n_realizations,
                                N_VARS=self.config.n_vars,
                                train_data_path=self.config.train_data_path,
                                result_path=self.config.result_path
                                )

        feat_sampled = data_sampled["feat"]
        cuts_pred = data_sampled["cuts_pred"]
        cuts_pred_re = data_sampled["cuts_pred_re"]
        for i in range(max_lag):
            data_sampled[f"cuts_pred_{i + 1}_lag"] = []
            data_sampled[f"obj_pred_{i + 1}_lag_only_one"] = []
            data_sampled[f"time_pred_{i + 1}_lag"] = []  # 只是lag的时间，还要加上pred
            data_sampled[f"cuts_pred_re_{i + 1}_lag"] = []
            data_sampled[f"obj_pred_re_{i + 1}_lag_only_one"] = []  # 以及re的时间
            data_sampled[f"time_pred_re_{i + 1}_lag"] = []
        for index in tqdm(range(len(feat_sampled)), desc=f"pred sddip"):
            print(f"pred sddip step: {index} / {len(feat_sampled)}")
            lag_time_list, lag_obj_list, lag_cuts_list = inference_sddip.sddip_n_lag(feat_sampled[index], cuts_pred[index], max_lag)
            lag_time_list_re, lag_obj_list_re, lag_cuts_list_re = inference_sddip.sddip_n_lag(feat_sampled[index], cuts_pred_re[index], max_lag)

            for i in range(max_lag):
                data_sampled[f"cuts_pred_{i + 1}_lag"].append(lag_cuts_list[i])
                data_sampled[f"obj_pred_{i + 1}_lag_only_one"].append(lag_obj_list[i])
                data_sampled[f"time_pred_{i + 1}_lag"].append(lag_time_list[i])
                data_sampled[f"cuts_pred_re_{i + 1}_lag"].append(lag_cuts_list_re[i])
                data_sampled[f"obj_pred_re_{i + 1}_lag_only_one"].append(lag_obj_list_re[i])
                data_sampled[f"time_pred_re_{i + 1}_lag"].append(lag_time_list_re[i])
                # 简单看一下是不是增加了cut
                # print(f"{i + 1}_lag cuts shape: {lag_cuts_list[i].shape}")
                # print(f"{i + 1}_lag_re cuts shape: {lag_cuts_list_re[i].shape}")
        t = datetime.now().strftime("%m-%d-%H-%M")
        save_path = os.path.join(self.config.result_path, "compare", f"num-{len(feat_sampled)}_sampled_sddip_pred_and_re_maxl-{max_lag}_{t}.pkl")
        torch.save(data_sampled, save_path)
        return data_sampled



    def compare_obj(self, data_sampled, fw_n_samples=20):
        """
        只是比较obj
        :return:
        """
        compare_obj_result = {
            "obj_pred": [],
            "obj_pred_re": [],
        }
        feat_sampled = data_sampled["feat"]
        cuts_pred = data_sampled["cuts_pred"]
        cuts_pred_re = data_sampled["cuts_pred_re"]

        inference_sddip = Infer(n_stages=self.config.num_stage,
                                n_realizations=self.config.n_realizations,
                                N_VARS=self.config.n_vars,
                                train_data_path=self.config.train_data_path,
                                result_path=self.config.result_path
                                )

        num_instances = len(data_sampled["feat"])
        for index in tqdm(range(num_instances), desc="obj compare"):
            print(f"obj compare step: {index} / {num_instances}")
            samples = inference_sddip.get_fw_samples(feat_sampled[index], fw_n_samples)
            # pred
            obj_list_pred = inference_sddip.forward_obj_calculate(feat_sampled[index], samples, cuts_pred[index])
            # pred_re
            obj_list_pred_re = inference_sddip.forward_obj_calculate(feat_sampled[index], samples, cuts_pred_re[index])

            compare_obj_result["obj_pred"].append(obj_list_pred)
            compare_obj_result["obj_pred_re"].append(obj_list_pred_re)
        return compare_obj_result

    def compare(self, fw_n_samples, max_lag=None, data_sampled=None):

        """
        6. 比较时间和obj
            sddip
            inference
            inference + recalculate
        """
        '''
        data_sampled = {
            "instance_index": [],
            "feat": [],
            "scenario": [],
            "cuts": [],
            "x_cuts": []
            "inference_calculate_X_time": []
            "cuts_pred": []
            "x_calculated": []
            "time_sddip"
            "obj_sddip"
            "cut_sddip"
            "cuts_pred_re" 
            "recalculate_time"
        }
        '''
        inference_sddip = Infer(n_stages=self.config.num_stage,
                                n_realizations=self.config.n_realizations,
                                N_VARS=self.config.n_vars,
                                train_data_path=self.config.train_data_path,
                                result_path=self.config.result_path
                                )
        compare_result = {
            "instance_id": data_sampled["instance_index"],
            "obj_pred": [],  # list [num_instances, num_scenarios]
            "time_pred": [],
            "obj_pred_re": [],
            "time_pred_re": [],
            "obj_sddip": [],
            "time_sddip": [],
            "obj_nocut": [],
        }
        if max_lag is not None:
            for i in range(max_lag):
                compare_result[f"obj_pred_{i + 1}_lag"] = []
                compare_result[f"time_pred_{i + 1}_lag"] = []
                compare_result[f"obj_pred_re_{i + 1}_lag"] = []
                compare_result[f"time_pred_re_{i + 1}_lag"] = []

        # obj compare
        """
        使用相同的前向路径，记录不同路径对应的obj
        """
        # cuts_sddip = data_sampled["cuts_sddip"]
        time_sddip = data_sampled["time_sddip"]
        cuts_pred = data_sampled["cuts_pred"]
        inference_time = data_sampled["inference_calculate_X_time"]
        cuts_pred_re = data_sampled["cuts_pred_re"]
        recalculate_time = data_sampled["recalculate_time"]
        feat_sampled = data_sampled["feat"]

        num_instances = len(time_sddip)
        for index in tqdm(range(num_instances), desc="obj compare"):
            print(f"obj compare step: {index} / {num_instances}")
            samples = inference_sddip.get_fw_samples(feat_sampled[index], fw_n_samples)
            # pred
            obj_list_pred = inference_sddip.forward_obj_calculate(feat_sampled[index], samples, cuts_pred[index])
            # pred_re
            obj_list_pred_re = inference_sddip.forward_obj_calculate(feat_sampled[index], samples, cuts_pred_re[index])
            # sddip
            # obj_list_sddip = inference_sddip.forward_obj_calculate(feat_sampled[index], samples, cuts_sddip[index])
            # no cut
            obj_list_nocut = inference_sddip.forward_obj_calculate(feat_sampled[index], samples, None)

            compare_result["obj_pred"].append(obj_list_pred)
            compare_result["time_pred"].append(inference_time[index])
            compare_result["obj_pred_re"].append(obj_list_pred_re)
            compare_result["time_pred_re"].append(inference_time[index] + recalculate_time[index])
            # compare_result["obj_sddip"].append(obj_list_sddip)
            compare_result["time_sddip"].append(time_sddip[index])
            compare_result["obj_nocut"].append(obj_list_nocut)
            if max_lag is not None:
                # n lag
                for i in range(max_lag):
                    obj_list_lag = inference_sddip.forward_obj_calculate(feat_sampled[index], samples, data_sampled[f"cuts_pred_{i + 1}_lag"][index])
                    compare_result[f"obj_pred_{i + 1}_lag"].append(obj_list_lag)
                    compare_result[f"time_pred_{i + 1}_lag"].append(inference_time[index] + data_sampled[f"time_pred_{i + 1}_lag"][index])
                    obj_list_lag_re = inference_sddip.forward_obj_calculate(feat_sampled[index], samples, data_sampled[f"cuts_pred_re_{i + 1}_lag"][index])
                    compare_result[f"obj_pred_re_{i + 1}_lag"].append(obj_list_lag_re)
                    # compare_result[f"time_pred_re_{i + 1}_lag"].append(inference_time[index] + recalculate_time[index] + data_sampled[f"time_pred_re_{i + 1}_lag"][index])

        torch.save(compare_result, os.path.join(self.config.result_path, "compare", f"num-{num_instances}_compare_result_fw-{fw_n_samples}.pkl"))

        return compare_result

    def compare_stage(self, fw_n_samples, max_lag=None, data_sampled=None):

        """
        6. 比较时间和obj  不同阶段的obj分别记录
            sddip
            inference
            inference + recalculate
        """
        '''
        data_sampled = {
            "instance_index": [],
            "feat": [],
            "scenario": [],
            "cuts": [],
            "x_cuts": []
            "inference_calculate_X_time": []
            "cuts_pred": []
            "x_calculated": []
            "time_sddip"
            "obj_sddip"
            "cut_sddip"
            "cuts_pred_re" 
            "recalculate_time"
        }
        '''
        inference_sddip = Infer(n_stages=self.config.num_stage,
                                n_realizations=self.config.n_realizations,
                                N_VARS=self.config.n_vars,
                                train_data_path=self.config.train_data_path,
                                result_path=self.config.result_path
                                )
        compare_result = {
            "instance_id": data_sampled["instance_index"],
            "obj_pred": [],  # list [num_instances, num_scenarios]
            "time_pred": [],
            "obj_pred_re": [],
            "time_pred_re": [],
            "obj_sddip": [],
            "time_sddip": [],
            "obj_nocut": [],
        }
        if max_lag is not None:
            for i in range(max_lag):
                compare_result[f"obj_pred_{i + 1}_lag"] = []
                compare_result[f"time_pred_{i + 1}_lag"] = []
                compare_result[f"obj_pred_re_{i + 1}_lag"] = []
                compare_result[f"time_pred_re_{i + 1}_lag"] = []

        # obj compare
        """
        使用相同的前向路径，记录不同路径对应的obj
        """
        cuts_sddip = data_sampled["cuts_sddip"]
        time_sddip = data_sampled["time_sddip"]
        cuts_pred = data_sampled["cuts_pred"]
        inference_time = data_sampled["inference_calculate_X_time"]
        cuts_pred_re = data_sampled["cuts_pred_re"]
        recalculate_time = data_sampled["recalculate_time"]
        feat_sampled = data_sampled["feat"]

        num_instances = len(time_sddip)
        for index in tqdm(range(num_instances), desc="obj compare"):
            print(f"obj compare step: {index} / {num_instances}")
            samples = inference_sddip.get_fw_samples(feat_sampled[index], fw_n_samples)
            # pred
            obj_list_pred = inference_sddip.forward_obj_calculate_stage(feat_sampled[index], samples, cuts_pred[index])
            # pred_re
            obj_list_pred_re = inference_sddip.forward_obj_calculate_stage(feat_sampled[index], samples, cuts_pred_re[index])
            # sddip
            obj_list_sddip = inference_sddip.forward_obj_calculate_stage(feat_sampled[index], samples, cuts_sddip[index])
            # no cut
            obj_list_nocut = inference_sddip.forward_obj_calculate_stage(feat_sampled[index], samples, None)

            compare_result["obj_pred"].append(obj_list_pred)
            compare_result["time_pred"].append(inference_time[index])
            compare_result["obj_pred_re"].append(obj_list_pred_re)
            compare_result["time_pred_re"].append(inference_time[index] + recalculate_time[index])
            compare_result["obj_sddip"].append(obj_list_sddip)
            compare_result["time_sddip"].append(time_sddip[index])
            compare_result["obj_nocut"].append(obj_list_nocut)
            if max_lag is not None:
                # n lag
                for i in range(max_lag):
                    obj_list_lag = inference_sddip.forward_obj_calculate_stage(feat_sampled[index], samples, data_sampled[f"cuts_pred_{i + 1}_lag"][index])
                    compare_result[f"obj_pred_{i + 1}_lag"].append(obj_list_lag)
                    compare_result[f"time_pred_{i + 1}_lag"].append(inference_time[index] + data_sampled[f"time_pred_{i + 1}_lag"][index])
                    obj_list_lag_re = inference_sddip.forward_obj_calculate_stage(feat_sampled[index], samples, data_sampled[f"cuts_pred_re_{i + 1}_lag"][index])
                    compare_result[f"obj_pred_re_{i + 1}_lag"].append(obj_list_lag_re)
                    compare_result[f"time_pred_re_{i + 1}_lag"].append(inference_time[index] + recalculate_time[index] + data_sampled[f"time_pred_re_{i + 1}_lag"][index])

        torch.save(compare_result, os.path.join(self.config.result_path, "compare", f"num-{num_instances}_compare_result_fw-{fw_n_samples}_stage.pkl"))

        return compare_result


    def compare_mutil_process(self, fw_n_samples, max_lag, data_sampled=None, num_processes=None):
        """
        6. 比较时间和obj
            sddip
            inference
            inference + recalculate
        """
        from tqdm import tqdm
        import multiprocessing
        from functools import partial

        if data_sampled is None:
            data_sampled = torch.load(os.path.join(self.config.result_path, "compare", f"num-50_sampled_sddip_fixed.pkl"))

        inference_sddip = Infer(n_stages=self.config.num_stage,
                                n_realizations=self.config.n_realizations,
                                N_VARS=self.config.n_vars,
                                train_data_path=self.config.train_data_path,
                                result_path=self.config.result_path
                                )

        compare_result = {
            "instance_id": data_sampled["instance_index"],
            "obj_pred": [],  # list [num_instances, num_scenarios]
            "time_pred": [],
            "obj_pred_re": [],
            "time_pred_re": [],
            "obj_sddip": [],
            "time_sddip": [],
            "obj_nocut": [],
        }

        for i in range(max_lag):
            compare_result[f"obj_pred_{i + 1}_lag"] = []
            compare_result[f"time_pred_{i + 1}_lag"] = []
            compare_result[f"obj_pred_re_{i + 1}_lag"] = []
            compare_result[f"time_pred_re_{i + 1}_lag"] = []

        # Prepare data for multiprocessing
        cuts_sddip = data_sampled["sddip_cuts"]
        time_sddip = data_sampled["sddip_time"]
        cuts_pred = data_sampled["cuts_pred"]
        inference_time = data_sampled["inference_calculate_X_time"]
        cuts_pred_re = data_sampled["cuts_pred_re"]
        recalculate_time = data_sampled["recalculate_time"]
        feat_sampled = data_sampled["feat"]
        num_instances = len(time_sddip)

        # Create a list of all indices to process
        indices = list(range(num_instances))

        # Determine number of processes to use
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()

        # Function to process a single index
        def process_index(index, inference_sddip, feat_sampled, cuts_pred, cuts_pred_re,
                          cuts_sddip, inference_time, recalculate_time, data_sampled, max_lag):
            samples = inference_sddip.get_fw_samples(feat_sampled[index], fw_n_samples)

            # pred
            obj_list_pred = inference_sddip.forward_obj_calculate(feat_sampled[index], samples, cuts_pred[index])
            # pred_re
            obj_list_pred_re = inference_sddip.forward_obj_calculate(feat_sampled[index], samples, cuts_pred_re[index])
            # sddip
            obj_list_sddip = inference_sddip.forward_obj_calculate(feat_sampled[index], samples, cuts_sddip[index])
            # no cut
            obj_list_nocut = inference_sddip.forward_obj_calculate(feat_sampled[index], samples, None)

            result = {
                "index": index,
                "obj_pred": obj_list_pred,
                "obj_pred_re": obj_list_pred_re,
                "obj_sddip": obj_list_sddip,
                "obj_nocut": obj_list_nocut,
                "time_pred": inference_time[index],
                "time_pred_re": inference_time[index] + recalculate_time[index],
                "time_sddip": time_sddip[index],
            }

            # n lag
            for i in range(max_lag):
                obj_list_lag = inference_sddip.forward_obj_calculate(feat_sampled[index], samples,
                                                                     data_sampled[f"cuts_pred_{i + 1}_lag"][index])
                result[f"obj_pred_{i + 1}_lag"] = obj_list_lag
                result[f"time_pred_{i + 1}_lag"] = inference_time[index] + data_sampled[f"time_pred_{i + 1}_lag"][index]

                obj_list_lag_re = inference_sddip.forward_obj_calculate(feat_sampled[index], samples,
                                                                        data_sampled[f"cuts_pred_re_{i + 1}_lag"][
                                                                            index])
                result[f"obj_pred_re_{i + 1}_lag"] = obj_list_lag_re
                result[f"time_pred_re_{i + 1}_lag"] = inference_time[index] + recalculate_time[index] + \
                                                      data_sampled[f"time_pred_re_{i + 1}_lag"][index]

            return result

        # Create a partial function with fixed arguments
        partial_process = partial(process_index,
                                  inference_sddip=inference_sddip,
                                  feat_sampled=feat_sampled,
                                  cuts_pred=cuts_pred,
                                  cuts_pred_re=cuts_pred_re,
                                  cuts_sddip=cuts_sddip,
                                  inference_time=inference_time,
                                  recalculate_time=recalculate_time,
                                  data_sampled=data_sampled,
                                  max_lag=max_lag)

        # Process in parallel
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(partial_process, indices), total=len(indices), desc="Processing instances"))

        # Sort results by original index (important if order matters)
        results.sort(key=lambda x: x['index'])

        # Combine results into the final structure
        for res in results:
            compare_result["obj_pred"].append(res["obj_pred"])
            compare_result["time_pred"].append(res["time_pred"])
            compare_result["obj_pred_re"].append(res["obj_pred_re"])
            compare_result["time_pred_re"].append(res["time_pred_re"])
            compare_result["obj_sddip"].append(res["obj_sddip"])
            compare_result["time_sddip"].append(res["time_sddip"])
            compare_result["obj_nocut"].append(res["obj_nocut"])

            for i in range(max_lag):
                compare_result[f"obj_pred_{i + 1}_lag"].append(res[f"obj_pred_{i + 1}_lag"])
                compare_result[f"time_pred_{i + 1}_lag"].append(res[f"time_pred_{i + 1}_lag"])
                compare_result[f"obj_pred_re_{i + 1}_lag"].append(res[f"obj_pred_re_{i + 1}_lag"])
                compare_result[f"time_pred_re_{i + 1}_lag"].append(res[f"time_pred_re_{i + 1}_lag"])

        torch.save(compare_result,
                   os.path.join(self.config.result_path, "compare", f"num-{num_instances}_compare_result_fw-{fw_n_samples}.pkl"))

        return compare_result


    def calculate_obj_with_sddip_iteration(self, index, fw_n_samples, feat_sampled, cuts_pred, cut_pred_init_time, cuts_pred_re=None, cut_pred_re_init_time=None):
        """
        在给定的cut基础上迭代sddip，记录每次迭代的obj（用于计算相对误差）和时间，用于绘制进一步迭代的比较图
        需要计算两种：没有cut的sddip收敛，以及在预测cut的基础上的结果
        :return:
        """
        inference_sddip = Infer(n_stages=self.config.num_stage,
                                n_realizations=self.config.n_realizations,
                                N_VARS=self.config.n_vars,
                                train_data_path=self.config.train_data_path,
                                result_path=self.config.result_path
                                )

        result = {}

        time_list, obj_list, LB_list = inference_sddip.sddip_fw_n_samples(feat_sampled, None, fw_n_samples)
        result["sddip_time"] = time_list
        result["sddip_obj"] = obj_list
        result["sddip_LB"] = LB_list

        time_list_pred, obj_list_pred, LB_list_pred = inference_sddip.sddip_fw_n_samples(feat_sampled, cuts_pred, fw_n_samples)
        result["pred_time"] = [time + cut_pred_init_time for time in time_list_pred]
        result["pred_obj"] = obj_list_pred
        result["pred_LB"] = LB_list_pred

        if cuts_pred_re is not None:
            time_list_re, obj_list_re, LB_list_re = inference_sddip.sddip_fw_n_samples(feat_sampled, cuts_pred_re,
                                                                                       fw_n_samples)
            result["pred_re_time"] = [time + cut_pred_re_init_time for time in time_list_re]
            result["pred_re_LB"] = LB_list_re
            result["pred_re_obj"] = obj_list_re

        torch.save(result,
                   os.path.join(r"D:\tools\workspace_pycharm\sddip-SCUC-6-24\NN_torch_24\result_sigma_7_30\compare\calculate_sddip_with_sddip", f"calculate_obj_with_sddip_result_{index}.pkl"))

        return result

