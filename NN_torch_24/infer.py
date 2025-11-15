import logging
import os
import torch
from sddip_script_update import sddipclassical
import time



class Infer:

    def __init__(self, n_stages, n_realizations, N_VARS, train_data_path, result_path):
        self.n_stages = n_stages
        self.n_realizations = n_realizations
        self.N_VARS = N_VARS
        self.train_data_path = train_data_path
        self.result_path = result_path

    # 给定cuts，计算前向过程的obj
    # def forward_obj_calculate(self, feat, cuts_test, cuts_predicted, recalculate_flag=False):
    #
    #     '''
    #     :param feat: [n_stages-1, n_feats]
    #     :param cuts_test: [n_stages-1, n_pieces, N_VARS-1]
    #     :param cuts_predicted: [n_stages-1, n_pieces, N_VARS-1]
    #     :param recalculate_flag: boolean
    #     :return:
    #     '''
    #
    #     n_pieces = cuts_predicted.shape[1]
    #
    #     # 重算截距？
    #     if recalculate_flag:
    #         if cuts_predicted.size(-1) > self.N_VARS:
    #             cuts_predicted = cuts_predicted[..., :self.N_VARS]
    #         algo = self._get_algo(feat)
    #         cuts_predicted = algo.intercept_recalculate(cuts_predicted)
    #         cuts_predicted = [
    #             [cuts_predicted[(t, k)] for k in range(n_pieces)]
    #             for t in range(self.n_stages - 1)
    #         ]
    #         cuts_predicted = torch.tensor(cuts_predicted).float()
    #
    #     algo = self._get_algo(feat)
    #     # 前向采样个数
    #     samples = algo.sc_sampler.generate_samples(self.fw_n_samples)
    #
    #     # 对照 no cut
    #     algo.cut_add_flag = False
    #     # 检查是否真的没有cut
    #     if not algo.cuts_storage.is_empty():
    #         raise RuntimeError("Expected cuts_storage to be empty, but it is not.")
    #     v_opt_list = algo.forward_pass(0, samples)
    #     obj_nocut = np.mean(np.array(v_opt_list))
    #
    #     # 测试 test cut
    #     algo.cuts_storage = result_Dict("cuts_storage1")
    #     # 检查是否真的没有cut
    #     if not algo.cuts_storage.is_empty():
    #         raise RuntimeError("Expected cuts_storage to be empty, but it is not.")
    #     for t in range(self.n_stages - 1):
    #         for k in range(cuts_test.shape[1]):
    #             # i:随便写 k:  t:指定阶段
    #             algo.cuts_storage.add(0, k, t, cuts_test[t][k].tolist())
    #     algo.cut_add_flag = True
    #     v_opt_list = algo.forward_pass(0, samples)
    #     obj_test = np.mean(np.array(v_opt_list))
    #
    #     # 预测 pred cut
    #     algo.cuts_storage = result_Dict("cuts_storage2")
    #     # 检查是否真的没有cut
    #     if not algo.cuts_storage.is_empty():
    #         raise RuntimeError("Expected cuts_storage to be empty, but it is not.")
    #     for t in range(self.n_stages - 1):
    #         for k in range(cuts_predicted.shape[1]):
    #             # i:随便写 k:  t:指定阶段
    #             algo.cuts_storage.add(0, k, t, cuts_predicted[t][k].tolist())
    #     algo.cut_add_flag = True
    #     v_opt_list = algo.forward_pass(0, samples)
    #     obj_pred = np.mean(np.array(v_opt_list))
    #
    #     return obj_test, obj_pred, obj_nocut
    def get_fw_samples(self, feat, fw_n_samples):
        algo = self._get_algo(feat)
        return algo.sc_sampler.generate_samples(fw_n_samples)

    def forward_obj_calculate(self, feat, samples, cuts=None):
        """obj计算"""
        algo = self._get_algo(feat)
        if cuts is not None:
            # 添加cuts
            for t in range(self.n_stages - 1):
                for k in range(cuts.shape[1]):
                    # i:一定要用0, k和t:对应piece和阶段
                    algo.cuts_storage.add(0, k, t, cuts[t][k].tolist())
            algo.cut_add_flag = True
        else:
            algo.cut_add_flag = False
        obj_list = algo.forward_pass(0, samples)
        return obj_list

    def forward_obj_calculate_stage(self, feat, samples, cuts=None):
        """不同阶段的obj"""
        algo = self._get_algo(feat)
        if cuts is not None:
            # 添加cuts
            for t in range(self.n_stages - 1):
                for k in range(cuts.shape[1]):
                    # i:一定要用0, k和t:对应piece和阶段
                    algo.cuts_storage.add(0, k, t, cuts[t][k].tolist())
            algo.cut_add_flag = True
        else:
            algo.cut_add_flag = False
        obj_list = algo.forward_pass_stage(0, samples)
        return obj_list

    def sddip(self, feat, cuts=None, return_cuts=False):
        """
        使用cuts计算sddip，直到收敛，cuts中可能是pred、pred_re、nocut
        :param feat:
        :param cuts:
        :return:
        """
        infer_sddip_result_path = os.path.join(self.result_path, "sampled_sddip_result")
        file_name = int(feat[0][0])
        algo = self._get_algo(feat)
        if cuts is not None:
            # 添加cuts
            for t in range(self.n_stages - 1):
                for k in range(cuts.shape[1]):
                    # i:一定要用0, k和t:对应piece和阶段
                    algo.cuts_storage.add(0, k, t, cuts[t][k].tolist())
            algo.cut_add_flag = True
        # 返回运行时间以及收敛obj
        if return_cuts:
            sddip_time, obj, cuts_tensor = algo.run_time(file_name, infer_sddip_result_path, return_cuts)
            return sddip_time, obj, cuts_tensor
        else:
            sddip_time, obj = algo.run_time(file_name, infer_sddip_result_path, return_cuts)
            return sddip_time, obj

    def sddip_fw_n_samples(self, feat, cuts, fw_n_samples):
        """
        使用cuts计算sddip，直到收敛，cuts中可能是pred、pred_re、nocut，多次前向计算平均obj
        :param feat:
        :param cuts:
        :return:
        """
        file_name = int(feat[0][0])
        algo = self._get_algo(feat)
        if cuts is not None:
            # 添加cuts
            for t in range(self.n_stages - 1):
                for k in range(cuts.shape[1]):
                    # i:一定要用0, k和t:对应piece和阶段
                    algo.cuts_storage.add(0, k, t, cuts[t][k].tolist())
            algo.cut_add_flag = True
        # 返回运行时间以及收敛obj
        time_list, obj_list, LB_list = algo.run_sddip_fw_n_samples(fw_n_samples)
        return time_list, obj_list, LB_list

    def sddip_n_lag(self, feat, cuts, max_lag, logger=None):
        algo = self._get_algo(feat)
        if cuts is not None:
            # 添加cuts
            for t in range(self.n_stages - 1):
                for k in range(cuts.shape[1]):
                    # i:一定要用0, k和t:对应piece和阶段
                    algo.cuts_storage.add(0, k, t, cuts[t][k].tolist())
            algo.cut_add_flag = True
        lag_time_list, lag_obj_list, lag_cuts_list = algo.run_n_lag(max_lag, logger)
        return lag_time_list, lag_obj_list, lag_cuts_list


    def intercept_recalculate(self, feat, cuts_predicted, logger=None):
        n_pieces = cuts_predicted.shape[1]
        # cuts_predicted中包含截距，需要截掉
        if cuts_predicted.shape[-1] > self.N_VARS:
            cuts_predicted = cuts_predicted[..., :self.N_VARS]
        algo = self._get_algo(feat)
        start = time.time()
        cuts_predicted = algo.intercept_recalculate(cuts_predicted, logger)
        cuts_predicted_re = [
            [cuts_predicted[(t, k)] for k in range(n_pieces)]
            for t in range(self.n_stages - 1)
        ]
        cuts_predicted_re = torch.tensor(cuts_predicted_re).float()
        recalculate_time = time.time() - start
        return recalculate_time, cuts_predicted_re

    def calculate_x(self, feat, num_x):

        algo = self._get_algo(feat)

        return algo.get_x_with_nocut(num_x = num_x)



    def _get_algo(self, feat):
        logger = logging.getLogger(__name__)
        if self.N_VARS < 20:
            test_case = "case6ww"
        else:
            test_case = "case118"
        init_n_binaries = 10
        time_limit_minutes = 5 * 60
        refinement_stabilization_count = 1
        # Setup
        algo = sddipclassical.Algorithm(
            test_case,
            self.n_stages,
            self.n_realizations,
            logger,
            self.train_data_path
        )
        algo.n_binaries = init_n_binaries
        algo.n_samples_primary = 3
        algo.n_samples_secondary = 1
        algo.time_limit_minutes = time_limit_minutes
        algo.refinement_stabilization_count = refinement_stabilization_count
        algo.n_samples_final_ub = 300
        instance_index = int(feat[0][0])

        if instance_index == 0:
            print("instance_index==0")
            print(feat)

        scenario_dir = os.path.join(self.train_data_path, "scenarios", f"{instance_index}_scenario.csv")

        algo.init(scenario_dir=scenario_dir)

        return algo