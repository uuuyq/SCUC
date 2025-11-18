import logging
import os

import numpy
import numpy as np
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

    def get_fw_samples(self, feat, fw_n_samples):
        algo = self._get_algo(feat)
        return algo.sc_sampler.generate_samples(fw_n_samples)

    def forward_obj_calculate(self, feat: np.ndarray, samples, cuts: np.ndarray):
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

    def forward_obj_calculate_stage(self, feat, samples, cuts: np.ndarray):
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


    def sddip_fw_n_samples(self, feat, cuts, fw_n_samples, max_iterations, logger):
        """
        计算sddip，可提供初始化的cuts
        返回每次迭代的time、obj、LB
        obj根据参数fw_n_samples进行多次计算

        """
        algo = self._get_algo(feat)
        if cuts is not None:
            # 添加cuts
            for t in range(self.n_stages - 1):
                for k in range(cuts.shape[1]):
                    # i:一定要用0, k和t:对应piece和阶段
                    algo.cuts_storage.add(0, k, t, cuts[t][k].tolist())
            algo.cut_add_flag = True
        # 返回运行时间以及收敛obj
        time_list, obj_list, LB_list = algo.run_sddip_fw_n_samples(fw_n_samples, max_iterations, logger)
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
        cuts_predicted_re = np.asarray(cuts_predicted_re, dtype=float)
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