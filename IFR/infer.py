import logging
import os
import torch
import time

from sddip_script_update import sddipclassical


class Infer:

    def __init__(self, n_stages, n_realizations, N_VARS, train_data_path, result_path):
        self.n_stages = n_stages
        self.n_realizations = n_realizations
        self.N_VARS = N_VARS
        self.train_data_path = train_data_path
        self.result_path = result_path

    def get_fw_samples(self, instance_index, fw_n_samples):
        algo = self._get_algo(instance_index)
        return algo.sc_sampler.generate_samples(fw_n_samples)

    def forward_obj_calculate(self, instance_index, samples, cuts=None):
        """obj计算"""
        algo = self._get_algo(instance_index)
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


    def _get_algo(self, instance_index):
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


        scenario_dir = os.path.join(self.train_data_path, "scenarios", f"{instance_index}_scenario.csv")

        algo.init(scenario_dir=scenario_dir)

        return algo