import datetime
import math
import time
from statistics import mean
import gurobipy as gp
import numpy as np
from scipy import stats

# from NN_torch_24.compare_plot1 import time_result
from . import scenarios, ucmodelclassical, parameters, outermodel, dualsolver
from .result_storage import result_Dict
import os
import matplotlib.pyplot as plt



class Algorithm:
    def __init__(
        self,
        test_case: str,
        n_stages: int,
        n_realizations: int,
        logger,
        train_data_path
    ):
        self.test_case = test_case
        self.n_stages = n_stages
        self.n_realizations = n_realizations
        self.logger = logger
        self.train_data_path = train_data_path
        # Algorithm paramters
        # self.n_binaries = 10 #近似二元整数的位数
        # self.error_threshold = 10 ** (-1)
        # self.max_n_binaries = 10
        # Absolute change in lower bound
        self.refinement_tolerance = 10 ** (-8)

        self.no_improvement_tolerance = 10 ** (-8)

        self.relative_tolerance = 2e-5 # 相对误差
        self.stop_stabilization_count = 3

        self.refinement_stabilization_count = 2
        self.big_m = 10**6
        self.time_limit_minutes = 5 * 60
        self.n_samples_final_ub = 150
        self.bin_multipliers = None
        self.length_z = None
        # iter-stage-k
        self.cut_add_flag = False
        self.forward_result = result_Dict("forward_result")
        self.cuts_storage = result_Dict("cuts_storage")
        self.x_storage = result_Dict("x_storage")

        self.dual_solver = dualsolver.BundleMethod(
            5000,
            10 ** -6,
            log_dir="",
            predicted_ascent="abs",
            time_limit=5 * 60,
        )
        self.primary_cut_mode = 1
        self.secondary_cut_mode = 2

        self.current_cut_mode = self.primary_cut_mode


    # 初始化
    def init(self,scenario_dir, scenario_df=None):  # 前向推理时直接提供scenario_df
        # Problem specific 04_scenarios
        if scenario_df is not None:
            self.problem_params = parameters.Parameters(
                self.test_case, self.n_stages, self.n_realizations, scenario_dir=scenario_dir, scenario_df=scenario_df
            )
        else:
            self.problem_params = parameters.Parameters(
                self.test_case, self.n_stages, self.n_realizations, scenario_dir
            )

        self.length_z = self.problem_params.n_gens * 2 + sum(self.problem_params.backsight_periods) + self.problem_params.n_storages

        self.sc_sampler = scenarios.ScenarioSampler(
            self.problem_params.n_stages,
            self.problem_params.n_realizations_per_stage[1],
        )
        self.n_samples_primary = 3  # benders前向过程采样的场景路径数
        self.n_samples_secondary = 1  # lag采样的场景路径数

    def run(self, file_name):
        self.logger.info("#### SDDiP-Algorithm started ####")
        stop_stabilization_count = 3

        LB_list = []
        UB_list = []
        obj_list = []

        start_time = time.time()

        i = 0  # 全局迭代计数器
        # SB阶段，最多执行20次
        sb_count = 0
        while sb_count < 5:
            v_lower, obj_mean, v_upper_l, v_upper_r = self.SB_iteration(i)
            obj_list.append(obj_mean)
            LB_list.append(v_lower)
            UB_list.append(v_upper_l)
            if len(LB_list) >= stop_stabilization_count + 1 and \
                    (LB_list[-1] - LB_list[-(stop_stabilization_count + 1)]) / (
                    abs(LB_list[-(stop_stabilization_count + 1)]) + 1e-8) < self.relative_tolerance * 100:
                break
            i += 1
            sb_count += 1
        # 总共执行不超过30次
        while i < 20:
            # 优先使用SB
            v_lower, obj_mean, v_upper_l, v_upper_r = self.SB_lag_iteration(LB_list, i)
            obj_list.append(obj_mean)
            LB_list.append(v_lower)
            UB_list.append(v_upper_l)
            if len(LB_list) >= stop_stabilization_count + 1 and \
                    (LB_list[-1] - LB_list[-(stop_stabilization_count + 1)]) / (
                    abs(LB_list[-(stop_stabilization_count + 1)]) + 1e-8) < self.relative_tolerance:
                break
            i += 1
        run_time = time.time() - start_time
        self.logger.info("#### SDDiP-Algorithm finished ####")

        time_path = os.path.join(self.train_data_path, "time.txt")
        with open(time_path, "a") as f:
            f.write(f"{file_name} : {run_time} seconds\n")

        self.cuts_storage.save_cut(file_name, self.train_data_path)
        self.x_storage.save_cut(file_name.split("_")[0]+"_x", self.train_data_path)


        # 1. 绘制 LB_list 并保存
        plt.figure()
        x_values = list(range(1, len(LB_list) + 1))
        plt.plot(x_values, LB_list, marker='o', linestyle='-', color='b', label="LB_UB")
        plt.xlabel("Iteration")
        plt.ylabel("Lower Bound (LB)")
        plt.title("LB_UB Progress")
        plt.legend()
        plt.grid(True)
        lb_fig_path = os.path.join(self.train_data_path, "fig", f"{file_name}_LB.png")
        plt.savefig(lb_fig_path)
        plt.close()

        # 2. 绘制 obj_list 并保存
        plt.figure()
        x_values = list(range(1, len(obj_list) + 1))
        plt.plot(x_values, obj_list, marker='s', linestyle='--', color='g', label="Obj")
        plt.xlabel("Iteration")
        plt.ylabel("Objective Value")
        plt.title("Objective Progress")
        plt.legend()
        plt.grid(True)
        obj_fig_path = os.path.join(self.train_data_path, "fig", f"{file_name}_obj.png")
        plt.savefig(obj_fig_path)
        plt.close()

        np.savetxt(os.path.join(self.train_data_path, "LB_obj_list", f"{file_name}_LB.txt"), LB_list)
        np.savetxt(os.path.join(self.train_data_path, "LB_obj_list", f"{file_name}_obj.txt"), obj_list)


    def get_all_cuts_tensor(self, n_iterations):
        """
        收集所有的cuts，整理为tensor shape[t, n, d] n是将k和i合在一起
        :param n_iterations: 迭代次数，用i作为输入就行
        :return:
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence

        cuts_t_list = []
        for t in range(self.n_stages - 1):
            cuts_t = self.cuts_storage.get_values_by_t(t)
            cuts_t_list.append(cuts_t)

        # 先将每个 cuts_t 变为 tensor
        cuts_tensor_list = []
        for cuts_t in cuts_t_list:
            cuts_tensor = torch.tensor(cuts_t, dtype=torch.float)
            cuts_tensor_list.append(cuts_tensor)

        # pad 到相同数量的 cuts（在 dim=0），长度不足的 pad 0
        cuts_padded = pad_sequence(cuts_tensor_list, batch_first=True)  # shape: (n_stage-1, max_num_cuts, cut_dim)
        return cuts_padded

    def run_n_lag(self, max_lag):
        self.logger.info("#### SDDiP-Algorithm started ####")
        self.n_samples = self.n_samples_primary

        obj_list = []

        # pred sddip返回值
        lag_time_list = []
        lag_obj_list = []
        lag_cuts_list = []

        start_time = time.time()
        for i in range(max_lag):
            v_lower, obj_mean, v_upper_l, v_upper_r = self.lag_iteration(i)  # iteration中cut从i=1开始
            obj_list.append(obj_mean)

            lag_obj_list.append(obj_mean)
            lag_time_list.append(time.time() - start_time)
            cuts_tensor = self.get_all_cuts_tensor(n_iterations=i + 1)
            lag_cuts_list.append(cuts_tensor)

        self.logger.info("#### SDDiP-Algorithm finished ####")

        return lag_time_list, lag_obj_list, lag_cuts_list

    def run_sddip_fw_n_samples(self, fw_n_samples):
        '''
        执行sddip，记录时间和obj，LB也记录下吧，注意前向需要多次
        :return: obj_list, time_list
        '''
        print("#### SDDiP-Algorithm started ####")
        self.n_samples = self.n_samples_primary
        stop_stabilization_count = 3

        time_list = []
        LB_list = []
        obj_list = []

        time_cost = 0
        # 初始记录
        obj_mean, v_upper_l, v_upper_r = self.run_statistical(fw_n_samples)
        time_list.append(time_cost)
        obj_list.append(obj_mean)
        v_lower = self.lower_bound(0)
        LB_list.append(v_lower)

        i = 0  # 全局迭代计数器
        # SB阶段，最多执行20次
        # sb_count = 0
        # while sb_count < 10:
        #     start_time = time.time()
        #     self.SB_iteration(i)
        #     end_time = time.time()
        #     time_cost += end_time - start_time
        #
        #     print(f"SB iteration {i}")
        #     print(f"time_list: {time_list}")
        #     print(f"obj_list: {obj_list}")
        #     print(f"LB_list: {LB_list}")
        #
        #     # 统计obj
        #     obj_mean, v_upper_l, v_upper_r = self.run_statistical(fw_n_samples)
        #     time_list.append(time_cost)
        #     obj_list.append(obj_mean)
        #     v_lower = self.lower_bound(0)
        #     LB_list.append(v_lower)
        #
        #
        #     if len(LB_list) >= stop_stabilization_count + 1 and \
        #             (LB_list[-1] - LB_list[-(stop_stabilization_count + 1)]) / (
        #             abs(LB_list[-(stop_stabilization_count + 1)]) + 1e-8) < self.relative_tolerance:
        #         break
        #     i += 1
        #     sb_count += 1

        # 总共执行40次
        while i < 40:
            start_time = time.time()
            self.SB_lag_iteration(LB_list, i)
            end_time = time.time()
            time_cost += end_time - start_time

            print(f"SB-1 or lag-2 choose:{self.current_cut_mode}  iteration:{i} ")
            print(f"time_list: {time_list}")
            print(f"obj_list: {obj_list}")
            print(f"LB_list: {LB_list}")


            # 统计obj
            obj_mean, v_upper_l, v_upper_r = self.run_statistical(fw_n_samples)
            time_list.append(time_cost)
            obj_list.append(obj_mean)
            v_lower = self.lower_bound(0)
            LB_list.append(v_lower)

            if len(LB_list) >= stop_stabilization_count + 1 and \
                    (LB_list[-1] - LB_list[-(stop_stabilization_count + 1)]) / (abs(LB_list[-(stop_stabilization_count + 1)]) + 1e-8) < self.relative_tolerance:
                break
            i += 1
        print("#### SDDiP-Algorithm finished ####")

        return time_list, obj_list, LB_list


    def run_sddip_statistical(self, index, n_samples_statistical):
        self.logger.info("#### SDDiP-Algorithm started ####")
        self.n_samples = self.n_samples_primary
        stop_stabilization_count = 3


        result = []
        Obj_time_list = []

        LB_list = []


        start_time = time.time()
        i = 0  # 全局迭代计数器

        # SB阶段，最多执行20次

        sb_count = 0
        # while sb_count < 20:
        #     start_time = time.time()
        #     obj_mean, v_upper_l, v_upper_r = self.run_statistical(n_samples_statistical)
        #     self.logger.info(f"SB iteration {i} Forward statistical time: {time.time() - start_time}")
        #
        #     start_time = time.time()
        #     self.SB_iteration(i)
        #     self.logger.info(f"SB iteration {i} time: {time.time() - start_time}")
        #     v_lower = self.lower_bound(0)
        #     LB_list.append(v_lower)
        #     result.append((obj_mean, v_lower, v_upper_l, v_upper_r))
        #     if len(LB_list) >= stop_stabilization_count + 1 and \
        #             (LB_list[-1] - LB_list[-(stop_stabilization_count + 1)]) / (
        #             abs(LB_list[-(stop_stabilization_count + 1)]) + 1e-8) < self.relative_tolerance:
        #         break
        #     i += 1
        #     sb_count += 1

        # 总共执行50次
        lag_count = 0
        while i < 50:
            start_time = time.time()
            obj_mean, v_upper_l, v_upper_r = self.run_statistical(n_samples_statistical)
            self.logger.info(f"lag iteration {i} Forward statistical time: {time.time() - start_time}")

            start_time = time.time()
            self.lag_iteration(i)
            self.logger.info(f"lag iteration {i} time: {time.time() - start_time}")
            v_lower = self.lower_bound(0)
            LB_list.append(v_lower)
            result.append((obj_mean, v_lower, v_upper_l, v_upper_r))
            # if len(LB_list) >= stop_stabilization_count + 1 and \
            #         (LB_list[-1] - LB_list[-(stop_stabilization_count + 1)]) / (abs(LB_list[-(stop_stabilization_count + 1)]) + 1e-8) < self.relative_tolerance:
            #     break
            i += 1
            lag_count += 1
        self.logger.info("#### SDDiP-Algorithm finished ####")
        t = time.strftime("%m-%d-%H")
        LB_UB_path = os.path.join(self.train_data_path, "LB_UB", f"{index}_LB_UB_{t}.csv")
        import csv
        # 写入 CSV 文件
        with open(LB_UB_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            # 写表头
            writer.writerow(["UB_mean", "LB", "CI_lower", "CI_upper"])
            # 写数据
            for obj_mean, v_lower, v_upper_l, v_upper_r in result:
                writer.writerow([f"{obj_mean:.6f}", f"{v_lower:.6f}", f"{v_upper_l:.6f}", f"{v_upper_r:.6f}"])


    def run_statistical(self, n_samples_statistical):
        v_opt_k = self.forward_pass_statistical(0, n_samples_statistical)
        v_upper_l, v_upper_r = self.statistical_upper_bound(
            v_opt_k, n_samples_statistical
        )

        return mean(v_opt_k), v_upper_l, v_upper_r


    # TODO: SB和lag混合迭代方式

    def SB_lag_iteration(self, LB_list, index):
        # 根据LB_list的情况，选择SB或lag迭代
        no_improvement_condition = False
        if len(LB_list) > 1:
            no_improvement_condition = (LB_list[-1] - LB_list[-2]) / (abs(LB_list[-2]) + 1e-8) <= self.relative_tolerance * 100
            # print(f"RE: {(LB_list[-1] - LB_list[-2]) / (abs(LB_list[-2]) + 1e-8)}  <= {self.relative_tolerance * 100}")
        if self.current_cut_mode == self.secondary_cut_mode:
            self.current_cut_mode = self.primary_cut_mode
        elif no_improvement_condition:
            self.current_cut_mode = self.secondary_cut_mode

        if self.current_cut_mode == self.primary_cut_mode:
            return self.SB_iteration(index)
        elif self.current_cut_mode == self.secondary_cut_mode:
            return self.lag_iteration(index)
        else:
            raise ValueError("self.current_cut_mode出错")



    def SB_iteration(self, i):
        # SB
        UB = float('inf')
        n_samples = self.n_samples_primary
        samples = self.sc_sampler.generate_samples(
            n_samples
        )  # samples：[[1,2,5,3][]] len(samples)代表前向步骤执行次数（采样次数）,每个其中数字代表realization序号
        self.logger.info("Samples: %s", samples)

        # Forward pass
        self.logger.info("Forward pass")
        v_opt_k = self.forward_pass(i, samples)
        # Statistical upper bound
        v_upper_l, v_upper_r = self.statistical_upper_bound(
            v_opt_k, n_samples
        )

        # Backward pass
        self.logger.info("Backward benders pass")
        self.backward_benders(i + 1, samples)

        # Lower bound
        v_lower = self.lower_bound(i + 1)
        self.logger.info(f"benders iteration {i} LB:{v_lower} UB:{v_upper_l}")
        return v_lower, mean(v_opt_k), v_upper_l, v_upper_r


    def lag_iteration(self, i):

        n_samples = self.n_samples_secondary
        samples = self.sc_sampler.generate_samples(n_samples)
        self.logger.info("Samples: %s", samples)

        self.logger.info("Forward pass")
        v_opt_k = self.forward_pass(i, samples)
        v_upper_l, v_upper_r = self.statistical_upper_bound(
            v_opt_k, n_samples
        )
        self.logger.info("Backward lag pass")
        self.backward_pass_norm(i + 1, samples)
        v_lower = self.lower_bound(i + 1)
        self.logger.info(f"lag iteration {i} LB:{v_lower} UB:{v_upper_l}")

        return v_lower, mean(v_opt_k), v_upper_l, v_upper_r

    def forward_pass(self, iteration: int, samples: list) -> list:
        i = iteration  # 从0开始
        n_samples = len(samples)
        v_opt_k = []

        for k in range(n_samples):
            # t = -1对应的状态值
            x_trial_point = self.problem_params.init_x_trial_point
            y_trial_point = self.problem_params.init_y_trial_point
            x_bs_trial_point = self.problem_params.init_x_bs_trial_point
            soc_trial_point = self.problem_params.init_soc_trial_point

            v_opt_k.append(0)
            #使用samples的方式进一步使每次迭代中的前向路径都不同
            for t, n in zip(range(self.problem_params.n_stages), samples[k]):

                # Create forward model
                # 模型定义中添加了问题所需要的变量，x(前向过程中x为二元变量)，z
                uc_fw = ucmodelclassical.ClassicalModel(
                    self.problem_params.n_buses,
                    self.problem_params.n_lines,
                    self.problem_params.n_gens,
                    self.problem_params.n_storages,
                    self.problem_params.gens_at_bus,
                    self.problem_params.storages_at_bus,
                    self.problem_params.backsight_periods,
                    lp_relax=False
                )

                uc_fw: ucmodelclassical.ClassicalModel = (
                    # 问题约束中没有Xa(n),转换为z与Xn的约束
                    self.add_problem_constraints(uc_fw, t, n)
                )
                # 目标函数
                uc_fw.add_objective(self.problem_params.cost_coeffs)
                '''
                下边就是计算z与trial_point的差值relaxed_terms，前向过程需要relaxed_terms=0
                '''
                uc_fw.calculate_relaxed_terms(
                    x_trial_point,
                    y_trial_point,
                    x_bs_trial_point,
                    soc_trial_point,
                )
                # 置0
                uc_fw.zero_relaxed_terms()

                # Solve problem
                uc_fw.disable_output()
                uc_fw.model.optimize()
                if uc_fw.model.status != 2:
                    self.logger.info(f"model.status {uc_fw.model.status}")

                try:
                    # 获取结果
                    x_kt = [x_g.x for x_g in uc_fw.x]
                    y_kt = [y_g.x for y_g in uc_fw.y]
                    x_bs_kt = [
                        [x_bs.x for x_bs in x_bs_g] for x_bs_g in uc_fw.x_bs
                    ]
                    soc_kt = [soc_s.x for soc_s in uc_fw.soc]
                except AttributeError:
                    print("uc_fw.model 求解错误")
                    uc_fw.model.write("model.lp")
                    uc_fw.model.computeIIS()
                    uc_fw.model.write("model.ilp")
                    raise


                # Value of stage t objective function
                v_value_function = uc_fw.model.getObjective().getValue()
                v_opt_kt = v_value_function - uc_fw.theta.x
                v_opt_k[-1] += v_opt_kt

                # theta_hat = theta_{t-1}=obj_t
                # theta_value = v_value_function
                theta_value = uc_fw.theta.x

                x_trial_point = x_kt
                y_trial_point = y_kt
                if any(x_bs_kt):
                    # 更新bs
                    x_bs_trial_point = [
                        [x_trial_point[g]] + x_bs_kt[g][:-1]
                        for g in range(self.problem_params.n_gens)
                    ]
                soc_trial_point = soc_kt

                # x_bs_trial_point_one_dim = [item for sublist in x_bs_trial_point for item in sublist]
                # [X + theta]
                stage_result = []
                stage_result.append(x_kt)
                stage_result.append(y_kt)
                stage_result.append(x_bs_trial_point)
                stage_result.append(soc_kt)
                stage_result.append(theta_value)
                self.forward_result.add(i, k, t, stage_result)
        return v_opt_k


    # 返回不同阶段的obj
    def forward_pass_stage(self, iteration: int, samples: list) -> list:
        i = iteration  # 从0开始
        n_samples = len(samples)
        v_opt_k = []

        obj_stage = []  # shape: n_samples , n_stages
        for k in range(n_samples):
            obj_stage_k = []
            # t = -1对应的状态值
            x_trial_point = self.problem_params.init_x_trial_point
            y_trial_point = self.problem_params.init_y_trial_point
            x_bs_trial_point = self.problem_params.init_x_bs_trial_point
            soc_trial_point = self.problem_params.init_soc_trial_point

            v_opt_k.append(0)
            #使用samples的方式进一步使每次迭代中的前向路径都不同
            for t, n in zip(range(self.problem_params.n_stages), samples[k]):

                # Create forward model
                # 模型定义中添加了问题所需要的变量，x(前向过程中x为二元变量)，z
                uc_fw = ucmodelclassical.ClassicalModel(
                    self.problem_params.n_buses,
                    self.problem_params.n_lines,
                    self.problem_params.n_gens,
                    self.problem_params.n_storages,
                    self.problem_params.gens_at_bus,
                    self.problem_params.storages_at_bus,
                    self.problem_params.backsight_periods,
                    lp_relax=False
                )

                uc_fw: ucmodelclassical.ClassicalModel = (
                    # 问题约束中没有Xa(n),转换为z与Xn的约束
                    self.add_problem_constraints(uc_fw, t, n)
                )
                # 目标函数
                uc_fw.add_objective(self.problem_params.cost_coeffs)
                '''
                下边就是计算z与trial_point的差值relaxed_terms，前向过程需要relaxed_terms=0
                '''
                uc_fw.calculate_relaxed_terms(
                    x_trial_point,
                    y_trial_point,
                    x_bs_trial_point,
                    soc_trial_point,
                )
                # 置0
                uc_fw.zero_relaxed_terms()

                # Solve problem
                uc_fw.disable_output()
                uc_fw.model.optimize()
                if uc_fw.model.status != 2:
                    self.logger.info(f"model.status {uc_fw.model.status}")

                try:
                    # 获取结果
                    x_kt = [x_g.x for x_g in uc_fw.x]
                    y_kt = [y_g.x for y_g in uc_fw.y]
                    x_bs_kt = [
                        [x_bs.x for x_bs in x_bs_g] for x_bs_g in uc_fw.x_bs
                    ]
                    soc_kt = [soc_s.x for soc_s in uc_fw.soc]
                except AttributeError:
                    print("uc_fw.model 求解错误")
                    uc_fw.model.write("model.lp")
                    uc_fw.model.computeIIS()
                    uc_fw.model.write("model.ilp")
                    raise


                # Value of stage t objective function
                v_value_function = uc_fw.model.getObjective().getValue()
                v_opt_kt = v_value_function - uc_fw.theta.x
                # 每个阶段的 obj
                obj_stage_k.append(v_opt_kt)
                v_opt_k[-1] += v_opt_kt

                # theta_hat = theta_{t-1}=obj_t
                # theta_value = v_value_function
                theta_value = uc_fw.theta.x

                x_trial_point = x_kt
                y_trial_point = y_kt
                if any(x_bs_kt):
                    # 更新bs
                    x_bs_trial_point = [
                        [x_trial_point[g]] + x_bs_kt[g][:-1]
                        for g in range(self.problem_params.n_gens)
                    ]
                soc_trial_point = soc_kt

                # x_bs_trial_point_one_dim = [item for sublist in x_bs_trial_point for item in sublist]
                # [X + theta]
                stage_result = []
                stage_result.append(x_kt)
                stage_result.append(y_kt)
                stage_result.append(x_bs_trial_point)
                stage_result.append(soc_kt)
                stage_result.append(theta_value)
                self.forward_result.add(i, k, t, stage_result)

            obj_stage.append(obj_stage_k)

        return obj_stage


    # 多个采样路径评估上界，不保存trial_points
    def forward_pass_statistical(self, iteration: int, n_samples: int) -> list:
        i = iteration  # 从0开始
        samples = self.sc_sampler.generate_samples(n_samples)
        v_opt_k = []

        for k in range(n_samples):
            # t = -1对应的状态值
            x_trial_point = self.problem_params.init_x_trial_point
            y_trial_point = self.problem_params.init_y_trial_point
            x_bs_trial_point = self.problem_params.init_x_bs_trial_point
            soc_trial_point = self.problem_params.init_soc_trial_point

            v_opt_k.append(0)
            #使用samples的方式进一步使每次迭代中的前向路径都不同
            for t, n in zip(range(self.problem_params.n_stages), samples[k]):

                # Create forward model
                # 模型定义中添加了问题所需要的变量，x(前向过程中x为二元变量)，z
                uc_fw = ucmodelclassical.ClassicalModel(
                    self.problem_params.n_buses,
                    self.problem_params.n_lines,
                    self.problem_params.n_gens,
                    self.problem_params.n_storages,
                    self.problem_params.gens_at_bus,
                    self.problem_params.storages_at_bus,
                    self.problem_params.backsight_periods,
                    lp_relax=False
                )

                uc_fw: ucmodelclassical.ClassicalModel = (
                    # 问题约束中没有Xa(n),转换为z与Xn的约束
                    self.add_problem_constraints(uc_fw, t, n)
                )
                # 目标函数
                uc_fw.add_objective(self.problem_params.cost_coeffs)
                '''
                下边就是计算z与trial_point的差值relaxed_terms，前向过程需要relaxed_terms=0
                '''
                uc_fw.calculate_relaxed_terms(
                    x_trial_point,
                    y_trial_point,
                    x_bs_trial_point,
                    soc_trial_point,
                )
                # 置0
                uc_fw.zero_relaxed_terms()

                # Solve problem
                uc_fw.disable_output()
                uc_fw.model.optimize()
                if uc_fw.model.status != 2:
                    self.logger.info(f"model.status {uc_fw.model.status}")

                try:
                    # 获取结果
                    x_kt = [x_g.x for x_g in uc_fw.x]
                    y_kt = [y_g.x for y_g in uc_fw.y]
                    x_bs_kt = [
                        [x_bs.x for x_bs in x_bs_g] for x_bs_g in uc_fw.x_bs
                    ]
                    soc_kt = [soc_s.x for soc_s in uc_fw.soc]
                except AttributeError:
                    print("uc_fw.model 求解错误")
                    uc_fw.model.write("model.lp")
                    uc_fw.model.computeIIS()
                    uc_fw.model.write("model.ilp")
                    raise


                # Value of stage t objective function
                v_value_function = uc_fw.model.getObjective().getValue()
                v_opt_kt = v_value_function - uc_fw.theta.x
                v_opt_k[-1] += v_opt_kt
                theta_value = uc_fw.theta.x

                x_trial_point = x_kt
                y_trial_point = y_kt
                if any(x_bs_kt):
                    # 更新bs
                    x_bs_trial_point = [
                        [x_trial_point[g]] + x_bs_kt[g][:-1]
                        for g in range(self.problem_params.n_gens)
                    ]
                soc_trial_point = soc_kt

        return v_opt_k

    # lag zou
    def backward_pass(self, iteration: int, samples: list) -> None:
        i = iteration
        n_samples = len(samples)

        for t in reversed(range(1, self.problem_params.n_stages)):
            for k in range(n_samples):
                n_realizations = self.problem_params.n_realizations_per_stage[
                    t
                ]
                # Get trial points
                stage_result = self.forward_result.get_values(i - 1, k, t - 1)
                x_trial_point = stage_result[0]
                y_trial_point = stage_result[1]
                x_bs_trial_point = stage_result[2]
                soc_trial_point = stage_result[3]

                trial_point = (
                        x_trial_point
                        + y_trial_point
                        + [
                            x_bs_g
                            for x_bs in x_bs_trial_point
                            for x_bs_g in x_bs
                        ]
                        + soc_trial_point
                )

                lag_cuts_list = []
                for n in range(n_realizations):

                    # Build backward model
                    uc_bw = ucmodelclassical.ClassicalModel(
                        self.problem_params.n_buses,
                        self.problem_params.n_lines,
                        self.problem_params.n_gens,
                        self.problem_params.n_storages,
                        self.problem_params.gens_at_bus,
                        self.problem_params.storages_at_bus,
                        self.problem_params.backsight_periods,
                    )

                    # uc_bw.binary_approximation(
                    #     self.bin_multipliers["y"], self.bin_multipliers["soc"]
                    # )

                    uc_bw: ucmodelclassical.ClassicalModel = (
                        self.add_problem_constraints(uc_bw, t, n)
                    )
                    uc_bw.add_objective(self.problem_params.cost_coeffs)

                    uc_bw.calculate_relaxed_terms(
                        x_trial_point,
                        y_trial_point,
                        x_bs_trial_point,
                        soc_trial_point,
                    )

                    objective_terms = uc_bw.objective_terms
                    relaxed_terms = uc_bw.relaxed_terms

                    # if t == 1 and n == 1:
                    #     logger.info(relaxed_terms)
                    #     uc_bw.model.write("model.lp")

                    uc_bw.disable_output()

                    _, sg_results = self.dual_solver.solve(
                        uc_bw.model,
                        objective_terms,
                        relaxed_terms,
                    )
                    dual_multipliers = sg_results.multipliers.tolist()
                    dual_value = sg_results.obj_value - np.array(
                        dual_multipliers
                    ).dot(trial_point)

                    lag_cuts_list.append(
                        dual_multipliers + [dual_value]
                    )

                    # Dual value and multiplier for each realization

                lag_cuts_list = np.mean(np.array(lag_cuts_list), axis=0).tolist()
                # print("lag_average", lag_cuts_list)
                self.cuts_storage.add(i, k, t - 1, lag_cuts_list)
                self.x_storage.add(i, k, t - 1, trial_point)
        self.cut_add_flag = True



    # lag normalized
    def backward_pass_norm(self, iteration: int, samples: list):
        i = iteration  # 从1开始
        n_samples = len(samples)

        for t in reversed(range(1, self.problem_params.n_stages)):

            for k in range(n_samples):
                n_realizations = self.problem_params.n_realizations_per_stage[
                    t
                ]

                # Get trial points
                stage_result = self.forward_result.get_values(i - 1, k, t - 1)
                x_trial_point = stage_result[0]
                y_trial_point = stage_result[1]
                x_bs_trial_point = stage_result[2]
                soc_trial_point = stage_result[3]
                theta_trial = stage_result[4]

                X_trial = (
                        x_trial_point
                        + y_trial_point
                        + [val for bs in x_bs_trial_point for val in bs]
                        + soc_trial_point
                )

                lag_cuts_list = []
                for n in range(n_realizations):
                    inner_model = self.create_inner_model(t, n, False)
                    outer_model = self.create_outer_model(X_trial, theta_trial, alpha=1)
                    # level bundle methods
                    pi_star, pi0_star, flag = self.level_bundle_methods(inner_model, outer_model, X_trial,
                                                                        theta_trial, 100)
                    if pi0_star == None or pi0_star <= 1e-6 or not flag:
                        continue

                    # cut: pi * x + pi0 * theta >= inner_model_obj + pi * x_hat + pi0 * theta_hat
                    inner_model.set_inner_objective(self.problem_params.cost_coeffs, pi_star, pi0_star)
                    inner_model.model.optimize()
                    intercept = inner_model.model.getObjective().getValue()
                    pi = -pi_star / pi0_star
                    intercept = intercept / pi0_star
                    lag_cuts_list.append(list(pi) + [intercept])
                # Calculate and store cut coefficients
                if len(lag_cuts_list) > 1:
                    lag_cuts_list = np.mean(np.array(lag_cuts_list), axis=0).tolist()
                    # print("lag_average", lag_cuts_list)
                    self.cuts_storage.add(i, k, t - 1, lag_cuts_list)
                    self.x_storage.add(i, k, t - 1, X_trial)
        self.cut_add_flag = True
        return


    def backward_benders(self, iteration: int, samples: list) -> None:
        i = iteration
        n_samples = len(samples)
        for t in reversed(range(1, self.problem_params.n_stages)):
            for k in range(n_samples):
                n_realizations = self.problem_params.n_realizations_per_stage[
                    t
                ]
                # Get trial points
                stage_result = self.forward_result.get_values(i - 1, k, t - 1)
                x_trial_point = stage_result[0]
                y_trial_point = stage_result[1]
                x_bs_trial_point = stage_result[2]
                soc_trial_point = stage_result[3]

                trial_point = (
                    x_trial_point
                    + y_trial_point
                    + [val for bs in x_bs_trial_point for val in bs]
                    + soc_trial_point
                )

                dual_multipliers = []
                opt_values = []

                for n in range(n_realizations):
                    # Create forward model
                    uc_fw = ucmodelclassical.ClassicalModel(
                        self.problem_params.n_buses,
                        self.problem_params.n_lines,
                        self.problem_params.n_gens,
                        self.problem_params.n_storages,
                        self.problem_params.gens_at_bus,
                        self.problem_params.storages_at_bus,
                        self.problem_params.backsight_periods,
                        lp_relax=True,
                    )

                    uc_fw: ucmodelclassical.ClassicalModel = (
                        self.add_problem_constraints(uc_fw, t, n)
                    )

                    uc_fw.add_objective(self.problem_params.cost_coeffs)

                    uc_fw.calculate_relaxed_terms(
                        x_trial_point,
                        y_trial_point,
                        x_bs_trial_point,
                        soc_trial_point,
                    )
                    uc_fw.zero_relaxed_terms()

                    uc_fw.model.optimize()

                    copy_constrs = uc_fw.sddip_copy_constrs

                    dm = []

                    try:
                        for constr in copy_constrs:
                            dm.append(constr.getAttr(gp.GRB.attr.Pi))
                    except AttributeError:
                        uc_fw.model.write("model.lp")
                        uc_fw.model.computeIIS()
                        uc_fw.model.write("model.ilp")
                        raise

                    dual_multipliers.append(dm)

                    dual_model = ucmodelclassical.ClassicalModel(
                        self.problem_params.n_buses,
                        self.problem_params.n_lines,
                        self.problem_params.n_gens,
                        self.problem_params.n_storages,
                        self.problem_params.gens_at_bus,
                        self.problem_params.storages_at_bus,
                        self.problem_params.backsight_periods,
                    )

                    dual_model: ucmodelclassical.ClassicalModel = (
                        self.add_problem_constraints(dual_model, t, n)
                    )
                    dual_model.add_objective(self.problem_params.cost_coeffs)

                    dual_model.calculate_relaxed_terms(
                        x_trial_point,
                        y_trial_point,
                        x_bs_trial_point,
                        soc_trial_point,
                    )

                    # copy_terms = dual_model.relaxed_terms

                    # (
                    #     _,
                    #     dual_value,
                    # ) = self.dual_solver.get_subgradient_and_value(
                    #     dual_model.model,
                    #     dual_model.objective_terms,
                    #     copy_terms,
                    #     dm,
                    # )

                    total_objective = dual_model.objective_terms + gp.quicksum(
                        dual_model.relaxed_terms[i] * dm[i] for i in range(len(dm))
                    )
                    dual_model.model.setObjective(total_objective)
                    dual_model.model.update()
                    dual_model.model.optimize()
                    dual_value = dual_model.model.getObjective().getValue()
                    opt_values.append(dual_value)

                opt_values = np.array(opt_values)
                dual_multipliers = np.array(dual_multipliers)
                v = np.average(opt_values)
                pi = np.average(dual_multipliers, axis=0)
                intercept = v - pi @ np.array(trial_point)

                benders_cut = np.concatenate((pi, intercept.reshape(-1)), axis=0).tolist()

                self.cuts_storage.add(i, k, t - 1, benders_cut)
                self.x_storage.add(i, k, t - 1, trial_point)
                # self.logger.info(f"benders_pi_intercept: {benders_cut}")
        self.cut_add_flag = True


    def create_inner_model(self, t, n, relax=False):
        inner_model = ucmodelclassical.ClassicalModel(
            self.problem_params.n_buses,
            self.problem_params.n_lines,
            self.problem_params.n_gens,
            self.problem_params.n_storages,
            self.problem_params.gens_at_bus,
            self.problem_params.storages_at_bus,
            self.problem_params.backsight_periods,
            lp_relax=relax  # lp_relax=True时level bundle只有一个解，False则有多解，但每个解都相同
        )

        # inner_model 添加问题约束: z_X与X的约束
        inner_model: ucmodelclassical.ClassicalModel = (
            self.add_problem_constraints(inner_model, t, n)
        )

        inner_model: ucmodelclassical.ClassicalModel = self.add_z_Var_constrains(inner_model)
        return inner_model

    def create_outer_model(self, X_trial, theta_trial, alpha, benders_pi_list=None):
        outer_model = outermodel.OuterModel(
            self.length_z, X_trial, theta_trial, benders_pi_list=benders_pi_list, alpha=alpha
        )
        return outer_model

        # level

    def level_bundle_methods(
            self,
            inner_model: ucmodelclassical.ClassicalModel,
            outer_model,
            X_trial,
            theta_trial,
            iteration_limit,
            level_factor=0.3,
            timeLimit=5 * 60,
            atol=1e-2,
            rtol=1e-2,
            pi0Coef=1e-2,

    ):

        subgradient_list = []
        pi_hat = np.ones(self.length_z) * 0
        pi0_hat = 1
        pi_star = None
        pi0_star = None

        LB = float('-inf')  # LB
        UB = float('inf')  # UB
        lpiold = float("inf")
        iter = 0

        # 记录起始时间
        start_time = time.time()

        while iter < iteration_limit:

            # 检查时间是否超过限制
            if time.time() - start_time > timeLimit:
                self.logger.info(f"超出时间限制 {timeLimit} 秒，提前终止 bundle 方法")
                break

            # inner_model.model.params.PoolSearchMode = 2  # 启用多解搜索（灵活模式）
            # inner_model.model.params.PoolSolutions = 10  # 返回最多 10 个解
            # inner_model.model.params.PoolGap = 0.1  # 每个解的目标值误差不超过 5%
            # inner_model.model.setParam('MIPStart', True)  # 启用初始解

            # 更新pi_hat, pi0_hat
            inner_model.set_inner_objective(self.problem_params.cost_coeffs, pi_hat, pi0_hat)
            inner_model.model.optimize()
            if inner_model.model.status == 3:  # 如果模型不可行
                inner_model.model.computeIIS()  # 计算不可行约束集
                inner_model.model.write(r"D:\tools\workspace_pycharm\sddip-SCUC-6-24\sddip_result\infeasible_model.ilp")  # 将不可行的约束写入文件
            if inner_model.model.status != 2:
                self.logger.info(f"pi_hat: {pi_hat} pi0_hat: {pi0_hat}")
                self.logger.info(
                    f"inner_model optimize---bundle interation:{iter}---inner_model.status:{inner_model.model.status}")
                inner_model.model.write(
                    r"D:\tools\workspace_pycharm\sddip-SCUC-6-24\sddip_result\inner_model.mps")  # 将不可行的约束写入文件
                if inner_model.model.status == gp.GRB.UNBOUNDED:
                    self.logger.info("Model is unbounded. Unbounded ray:")
                    self.logger.info(inner_model.model.unbdRay)

                break

            # 最优解，添加到cut_history
            y_kt = [y_g.x for y_g in inner_model.y]
            s_up_kt = [s_up_g.x for s_up_g in inner_model.s_up]
            s_down_kt = [s_down_g.x for s_down_g in inner_model.s_down]
            ys_p_kt = inner_model.ys_p.x
            ys_n_kt = inner_model.ys_n.x
            socs_p_kt = [socs_p_g.x for socs_p_g in inner_model.socs_p]
            socs_n_kt = [socs_n_g.x for socs_n_g in inner_model.socs_n]
            x_bs_p_kt = [x_bs_p_g.x for g in range(inner_model.n_generators) for x_bs_p_g in inner_model.x_bs_p[g]]
            x_bs_n_kt = [x_bs_n_g.x for g in range(inner_model.n_generators) for x_bs_n_g in inner_model.x_bs_n[g]]
            delta_kt = inner_model.delta.x
            theta_kt = inner_model.theta.x
            coefficients = self.problem_params.cost_coeffs
            penalty = coefficients[-1]
            coefficients = (
                    coefficients + [penalty] * (2 * inner_model.n_storages + 2 * len(x_bs_p_kt) + 1) + [1]
            )
            variables = (
                    y_kt
                    + s_up_kt
                    + s_down_kt
                    + [ys_p_kt, ys_n_kt]
                    + socs_p_kt
                    + socs_n_kt
                    + x_bs_p_kt
                    + x_bs_n_kt
                    + [delta_kt]
                    + [theta_kt]
            )
            # 计算目标函数的形式计算theta
            theta_value = sum(variables[i] * coefficients[i] for i in range(len(coefficients)))
            # 获取z值
            z_x_kt = [x_g.x for x_g in inner_model.z_x]
            z_y_kt = [y_g.x for y_g in inner_model.z_y]
            z_x_bs_kt = [
                x_bs.x for x_bs_g in inner_model.z_x_bs for x_bs in x_bs_g
            ]
            z_soc_kt = [soc_s.x for soc_s in inner_model.z_soc]
            pi_subgradient = z_x_kt + z_y_kt + z_x_bs_kt + z_soc_kt
            pi0_subgradient = theta_value
            subgradient_list.append(pi_subgradient + [pi0_subgradient])
            # 次梯度构造cut加入到outer_model中
            # print("outer_model add cut:", subgradient_list[-1])
            outer_model.add_constrains(subgradient_list[-1])
            outer_model.model.optimize()
            pi_dummy = [outer_model.pi[i].x for i in range(len(outer_model.pi))]
            pi0_dummy = outer_model.pi0.x

            if outer_model.model.status != 2:
                outer_model.model.write("outer_model.lp")
                self.logger.info(
                    f"outer_model optimize---interation:{iter}---outer_model.status:{outer_model.model.status}")

            inner_obj = inner_model.model.getObjective().getValue()
            gap = inner_obj - sum(pi_hat[i] * X_trial[i] for i in range(self.length_z)) - pi0_hat * theta_trial
            if gap > LB:
                LB = gap
                pi_star = pi_hat.copy()
                pi0_star = pi0_hat
            outer_obj = outer_model.model.getObjective().getValue()
            UB = outer_obj
            # if iter % 10 == 0:
            #     print(f"LB:{LB} UB:{UB} iter:{iter}")
            gapTol = 5e-3
            tol = 1e-4
            if UB - LB < gapTol * UB or UB - LB < 1e-4:
                # print(f"LB:{LB} UB:{UB} ************bundle收敛***********")
                # if pi0_star > 1e-6:
                #     print(f"pi_star+pi0_star: {pi_star} {pi0_star}")
                #     print(f"pi0_best > 1e-6")
                # else:
                #     print(f"pi_star+pi0_star: {pi_star} {pi0_star}")
                #     print(f"pi0Hat <= 1e-6")
                if pi0_star > 1e-6 and LB / pi0_star >= tol * (abs(theta_trial) + 1):
                    return pi_star, pi0_star, True

            QPsolved = True
            # level
            level = UB - level_factor * (UB - LB)
            outer_model.set_lower_bound(level)
            outer_model.set_level_obj(pi_hat, pi0_hat)

            outer_model.model.params.Method = 2
            outer_model.model.update()
            outer_model.model.optimize()
            if outer_model.model.status != 2:
                # print('QP status: ' + str(outer_model.model.status) + ' with Method=' + str(
                #     outer_model.model.params.Method) + '... Switching to 1')
                outer_model.model.params.Method = 1
                outer_model.model.update()
                outer_model.model.optimize()
                if outer_model.model.status != 2:
                    # print('QP status: ' + str(outer_model.model.status) + ' with Method=' + str(
                    #     outer_model.model.params.Method) + '... Switching to 0')
                    outer_model.model.params.Method = 0
                    outer_model.model.update()
                    outer_model.model.optimize()
                    if outer_model.model.status != 2:
                        # print('QP status: ' + str(outer_model.model.status) + ' with Method=' + str(
                        #     outer_model.model.params.Method) + '... Stop!')
                        QPsolved = False

            pi_hat_old = pi_hat.copy()
            pi0_hat_old = pi0_hat
            if QPsolved:
                for i in range(self.length_z):
                    pi_hat[i] = outer_model.pi[i].x
                pi0_hat = outer_model.pi0.x
                lpiold = outer_model.L.x
            else:
                pi_hat = pi_dummy
                pi0_hat = pi0_dummy

            # 若未找到最优解，或当前解与上次迭代的解非常接近（小于 1e-10），则认为解已收敛
            if QPsolved == False or (max(abs(pi_hat[i] - pi_hat_old[i]) for i in range(self.length_z)) < 1e-10 and abs(
                    pi0_hat - pi0_hat_old) < 1e-10 and abs(outer_model.L.x - lpiold) < 1e-10):
                # print('Same Solution/QP not solved! QPsolved:', QPsolved)
                # if pi0_star > 1e-6:
                #     print('pi0_best > 1e-6')
                # 若 pi0Best 足够大并且满足界限条件，则将该情景的割平面约束添加到主问题模型中
                if pi0_star > 1e-6 and LB >= tol * (abs(theta_trial) + 1):
                    return pi_star, pi0_star, True

            # 恢复模型的目标函数、删去level约束
            outer_model.recover()
            outer_model.model.params.Method = -1  # -1是什么？？
            outer_model.model.update()

            iter = iter + 1

        return pi_star, pi0_star, False

    def statistical_upper_bound(self, v_opt_k: list, n_samples: int) -> float:
        v_mean = np.mean(v_opt_k)
        v_std = np.std(v_opt_k)
        alpha = 0.05

        v_upper_l = v_mean + stats.norm.ppf(alpha / 2) * v_std / np.sqrt(
            n_samples
        )
        v_upper_r = v_mean - stats.norm.ppf(alpha / 2) * v_std / np.sqrt(
            n_samples
        )

        return v_upper_l, v_upper_r

    def lower_bound(self, iteration: int) -> float:
        t = 0
        n = 0

        # cuts = self.cuts_storage.get_values_by_t(t=0)
        # print(f"n_cuts: {len(cuts)}")
        # print(f"cuts: {cuts}")


        i = iteration
        x_trial_point = self.problem_params.init_x_trial_point
        y_trial_point = self.problem_params.init_y_trial_point
        x_bs_trial_point = self.problem_params.init_x_bs_trial_point
        soc_trial_point = self.problem_params.init_soc_trial_point
        # Create forward model
        uc_fw = ucmodelclassical.ClassicalModel(
            self.problem_params.n_buses,
            self.problem_params.n_lines,
            self.problem_params.n_gens,
            self.problem_params.n_storages,
            self.problem_params.gens_at_bus,
            self.problem_params.storages_at_bus,
            self.problem_params.backsight_periods,
            lp_relax=False
        )

        uc_fw: ucmodelclassical.ClassicalModel = (
            # 问题约束中没有Xa(n),转换为z与Xn的约束
            self.add_problem_constraints(uc_fw, t, n)
        )
        # 目标函数
        uc_fw.add_objective(self.problem_params.cost_coeffs)
        '''
        下边就是计算z与trial_point的差值relaxed_terms，前向过程需要relaxed_terms=0
        '''
        uc_fw.calculate_relaxed_terms(
            x_trial_point,
            y_trial_point,
            x_bs_trial_point,
            soc_trial_point,
        )
        # 置0
        uc_fw.zero_relaxed_terms()
        uc_fw.disable_output()
        uc_fw.model.optimize()
        # Value of stage t objective function
        v_lower = uc_fw.model.getObjective().getValue()
        return v_lower

    def add_problem_constraints(
        self,
        model_builder: ucmodelclassical.ClassicalModel,
        stage: int,
        realization: int,
        # iteration: int,
    ) -> ucmodelclassical.ClassicalModel:

        model_builder.add_balance_constraints(
            sum(self.problem_params.p_d[stage][realization]),#p_d的阶段从0到num_stage-1,(与scenarios中相差1),数据提取时处理
            sum(self.problem_params.re[stage][realization]),
            self.problem_params.eff_dc,
        )

        model_builder.add_power_flow_constraints(
            self.problem_params.ptdf,
            self.problem_params.pl_max,
            self.problem_params.p_d[stage][realization],
            self.problem_params.re[stage][realization],
            self.problem_params.eff_dc,
        )

        model_builder.add_storage_constraints(
            self.problem_params.rc_max,
            self.problem_params.rdc_max,
            self.problem_params.soc_max,
        )

        if stage == self.problem_params.n_stages - 1:
            model_builder.add_final_soc_constraints(
                self.problem_params.init_soc_trial_point
            )
        model_builder.add_soc_transfer(self.problem_params.eff_c)

        model_builder.add_generator_constraints(
            self.problem_params.pg_min, self.problem_params.pg_max
        )

        model_builder.add_startup_shutdown_constraints()

        model_builder.add_ramp_rate_constraints(
            self.problem_params.r_up,
            self.problem_params.r_down,
            self.problem_params.r_su,
            self.problem_params.r_sd,
        )

        model_builder.add_up_down_time_constraints(
            self.problem_params.min_up_time, self.problem_params.min_down_time
        )

        model_builder.add_cut_lower_bound(self.problem_params.cut_lb[stage])

        if stage < self.problem_params.n_stages - 1 and self.cut_add_flag:
            # 添加cut约束
            cuts_list = self.cuts_storage.get_values_by_t(stage)
            model_builder.add_cut_constrains(
                cuts_list
            )
        return model_builder

    def add_z_Var_constrains(self, inner_model: ucmodelclassical.ClassicalModel):
        inner_model.add_z_var_constrains(self.problem_params.soc_max, self.problem_params.pg_min, self.problem_params.pg_max)
        return inner_model

    def intercept_recalculate(self, cuts_predicted):

        n_pieces = cuts_predicted.shape[1]  # 5
        # print("n_pieces", n_pieces)

        # 场景采样 3个不同的路径
        self.n_samples = self.n_samples_primary
        n_samples = self.n_samples
        samples = self.sc_sampler.generate_samples(n_samples)

        # forward
        result = result_Dict("result_storage_temp")

        n_samples = len(samples)
        v_opt_k = []
        for k in range(n_samples):
            # t = -1对应的状态值
            x_trial_point = self.problem_params.init_x_trial_point
            y_trial_point = self.problem_params.init_y_trial_point
            x_bs_trial_point = self.problem_params.init_x_bs_trial_point
            soc_trial_point = self.problem_params.init_soc_trial_point
            v_opt_k.append(0)
            # 使用samples的方式进一步使每次迭代中的前向路径都不同
            for t, n in zip(range(self.problem_params.n_stages), samples[k]):

                # Create forward model
                # 模型定义中添加了问题所需要的变量，x(前向过程中x为二元变量)，z
                uc_fw = ucmodelclassical.ClassicalModel(
                    self.problem_params.n_buses,
                    self.problem_params.n_lines,
                    self.problem_params.n_gens,
                    self.problem_params.n_storages,
                    self.problem_params.gens_at_bus,
                    self.problem_params.storages_at_bus,
                    self.problem_params.backsight_periods,
                    lp_relax=False
                )

                uc_fw: ucmodelclassical.ClassicalModel = (
                    # 问题约束中没有Xa(n),转换为z与Xn的约束
                    self.add_problem_constraints(uc_fw, t, n)
                )
                # 目标函数
                uc_fw.add_objective(self.problem_params.cost_coeffs)
                '''
                下边就是计算z与trial_point的差值relaxed_terms，前向过程需要relaxed_terms=0
                '''
                uc_fw.calculate_relaxed_terms(
                    x_trial_point,
                    y_trial_point,
                    x_bs_trial_point,
                    soc_trial_point,
                )
                # 置0
                uc_fw.zero_relaxed_terms()

                # Solve problem
                uc_fw.disable_output()
                uc_fw.model.optimize()
                if uc_fw.model.status != 2:
                    self.logger.info(f"model.status {uc_fw.model.status}")

                try:
                    # 获取结果
                    x_kt = [x_g.x for x_g in uc_fw.x]
                    y_kt = [y_g.x for y_g in uc_fw.y]
                    x_bs_kt = [
                        [x_bs.x for x_bs in x_bs_g] for x_bs_g in uc_fw.x_bs
                    ]
                    soc_kt = [soc_s.x for soc_s in uc_fw.soc]
                except AttributeError:
                    print("uc_fw.model 求解错误")
                    uc_fw.model.write("model.lp")
                    uc_fw.model.computeIIS()
                    uc_fw.model.write("model.ilp")
                    raise

                # Value of stage t objective function
                v_value_function = uc_fw.model.getObjective().getValue()
                v_opt_kt = v_value_function - uc_fw.theta.x
                v_opt_k[-1] += v_opt_kt
                theta_value = uc_fw.theta.x

                x_trial_point = x_kt
                y_trial_point = y_kt
                if any(x_bs_kt):
                    # 更新bs
                    x_bs_trial_point = [
                        [x_trial_point[g]] + x_bs_kt[g][:-1]
                        for g in range(self.problem_params.n_gens)
                    ]
                soc_trial_point = soc_kt

                # x_bs_trial_point_one_dim = [item for sublist in x_bs_trial_point for item in sublist]
                # [X + theta]
                stage_result = []
                stage_result.append(x_kt)
                stage_result.append(y_kt)
                stage_result.append(x_bs_trial_point)
                stage_result.append(soc_kt)
                stage_result.append(theta_value)
                result.add(0, k, t, stage_result)

        # SB
        n_samples = len(samples)
        cuts_predicted_recalculate = {}
        # print("n_stage", self.problem_params.n_stages)
        for t in reversed(range(1, self.problem_params.n_stages)):
            for piece in range(n_pieces):
                intercept_list = []
                for k in range(n_samples):  # nums of trial points
                    n_realizations = self.problem_params.n_realizations_per_stage[t]
                    # Get trial points
                    stage_result = result.get_values(0, k, t - 1)
                    x_trial_point = stage_result[0]
                    y_trial_point = stage_result[1]
                    x_bs_trial_point = stage_result[2]
                    soc_trial_point = stage_result[3]

                    trial_point = (
                            x_trial_point
                            + y_trial_point
                            + [val for bs in x_bs_trial_point for val in bs]
                            + soc_trial_point
                    )

                    for n in range(n_realizations):

                        dm = cuts_predicted[t - 1][piece].tolist()

                        # dual model MILP Z应该也是整数
                        dual_model = ucmodelclassical.ClassicalModel(
                            self.problem_params.n_buses,
                            self.problem_params.n_lines,
                            self.problem_params.n_gens,
                            self.problem_params.n_storages,
                            self.problem_params.gens_at_bus,
                            self.problem_params.storages_at_bus,
                            self.problem_params.backsight_periods,
                            lp_relax=False,
                        )
                        dual_model: ucmodelclassical.ClassicalModel = (
                            # 问题约束中没有Xa(n),转换为z与Xn的约束
                            self.add_problem_constraints(dual_model, t, n)
                        )
                        dual_model.add_objective(self.problem_params.cost_coeffs)
                        '''
                        下边就是计算z与trial_point的差值relaxed_terms，前向过程需要relaxed_terms==0，现在不需要
                        '''
                        dual_model.calculate_relaxed_terms(
                            x_trial_point,
                            y_trial_point,
                            x_bs_trial_point,
                            soc_trial_point,
                        )
                        # print("len(dm)", len(dm))
                        # print("dual_model.relaxed_terms", len(dual_model.relaxed_terms))
                        # 目标函数的减号：relaxed_terms = Z - X_trial
                        total_objective = dual_model.objective_terms - gp.quicksum(
                            dual_model.relaxed_terms[i] * dm[i] for i in range(len(dm))
                        )

                        dual_model.model.setObjective(total_objective)
                        dual_model.model.update()
                        dual_model.model.optimize()
                        dual_value = dual_model.model.getObjective().getValue()


                        intercept_k_n = dual_value - dm @ np.array(trial_point)
                        intercept_list.append(intercept_k_n)
                intercept = mean(intercept_list)
                # print("t, piece", t - 1, piece)
                cuts_predicted_recalculate[(t - 1, piece)] = cuts_predicted[t - 1][piece].tolist() + [intercept]
        return cuts_predicted_recalculate

    def get_x_with_nocut(self, num_x):
        import torch

        # 场景采样个数
        n_samples = num_x
        samples = self.sc_sampler.generate_samples(n_samples)

        # forward
        result = result_Dict("x_with_nocut")

        v_opt_k = []
        for k in range(n_samples):
            # t = -1对应的状态值
            x_trial_point = self.problem_params.init_x_trial_point
            y_trial_point = self.problem_params.init_y_trial_point
            x_bs_trial_point = self.problem_params.init_x_bs_trial_point
            soc_trial_point = self.problem_params.init_soc_trial_point
            v_opt_k.append(0)
            # 使用samples的方式进一步使每次迭代中的前向路径都不同
            for t, n in zip(range(self.problem_params.n_stages), samples[k]):

                # Create forward model
                # 模型定义中添加了问题所需要的变量，x(前向过程中x为二元变量)，z
                uc_fw = ucmodelclassical.ClassicalModel(
                    self.problem_params.n_buses,
                    self.problem_params.n_lines,
                    self.problem_params.n_gens,
                    self.problem_params.n_storages,
                    self.problem_params.gens_at_bus,
                    self.problem_params.storages_at_bus,
                    self.problem_params.backsight_periods,
                    lp_relax=False
                )

                uc_fw: ucmodelclassical.ClassicalModel = (
                    # 问题约束中没有Xa(n),转换为z与Xn的约束
                    self.add_problem_constraints(uc_fw, t, n)
                )
                # 目标函数
                uc_fw.add_objective(self.problem_params.cost_coeffs)
                '''
                下边就是计算z与trial_point的差值relaxed_terms，前向过程需要relaxed_terms=0
                '''
                uc_fw.calculate_relaxed_terms(
                    x_trial_point,
                    y_trial_point,
                    x_bs_trial_point,
                    soc_trial_point,
                )
                # 置0
                uc_fw.zero_relaxed_terms()

                # Solve problem
                uc_fw.disable_output()
                uc_fw.model.optimize()
                if uc_fw.model.status != 2:
                    self.logger.info(f"model.status {uc_fw.model.status}")

                try:
                    # 获取结果
                    x_kt = [x_g.x for x_g in uc_fw.x]
                    y_kt = [y_g.x for y_g in uc_fw.y]
                    x_bs_kt = [
                        [x_bs.x for x_bs in x_bs_g] for x_bs_g in uc_fw.x_bs
                    ]
                    soc_kt = [soc_s.x for soc_s in uc_fw.soc]
                except AttributeError:
                    print("uc_fw.model 求解错误")
                    uc_fw.model.write("model.lp")
                    uc_fw.model.computeIIS()
                    uc_fw.model.write("model.ilp")
                    raise

                # Value of stage t objective function
                v_value_function = uc_fw.model.getObjective().getValue()
                v_opt_kt = v_value_function - uc_fw.theta.x
                v_opt_k[-1] += v_opt_kt
                theta_value = uc_fw.theta.x

                x_trial_point = x_kt
                y_trial_point = y_kt
                if any(x_bs_kt):
                    # 更新bs
                    x_bs_trial_point = [
                        [x_trial_point[g]] + x_bs_kt[g][:-1]
                        for g in range(self.problem_params.n_gens)
                    ]
                soc_trial_point = soc_kt

                # x_bs_trial_point_one_dim = [item for sublist in x_bs_trial_point for item in sublist]
                # [X + theta]
                stage_result = []
                stage_result.extend(x_kt)
                stage_result.extend(y_kt)
                stage_result.extend([item for sublist in x_bs_trial_point for item in sublist])
                stage_result.extend(soc_kt)
                # stage_result.extend([theta_value])
                result.add(0, k, t, stage_result)
        # 将收集的x整理成x_tensor  [0~T-1]
        x_list = []
        for t in range(self.problem_params.n_stages - 1):
            x_k_list = []
            for k in range(n_samples):
                x_k_list.append(result.get_values(0, k, t))
            x_list.append(x_k_list)
        np_array = np.array(x_list, dtype=np.float32)
        # 转换为 tensor
        x_tensor = torch.from_numpy(np_array)

        return x_tensor

    '''IFR算法'''

    def run_IFR(self, ):

        # SB
        n_samples = self.n_samples_primary
        samples = self.sc_sampler.generate_samples(
            n_samples
        )  # samples：[[1,2,5,3][]] len(samples)代表前向步骤执行次数（采样次数）,每个其中数字代表realization序号
        self.logger.info("Samples: %s", samples)

        # Forward pass
        self.logger.info("Forward pass")
        v_opt_k = self.forward_pass_IFR(0, samples)

        # # Statistical upper bound
        # v_upper_l, v_upper_r = self.statistical_upper_bound(
        #     v_opt_k, n_samples
        # )
        #
        # Backward pass
        self.logger.info("Backward benders pass")
        self.backward_benders_IFR(1, samples)

        # Lower bound
        # return v_lower, mean(v_opt_k), v_upper_l, v_upper_r

    def forward_pass_IFR(self, iteration: int, samples: list) -> list:
        i = iteration  # 从0开始
        n_samples = len(samples)
        v_opt_k = []

        for k in range(n_samples):
            # t = -1对应的状态值
            x_trial_point = self.problem_params.init_x_trial_point
            y_trial_point = self.problem_params.init_y_trial_point
            x_bs_trial_point = self.problem_params.init_x_bs_trial_point
            soc_trial_point = self.problem_params.init_soc_trial_point

            v_opt_k.append(0)
            #使用samples的方式进一步使每次迭代中的前向路径都不同
            for t, n in zip(range(self.problem_params.n_stages), samples[k]):

                # Create forward model
                # 模型定义中添加了问题所需要的变量，x(前向过程中x为二元变量)，z
                uc_fw = ucmodelclassical.ClassicalModel(
                    self.problem_params.n_buses,
                    self.problem_params.n_lines,
                    self.problem_params.n_gens,
                    self.problem_params.n_storages,
                    self.problem_params.gens_at_bus,
                    self.problem_params.storages_at_bus,
                    self.problem_params.backsight_periods,
                    lp_relax=False
                )

                uc_fw: ucmodelclassical.ClassicalModel = (
                    # 问题约束中没有Xa(n),转换为z与Xn的约束
                    self.add_problem_constraints(uc_fw, t, n)
                )
                # 目标函数
                uc_fw.add_objective(self.problem_params.cost_coeffs)
                '''
                下边就是计算z与trial_point的差值relaxed_terms，前向过程需要relaxed_terms=0
                '''
                uc_fw.calculate_relaxed_terms(
                    x_trial_point,
                    y_trial_point,
                    x_bs_trial_point,
                    soc_trial_point,
                )
                # 置0
                uc_fw.zero_relaxed_terms()

                # 模型输出
                path = r'D:\Desktop\SCUC\SCUC\ifr_data\model'
                uc_fw.model.write(os.path.join(path, f"forward_model_{i}_{k}_{t}.lp"))


                # Solve problem
                uc_fw.disable_output()
                uc_fw.model.optimize()
                if uc_fw.model.status != 2:
                    self.logger.info(f"model.status {uc_fw.model.status}")

                try:
                    # 获取结果
                    x_kt = [x_g.x for x_g in uc_fw.x]
                    y_kt = [y_g.x for y_g in uc_fw.y]
                    x_bs_kt = [
                        [x_bs.x for x_bs in x_bs_g] for x_bs_g in uc_fw.x_bs
                    ]
                    soc_kt = [soc_s.x for soc_s in uc_fw.soc]
                except AttributeError:
                    print("uc_fw.model 求解错误")
                    uc_fw.model.write("model.lp")
                    uc_fw.model.computeIIS()
                    uc_fw.model.write("model.ilp")
                    raise


                # Value of stage t objective function
                v_value_function = uc_fw.model.getObjective().getValue()
                v_opt_kt = v_value_function - uc_fw.theta.x
                v_opt_k[-1] += v_opt_kt

                # theta_hat = theta_{t-1}=obj_t
                # theta_value = v_value_function
                theta_value = uc_fw.theta.x

                x_trial_point = x_kt
                y_trial_point = y_kt
                if any(x_bs_kt):
                    # 更新bs
                    x_bs_trial_point = [
                        [x_trial_point[g]] + x_bs_kt[g][:-1]
                        for g in range(self.problem_params.n_gens)
                    ]
                soc_trial_point = soc_kt

                # x_bs_trial_point_one_dim = [item for sublist in x_bs_trial_point for item in sublist]
                # [X + theta]
                stage_result = []
                stage_result.append(x_kt)
                stage_result.append(y_kt)
                stage_result.append(x_bs_trial_point)
                stage_result.append(soc_kt)
                stage_result.append(theta_value)
                self.forward_result.add(i, k, t, stage_result)
        return v_opt_k

    def backward_benders_IFR(self, iteration: int, samples: list) -> None:
        i = iteration
        n_samples = len(samples)
        for t in reversed(range(1, self.problem_params.n_stages)):
            for k in range(n_samples):
                n_realizations = self.problem_params.n_realizations_per_stage[
                    t
                ]
                # Get trial points
                stage_result = self.forward_result.get_values(i - 1, k, t - 1)
                x_trial_point = stage_result[0]
                y_trial_point = stage_result[1]
                x_bs_trial_point = stage_result[2]
                soc_trial_point = stage_result[3]

                trial_point = (
                        x_trial_point
                        + y_trial_point
                        + [val for bs in x_bs_trial_point for val in bs]
                        + soc_trial_point
                )

                dual_multipliers = []
                opt_values = []

                for n in range(n_realizations):
                    # Create forward model
                    uc_fw = ucmodelclassical.ClassicalModel(
                        self.problem_params.n_buses,
                        self.problem_params.n_lines,
                        self.problem_params.n_gens,
                        self.problem_params.n_storages,
                        self.problem_params.gens_at_bus,
                        self.problem_params.storages_at_bus,
                        self.problem_params.backsight_periods,
                        lp_relax=True,
                    )

                    uc_fw: ucmodelclassical.ClassicalModel = (
                        self.add_problem_constraints(uc_fw, t, n)
                    )

                    uc_fw.add_objective(self.problem_params.cost_coeffs)

                    uc_fw.calculate_relaxed_terms(
                        x_trial_point,
                        y_trial_point,
                        x_bs_trial_point,
                        soc_trial_point,
                    )
                    uc_fw.zero_relaxed_terms()

                    uc_fw.model.optimize()

                    copy_constrs = uc_fw.sddip_copy_constrs

                    # 模型输出
                    path = r'D:\Desktop\SCUC\SCUC\ifr_data\model'
                    uc_fw.model.write(os.path.join(path, f"benders_model_{i}_{k}_{t}.lp"))

                    dm = []

                    try:
                        for constr in copy_constrs:
                            dm.append(constr.getAttr(gp.GRB.attr.Pi))
                    except AttributeError:
                        uc_fw.model.write("model.lp")
                        uc_fw.model.computeIIS()
                        uc_fw.model.write("model.ilp")
                        raise

                    dual_multipliers.append(dm)

                    dual_model = ucmodelclassical.ClassicalModel(
                        self.problem_params.n_buses,
                        self.problem_params.n_lines,
                        self.problem_params.n_gens,
                        self.problem_params.n_storages,
                        self.problem_params.gens_at_bus,
                        self.problem_params.storages_at_bus,
                        self.problem_params.backsight_periods,
                    )

                    dual_model: ucmodelclassical.ClassicalModel = (
                        self.add_problem_constraints(dual_model, t, n)
                    )
                    dual_model.add_objective(self.problem_params.cost_coeffs)

                    dual_model.calculate_relaxed_terms(
                        x_trial_point,
                        y_trial_point,
                        x_bs_trial_point,
                        soc_trial_point,
                    )

                    # copy_terms = dual_model.relaxed_terms

                    # (
                    #     _,
                    #     dual_value,
                    # ) = self.dual_solver.get_subgradient_and_value(
                    #     dual_model.model,
                    #     dual_model.objective_terms,
                    #     copy_terms,
                    #     dm,
                    # )

                    total_objective = dual_model.objective_terms + gp.quicksum(
                        dual_model.relaxed_terms[i] * dm[i] for i in range(len(dm))
                    )
                    dual_model.model.setObjective(total_objective)
                    dual_model.model.update()
                    dual_model.model.optimize()
                    dual_value = dual_model.model.getObjective().getValue()
                    opt_values.append(dual_value)

                opt_values = np.array(opt_values)
                dual_multipliers = np.array(dual_multipliers)
                v = np.average(opt_values)
                pi = np.average(dual_multipliers, axis=0)
                intercept = v - pi @ np.array(trial_point)

                benders_cut = np.concatenate((pi, intercept.reshape(-1)), axis=0).tolist()

                self.cuts_storage.add(i, k, t - 1, benders_cut)
                self.x_storage.add(i, k, t - 1, trial_point)
                # self.logger.info(f"benders_pi_intercept: {benders_cut}")
        self.cut_add_flag = True

