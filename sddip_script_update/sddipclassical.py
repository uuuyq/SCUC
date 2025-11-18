import time
from statistics import mean
import gurobipy as gp
import numpy as np
from scipy import stats

from . import scenarios, ucmodelclassical, parameters, outermodel, dualsolver
from .result_storage import result_Dict
import os
import matplotlib.pyplot as plt
import pickle as pkl



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
            if self.check_convergence(LB_list, stop_stabilization_count, factor=100):
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
            if self.check_convergence(LB_list, stop_stabilization_count, factor=1):
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

    def check_convergence(self, LB_list, stop_stabilization_count, factor):
        """
        根据LB的变化检查是否收敛
        :param LB_list: 迭代的下界list
        :param stop_stabilization_count: stop_stabilization_count
        :param factor: 在相对误差上乘以factor，用于放松判断收敛的条件
        :return: boolean
        """
        if len(LB_list) >= stop_stabilization_count + 1 and \
                (LB_list[-1] - LB_list[-(stop_stabilization_count + 1)]) / (
                abs(LB_list[-(stop_stabilization_count + 1)]) + 1e-8) < self.relative_tolerance * factor:
            return True
        return False

    def get_all_cuts_array(self, n_iterations):
        """
        收集所有的cuts，整理为array shape[t, n, d] n是将k和i合在一起
        :param n_iterations: 迭代次数，用i作为输入就行
        :return: cuts_array 三维数组
        """
        # 1. 按 t 收集所有 cuts
        cuts_t_list = []
        max_len = 0
        cut_dim = None

        for t in range(self.n_stages - 1):
            cuts_t = np.asarray(self.cuts_storage.get_values_by_t(t), dtype=float)

            if cut_dim is None:
                cut_dim = cuts_t.shape[1] if cuts_t.ndim > 1 else 1

            cuts_t_list.append(cuts_t)
            max_len = max(max_len, cuts_t.shape[0])

        # 2. 构造最终 padded 数组
        cuts_array = np.zeros((self.n_stages - 1, max_len, cut_dim), dtype=float)

        # 3. 写入每个 t 的数据
        for t, arr in enumerate(cuts_t_list):
            length = arr.shape[0]
            cuts_array[t, :length, :] = arr

        return cuts_array

    def run_n_lag(self, max_lag, logger=None):
        if logger is not None:
            self.logger = logger
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
            cuts_array = self.get_all_cuts_array(n_iterations=i + 1)
            lag_cuts_list.append(cuts_array)

        self.logger.info("#### SDDiP-Algorithm finished ####")

        return lag_time_list, lag_obj_list, lag_cuts_list

    def run_sddip_fw_n_samples(self, fw_n_samples, max_iterations, logger=None):
        """
        执行sddip，记录时间和obj，LB也记录下吧，注意计算obj时前向需要多次
        :return: obj_list, time_list，LB_list
        """
        if logger is not None:
            self.logger = logger
        self.logger.info("#### SDDiP-Algorithm started ####")
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
        # 总共执行40次
        while i < max_iterations:
            start_time = time.time()
            self.SB_lag_iteration(LB_list, i)
            end_time = time.time()
            time_cost += end_time - start_time

            self.logger.info(f"SB-1 or lag-2 choose:{self.current_cut_mode}  iteration:{i} ")
            self.logger.info(f"time_list: {time_list}")
            self.logger.info(f"obj_list: {obj_list}")
            self.logger.info(f"LB_list: {LB_list}")


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
        self.logger.info("#### SDDiP-Algorithm finished ####")

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

    # def SB_lag_iteration(self, LB_list, index):
    #     # 根据LB_list的情况，选择SB或lag迭代
    #     no_improvement_condition = False
    #     if len(LB_list) > 1:
    #         no_improvement_condition = (LB_list[-1] - LB_list[-2]) / (abs(LB_list[-2]) + 1e-8) <= self.relative_tolerance * 100
    #         # print(f"RE: {(LB_list[-1] - LB_list[-2]) / (abs(LB_list[-2]) + 1e-8)}  <= {self.relative_tolerance * 100}")
    #     if self.current_cut_mode == self.secondary_cut_mode:
    #         self.current_cut_mode = self.primary_cut_mode
    #     elif no_improvement_condition:
    #         self.current_cut_mode = self.secondary_cut_mode
    #
    #     if self.current_cut_mode == self.primary_cut_mode:
    #         return self.SB_iteration(index)
    #     elif self.current_cut_mode == self.secondary_cut_mode:
    #         return self.lag_iteration(index)
    #     else:
    #         raise ValueError("self.current_cut_mode出错")

    def SB_lag_iteration(self, LB_list, index):
        # 初始化计数器（只在第一次调用时）
        if not hasattr(self, "primary_no_improve_count"):
            self.primary_no_improve_count = 0

        # 检查 LB 是否改善
        no_improvement_condition = False
        if len(LB_list) > 1:
            delta = (LB_list[-1] - LB_list[-2]) / (abs(LB_list[-2]) + 1e-8)
            no_improvement_condition = delta <= self.relative_tolerance

        # -------- 新增逻辑：primary模式下出现无改善，计数次数 --------
        if self.current_cut_mode == self.primary_cut_mode:
            if no_improvement_condition:
                self.primary_no_improve_count += 1
            else:
                self.primary_no_improve_count = 0

        # 设定一个阈值，例如连续 5 次无改善就放弃 primary
        if not hasattr(self, "primary_no_improve_limit"):
            self.primary_no_improve_limit = 2

        # 达到限制 → 永久切换到 secondary
        if self.primary_no_improve_count >= self.primary_no_improve_limit:
            self.logger.info("##########切换至lag模式#########")
            self.current_cut_mode = self.secondary_cut_mode
        else:
            # 原始逻辑：尽量使用 primary，无改善时切 secondary
            if self.current_cut_mode == self.secondary_cut_mode:
                self.current_cut_mode = self.primary_cut_mode
            elif no_improvement_condition:
                self.current_cut_mode = self.secondary_cut_mode

        # ---------------- 实际调用 ----------------
        if self.current_cut_mode == self.primary_cut_mode:
            return self.SB_iteration(index)
        elif self.current_cut_mode == self.secondary_cut_mode:
            return self.lag_iteration(index)
        else:
            raise ValueError("self.current_cut_mode 出错")



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

                    # 输出解
                    # self.logger.info(f"stage: {t}  n: {n}")
                    # self.logger.info(f"x: {[x.x for x in uc_fw.x]}")
                    # self.logger.info(f"y: {[y.x for y in uc_fw.y]}")
                    # self.logger.info(f"x_bs: {[x.x for x_bs in uc_fw.x_bs for x in x_bs]}")
                    # self.logger.info(f"soc: {[soc.x for soc in uc_fw.soc]}")
                    # self.logger.info(f"socs_p: {[soc.x for soc in uc_fw.socs_p]}")
                    # self.logger.info(f"socs_n: {[soc.x for soc in uc_fw.socs_n]}")
                    # self.logger.info(f"theta: {uc_fw.theta.x}")
                    # self.logger.info(f"delta: {uc_fw.delta.x}")
                    #
                    # self.logger.info(f"dual: {dm}")
                    # if t == 1 and k == 0:
                    #     uc_fw.model.write("SB_model.lp")

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

    def add_problem_constraints_matrix(
        self,
        model_builder: ucmodelclassical.ClassicalModel,
        stage: int,
        realization: int,
        # iteration: int,
    ):

        if stage == self.problem_params.n_stages - 1:
            init_soc_trial_point = self.problem_params.init_soc_trial_point
        else:
            init_soc_trial_point = None

        if stage < self.problem_params.n_stages - 1 and self.cut_add_flag:
            # 添加cut约束
            cuts_list = self.cuts_storage.get_values_by_t(stage)
            # self.logger.info(f"stage: {stage} n_cuts: {len(cuts_list)}")
        else:
            cuts_list = None
        # theta 的下界约束  直接写入模型中
        model_builder.add_cut_lower_bound(self.problem_params.cut_lb[stage])


        X_vector, A_eq, b_eq, A_ub, b_ub, A_ub_cut, b_ub_cut = model_builder.get_problem_constrains_matrix(
            # balance_constraints
            total_demand=sum(self.problem_params.p_d[stage][realization]),
            total_renewable_generation=sum(self.problem_params.re[stage][realization]),
            discharge_eff=self.problem_params.eff_dc,
            # generator_constraints
            min_generation=self.problem_params.pg_min,
            max_generation=self.problem_params.pg_max,
            # storage_constraints
            max_charge_rate=self.problem_params.rc_max,
            max_discharge_rate=self.problem_params.rdc_max,
            max_soc=self.problem_params.soc_max,
            # soc_transfer
            charge_eff=self.problem_params.eff_c,
            # final_soc_constraints
            final_soc=init_soc_trial_point,
            # power_flow_constraints
            ptdf=self.problem_params.ptdf,
            max_line_capacities=self.problem_params.pl_max,
            demand=self.problem_params.p_d[stage][realization],
            renewable_generation=self.problem_params.re[stage][realization],
            # ramp_rate_constraints
            max_rate_up=self.problem_params.r_up,
            max_rate_down=self.problem_params.r_down,
            startup_rate=self.problem_params.r_su,
            shutdown_rate=self.problem_params.r_sd,
            # up_down_time_constraints
            min_up_times=self.problem_params.min_up_time,
            min_down_times=self.problem_params.min_down_time,
            # cut_lower_bound
            cuts_list=cuts_list
        )
        # 添加等式约束 A_eq * X_vector == b_eq
        for row in range(A_eq.shape[0]):
            expr = 0
            for idx, col in enumerate(A_eq[row].nonzero()[1]):  # 非零列
                expr += A_eq[row, col] * X_vector[col]
            model_builder.model.addConstr(expr == b_eq[row], name=f"eq_constrains_{row}")

        # 添加不等式约束 A_ub * X_vector <= b_ub
        for row in range(A_ub.shape[0]):
            expr = 0
            for idx, col in enumerate(A_ub[row].nonzero()[1]):
                expr += A_ub[row, col] * X_vector[col]
            model_builder.model.addConstr(expr <= b_ub[row], name=f"ub_constrains_{row}")
        # 添加不等式割约束 A_ub_cut * X_vector <= b_ub_cut
        for row in range(A_ub_cut.shape[0]):
            expr = 0
            for idx, col in enumerate(A_ub_cut[row].nonzero()[1]):
                expr += A_ub_cut[row, col] * X_vector[col]
            # print(expr)
            # print(b_ub_cut[row])
            model_builder.model.addConstr(expr <= b_ub_cut[row], name=f"ub_cut_constrains_{row}")

        model_builder.model.update()

        # 转换为数组
        A_eq = A_eq.toarray()
        A_ub = A_ub.toarray()
        A_ub_cut = A_ub_cut.toarray()

        return model_builder, X_vector, A_eq, b_eq, A_ub, b_ub, A_ub_cut, b_ub_cut

    def add_z_Var_constrains(self, inner_model: ucmodelclassical.ClassicalModel):
        inner_model.add_z_var_constrains(self.problem_params.soc_max, self.problem_params.pg_min, self.problem_params.pg_max)
        return inner_model

    def intercept_recalculate(self, cuts_predicted, logger=None):

        # TODO: 截距重算应该还能改进，比如采样1阶段得到x，重算1阶段的cut截距，然后可以将cut添加到问题中，
        #  帮助前向过程中计算第2阶段x：前向计算从前往后计算，先计算出x_1，在去找x_2，如果x_1更优，应该有助于找到更好的x_2，从而得到更好的cut

        n_pieces = cuts_predicted.shape[1]  # 5
        # print("n_pieces", n_pieces)

        # 场景采样 3个不同的路径
        self.n_samples = self.n_samples_primary
        n_samples = 1
        samples = self.sc_sampler.generate_samples(n_samples)

        if logger is not None:
            logger.info(f"samples: {samples}")
            self.logger = logger

        # forward
        result = result_Dict("result_storage_temp")

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

        self.logger.info(f"forward complete")
        self.logger.info(f"forward result: {result}")
        self.logger.info("re start...")
        # SB
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
                self.logger.info(f"t, piece, cuts: {t - 1}, {piece}, {cuts_predicted_recalculate[(t - 1, piece)]}")
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
        # 将收集的x整理成x_array [0~T-1]
        x_list = []
        for t in range(self.problem_params.n_stages - 1):
            x_k_list = []
            for k in range(n_samples):
                x_k_list.append(result.get_values(0, k, t))
            x_list.append(x_k_list)
        x_array = np.array(x_list, dtype=np.float32)

        return x_array

    '''IFR算法'''
    # 测试结果，只有第一个阶段中的数值有一些差距，考虑精度问题，但不知道是哪里的精度问题导致？
    def test_matrix(self):
        # 采样 每次迭代只采一个样本
        samples = self.sc_sampler.generate_samples(1)
        self.logger.info("Samples: %s", samples)
        # Forward pass
        self.logger.info("Forward pass")
        v_opt_k = self.forward_pass(0, samples)
        # Backward pass
        self.logger.info("Backward benders pass")
        self.backward_benders(1, samples)

        self.cut_add_flag = False
        self.cuts_storage = result_Dict("cuts_storage1")
        self.x_storage = result_Dict("x_storage1")

        self.logger.info("Backward benders matrix pass")
        self.backward_benders_matrix(1, samples)

    def get_dual_values(self, file_name):
        """计算保存标准对偶值"""
        dual_values_dict_path = os.path.join(self.train_data_path, "dual_values_dict.pkl")
        if os.path.exists(dual_values_dict_path):
            self.logger.info("对偶数据已存在，无需重新生成")
            return

        num_cuts = 15
        self.dual_values_dict = {}
        for i in range(num_cuts):
            # 采样 每次迭代只采一个样本
            samples = self.sc_sampler.generate_samples(1)
            self.logger.info("Samples: %s", samples)
            # Forward pass
            self.logger.info(f"Forward pass iter: {i}")
            v_opt_k = self.forward_pass(i, samples)
            # Backward pass
            self.logger.info("Backward benders pass")
            self.backward_benders_matrix(i + 1, samples, storage_flag=True)

        # 保存到train_data_path中
        with open(dual_values_dict_path, 'wb') as f:
            pkl.dump(self.dual_values_dict, f)
        # 保存cut x
        self.cuts_storage.save_cut(file_name, self.train_data_path)
        self.x_storage.save_cut(file_name.split("_")[0] + "_x", self.train_data_path)





    def run_IFR(self, file_name):
        """执行IFR算法，得到近似的cut"""
        n_cuts = 15

        with open(os.path.join(self.train_data_path, "dual_values_dict.pkl"), 'rb') as f:
            dual_values_dict = pkl.load(f)
        # 输出查看标准对偶值
        # for t in reversed(range(1, self.problem_params.n_stages)):
        #     for i in range(n_cuts):
        #         self.logger.info(f"i: {i} t: {t}")
        #         self.logger.info(f"dual_value: {dual_values_dict[(i + 1, 0, t)]}")

        # self.n_stages
        # 最后一个阶段，使用SB求解得到num_cuts 个cut
        for i in range(n_cuts):
            # 采样 每次迭代只采一个样本
            samples = self.sc_sampler.generate_samples(1)
            self.logger.info("Samples: %s", samples)
            # Forward pass 前向都需要从前往后，只能使用全阶段的forward计算
            self.logger.info(f"Forward pass iter: {i}")
            v_opt_k = self.forward_pass(i, samples)
            # Backward pass
            self.logger.info(f"Backward benders pass iter: {i}")
            # 执行T-1阶段的后向，计算得到关于x_{T-2}阶段的cut
            self.backward_benders_matrix(iteration=i + 1, samples=samples, stage=self.problem_params.n_stages - 1, storage_flag=False)

        self.logger.info(f"IFR start...")



        # 前面的阶段
        self.IFR(dual_values_dict, n_cuts)

        self.cuts_storage.save_cut(file_name, self.train_data_path)
        self.x_storage.save_cut(file_name.split("_")[0] + "_x", self.train_data_path)

    def IFR(self, dual_values_dict, n_cuts):
        # [T-2, 1]
        for t in reversed(range(1, self.problem_params.n_stages - 1)):

            # 求解二次问题，得到预测的次梯度
            alpha_list = []
            for n_cut in range(1, n_cuts + 1):
                n = 0  # 下面不计算截距，因此不涉及场景的采样， 场景随便取一个就行
                # 获取标准对偶值
                dual_values = dual_values_dict[(n_cut, 0, t)]  # k暂时用0，后面也可以考虑尝试将所有的k聚合
                dual_values_eq = dual_values[0]  # 平等约束
                dual_values_ub = dual_values[1]  # 不等式约束
                dual_values_cut = dual_values[2]  # 割约束


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

                uc_fw, X_vector, A_eq, b_eq, A_ub, b_ub, A_ub_cut, b_ub_cut = (
                    self.add_problem_constraints_matrix(uc_fw, t, n)  # n对应的负荷场景，会影响到b_eq
                )
                # 只有一行
                A_obj = uc_fw.get_obj_coefficients(self.problem_params.cost_coeffs)

                uc_fw.model.dispose()  # 内存释放

                var_column_dict = uc_fw.get_var_column_index()
                X_column = var_column_dict['X']
                Y_column = var_column_dict['Y']
                Z_X_column = var_column_dict['Z_X']
                theta_column = var_column_dict['theta']
                # eq
                A_eq_X = A_eq[:, X_column[0]: X_column[1]]
                A_eq_Y = A_eq[:, Y_column[0]: Y_column[1]]
                A_eq_Z_X = A_eq[:, Z_X_column[0]: Z_X_column[1]]
                # ub
                A_ub_X = A_ub[:, X_column[0]: X_column[1]]
                A_ub_Y = A_ub[:, Y_column[0]: Y_column[1]]
                A_ub_Z_X = A_ub[:, Z_X_column[0]: Z_X_column[1]]
                # cut
                A_ub_cut_X = A_ub_cut[:, X_column[0]: X_column[1]]
                A_ub_cut_theta = A_ub_cut[:, theta_column[0]: theta_column[1]]
                # obj
                A_obj_X = A_obj[:, X_column[0]: X_column[1]]
                A_obj_Y = A_obj[:, Y_column[0]: Y_column[1]]
                A_obj_theta = A_obj[:, theta_column[0]: theta_column[1]]

                # 打印上面矩阵的shape
                # print("dual_values_eq: ", len(dual_values_eq))
                # print("dual_values_ub: ", len(dual_values_ub))
                # print("dual_values_cut: ", len(dual_values_cut))
                # print("A_eq_X shape: ", A_eq_X.shape)
                # print("A_eq_Y shape: ", A_eq_Y.shape)
                # print("A_eq_Z_X shape: ", A_eq_Z_X.shape)
                # print("A_ub_X shape: ", A_ub_X.shape)
                # print("A_ub_Y shape: ", A_ub_Y.shape)
                # print("A_ub_Z_X shape: ", A_ub_Z_X.shape)
                # print("A_ub_cut_X shape: ", A_ub_cut_X.shape)
                # print("A_ub_cut_theta shape: ", A_ub_cut_theta.shape)
                # print("A_obj_X shape: ", A_obj_X.shape)
                # print("A_obj_Y shape: ", A_obj_Y.shape)
                # print("A_obj_theta shape: ", A_obj_theta.shape)
                self.logger.info(f"t: {t} i: {n_cut}")
                self.logger.info("二次模型创建开始...")


                # 创建二次问题模型
                IFR_model = gp.Model("IFR model")
                IFR_model.setParam("OutputFlag", 0)
                # 对偶变量 - 使用紧凑的创建方式
                pi_eq = IFR_model.addVars(
                    len(dual_values_eq),
                    vtype=gp.GRB.CONTINUOUS,
                    name="pi_eq"
                )

                pi_ub = IFR_model.addVars(
                    len(dual_values_ub),
                    vtype=gp.GRB.CONTINUOUS,
                    lb=0,
                    name="pi_ub"
                )
                pi_cut = IFR_model.addVars(
                    n_cut,
                    vtype=gp.GRB.CONTINUOUS,
                    lb=0,
                    name="pi_cut"
                )

                # 为X、Y、theta部分创建绝对值变量
                abs_vars_X = IFR_model.addVars(A_obj_X.shape[1], vtype=gp.GRB.CONTINUOUS, lb=0, name="abs_X")
                abs_vars_Y = IFR_model.addVars(A_obj_Y.shape[1], vtype=gp.GRB.CONTINUOUS, lb=0, name="abs_Y")
                abs_var_theta = IFR_model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name="abs_theta")


                IFR_model.update()


                # 驻点约束权重
                w_X = 50.0
                w_Y = 50.0
                w_theta = 50.0
                # 正则项权重
                w_pi_eq = 1.0
                w_pi_ub = 1.0
                w_pi_cut = 1.0

                obj_terms = []

                # # X向量部分
                # for i in range(A_obj_X.shape[1]):
                #     expr = A_obj_X[0, i]
                #     if A_eq_X.shape[0] > 0:
                #         expr += gp.quicksum(pi_eq[j] * A_eq_X[j, i] for j in range(A_eq_X.shape[0]))
                #     if A_ub_X.shape[0] > 0:
                #         expr += gp.quicksum(pi_ub[j] * A_ub_X[j, i] for j in range(A_ub_X.shape[0]))
                #     if A_ub_cut_X.shape[0] > 0:
                #         expr += gp.quicksum(pi_cut[j] * A_ub_cut_X[j, i] for j in range(n_cut))
                #     obj_expr.add(expr * expr, w_X)
                #
                # # Y向量部分
                # for i in range(A_obj_Y.shape[1]):
                #     expr = A_obj_Y[0, i]
                #     if A_eq_Y.shape[0] > 0:
                #         expr += gp.quicksum(pi_eq[j] * A_eq_Y[j, i] for j in range(A_eq_Y.shape[0]))
                #     if A_ub_Y.shape[0] > 0:
                #         expr += gp.quicksum(pi_ub[j] * A_ub_Y[j, i] for j in range(A_ub_Y.shape[0]))
                #     obj_expr.add(expr * expr, w_Y)
                #
                # # theta部分
                # if A_ub_cut_theta.shape[0] > 0:
                #     expr_theta = A_obj_theta[0, 0] + gp.quicksum(
                #         pi_cut[j] * A_ub_cut_theta[j, 0] for j in range(n_cut))
                #     obj_expr.add(expr_theta * expr_theta, w_theta)

                # X向量部分 - 使用绝对值约束
                for i in range(A_obj_X.shape[1]):
                    expr = A_obj_X[0, i]
                    if A_eq_X.shape[0] > 0:
                        expr += gp.quicksum(pi_eq[j] * A_eq_X[j, i] for j in range(A_eq_X.shape[0]))
                    if A_ub_X.shape[0] > 0:
                        expr += gp.quicksum(pi_ub[j] * A_ub_X[j, i] for j in range(A_ub_X.shape[0]))
                    if A_ub_cut_X.shape[0] > 0:
                        expr += gp.quicksum(pi_cut[j] * A_ub_cut_X[j, i] for j in range(n_cut))

                    # 添加绝对值约束
                    IFR_model.addConstr(abs_vars_X[i] >= expr, name=f"abs_X_pos_{i}")
                    IFR_model.addConstr(abs_vars_X[i] >= -expr, name=f"abs_X_neg_{i}")

                # Y向量部分 - 使用绝对值约束
                for i in range(A_obj_Y.shape[1]):
                    expr = A_obj_Y[0, i]
                    if A_eq_Y.shape[0] > 0:
                        expr += gp.quicksum(pi_eq[j] * A_eq_Y[j, i] for j in range(A_eq_Y.shape[0]))
                    if A_ub_Y.shape[0] > 0:
                        expr += gp.quicksum(pi_ub[j] * A_ub_Y[j, i] for j in range(A_ub_Y.shape[0]))

                    # 添加绝对值约束
                    IFR_model.addConstr(abs_vars_Y[i] >= expr, name=f"abs_Y_pos_{i}")
                    IFR_model.addConstr(abs_vars_Y[i] >= -expr, name=f"abs_Y_neg_{i}")

                # theta部分 - 使用绝对值约束
                if A_ub_cut_theta.shape[0] > 0:
                    expr_theta = A_obj_theta[0, 0] + gp.quicksum(
                        pi_cut[j] * A_ub_cut_theta[j, 0] for j in range(n_cut))

                    # 添加绝对值约束
                    IFR_model.addConstr(abs_var_theta >= expr_theta, name="abs_theta_pos")
                    IFR_model.addConstr(abs_var_theta >= -expr_theta, name="abs_theta_neg")

                # 构建目标函数
                obj_expr = gp.QuadExpr()

                # X、Y、theta部分使用绝对值变量
                obj_expr.add(gp.quicksum(w_X * abs_vars_X[i] for i in range(A_obj_X.shape[1])))
                obj_expr.add(gp.quicksum(w_Y * abs_vars_Y[i] for i in range(A_obj_Y.shape[1])))
                obj_expr.add(w_theta * abs_var_theta)

                # 正则化项保持平方和
                for j in range(len(dual_values_eq)):
                    diff = dual_values_eq[j] - pi_eq[j]
                    obj_expr.add(diff * diff * w_pi_eq)

                for j in range(len(dual_values_ub)):
                    diff = dual_values_ub[j] - pi_ub[j]
                    obj_expr.add(diff * diff * w_pi_ub)

                for j in range(len(dual_values_cut)):
                    diff = dual_values_cut[j] - pi_cut[j]
                    obj_expr.add(diff * diff * w_pi_cut)

                # 设置目标函数
                IFR_model.setObjective(obj_expr, gp.GRB.MINIMIZE)

                # from pympler import muppy, summary
                #
                # all_objects = muppy.get_objects()
                # sum_stats = summary.summarize(all_objects)
                # summary.print_(sum_stats)

                """
                # 创建二次问题模型
                IFR_model = gp.Model("IFR model")
                IFR_model.setParam("OutputFlag", 0)
                # 对偶变量
                pi_eq = []
                pi_ub = []
                pi_cut = []
                for i in range(len(dual_values_eq)):
                    pi_eq.append(IFR_model.addVar(
                        vtype=gp.GRB.CONTINUOUS,
                        name=f"pi_eq_{i + 1}",
                    ))
                for i in range(len(dual_values_ub)):
                    pi_ub.append(IFR_model.addVar(
                        vtype=gp.GRB.CONTINUOUS,
                        lb=0,
                        name=f"pi_ub_{i + 1}",
                    ))
                for i in range(len(dual_values_cut)):
                    pi_cut.append(IFR_model.addVar(
                        vtype=gp.GRB.CONTINUOUS,
                        lb=0,
                        name=f"pi_cut_{i + 1}",
                    ))
                IFR_model.update()
                # 驻点约束
                # X向量
                expr_X = [
                    A_obj_X[0, i] +
                    gp.quicksum(pi_eq[j] * A_eq_X[j, i] for j in range(len(pi_eq))) +
                    gp.quicksum(pi_ub[j] * A_ub_X[j, i] for j in range(len(pi_ub))) +
                    gp.quicksum(pi_cut[j] * A_ub_cut_X[j, i] for j in range(len(pi_cut)))
                    for i in range(len(A_obj_X[0]))
                ]
                # Y向量
                expr_Y = [
                    A_obj_Y[0, i] +
                    gp.quicksum(pi_eq[j] * A_eq_Y[j, i] for j in range(len(pi_eq))) +
                    gp.quicksum(pi_ub[j] * A_ub_Y[j, i] for j in range(len(pi_ub)))
                    for i in range(len(A_obj_Y[0]))
                ]

                # print("A_obj_theta", A_obj_theta)
                # print("A_ub_cut_theta", A_ub_cut_theta)
                # theta
                expr_theta = A_obj_theta[0, 0] + gp.quicksum(pi_cut[j] * A_ub_cut_theta[j] for j in range(len(pi_cut)))

                expr_pi_eq = [dual_values_eq[i] - pi_eq[i] for i in range(len(dual_values_eq))]
                expr_pi_ub = [dual_values_ub[i] - pi_ub[i] for i in range(len(dual_values_ub))]
                expr_pi_cut = [dual_values_cut[i] - pi_cut[i] for i in range(len(dual_values_cut))]
                IFR_model.update()

                # 驻点约束权重
                w_X = 50.0
                w_Y = 50.0
                w_theta = 50.0
                # 正则项权重
                w_pi_eq = 1.0
                w_pi_ub = 1.0
                w_pi_cut = 1.0
                IFR_model.setObjective(w_X * gp.quicksum(x * x for x in expr_X) + w_Y * gp.quicksum(x * x for x in expr_Y)
                                       + w_theta * (expr_theta * expr_theta) + w_pi_eq * gp.quicksum(x * x for x in expr_pi_eq)
                                       + w_pi_ub * gp.quicksum(x * x for x in expr_pi_ub) + w_pi_cut * gp.quicksum(x * x for x in expr_pi_cut),
                                       gp.GRB.MINIMIZE)
                """
                IFR_model.update()
                IFR_model.optimize()

                pi_eq_value = [pi_eq[i].x for i in range(len(pi_eq))]
                pi_ub_value = [pi_ub[i].x for i in range(len(pi_ub))]
                pi_cut_value = [pi_cut[i].x for i in range(len(pi_cut))]  # 用于计算截距

                self.logger.info(f"t: {t} i: {n_cut}  stand_pi_cut.shape: {len(dual_values_cut)}")
                self.logger.info(f"pi_eq_value: {pi_eq_value}")
                self.logger.info(f"pi_ub_value: {pi_ub_value}")
                self.logger.info(f"pi_cut_value: {pi_cut_value}")


                # 根据计算出的pi值，计算cut
                alpha = [sum(pi_eq_value[j] * A_eq_Z_X[j, i] for j in range(len(pi_eq_value))) +
                         sum(pi_ub_value[j] * A_ub_Z_X[j, i] for j in range(len(pi_ub_value)))
                         for i in range(len(A_eq_Z_X[0]))]
                # beta = (-sum(pi_eq_value[i] * b_eq[i] for i in range(len(pi_eq_value)))
                #         - sum(pi_ub_value[i] * b_ub[i] for i in range(len(pi_ub_value)))
                #         - sum(pi_cut_value[i] * b_ub_cut[i] for i in range(len(pi_cut_value))))
                alpha_list.append(alpha)
                # self.logger.info(f"t: {t} cut_index: {i} alpha: {alpha}")

            # 找x，并计算截距，得到的完整的cut添加到问题中  i k t参数是(0, k, t - 1)
            self.logger.info("intercept_calculate...")
            self.intercept_calculate(t, alpha_list)


    def intercept_calculate(self, stage, alpha_list):
        """用于IFR算法中的截距计算，stage参数和后向的t保持一致，前向计算到stage-1，最后使用stage-1的x计算截距"""

        n_cuts = len(alpha_list)
        # 场景采样 和n_cuts相同数量
        n_samples = n_cuts
        samples = self.sc_sampler.generate_samples(n_samples)

        # forward
        fw_result = result_Dict("forward_result_temp")

        v_opt_k = []
        for k in range(n_samples):
            # t = -1对应的状态值
            x_trial_point = self.problem_params.init_x_trial_point
            y_trial_point = self.problem_params.init_y_trial_point
            x_bs_trial_point = self.problem_params.init_x_bs_trial_point
            soc_trial_point = self.problem_params.init_soc_trial_point
            v_opt_k.append(0)

            for t, n in zip(range(stage), samples[k]):  # 只需要计算到stage阶段即可，只需要x_stage， zip() 会自动截断到较短的那个序列的长度

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
                fw_result.add(0, k, t, stage_result)

        # SB 重算截距
        t = stage
        n_realizations = self.problem_params.n_realizations_per_stage[t]
        for k in range(n_samples):
            # 取出stage-1的x
            stage_result = fw_result.get_values(0, k, t - 1)
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
            # self.logger.info(f"trial_point {trial_point}")
            intercept_list = []
            dm = alpha_list[k]
            for n in range(n_realizations):
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

                intercept_n = dual_value - dm @ np.array(trial_point)
                intercept_list.append(intercept_n)
            intercept = mean(intercept_list)
            # 添加cut
            self.cuts_storage.add(0, k, t - 1, alpha_list[k] + [intercept])
            self.x_storage.add(0, k, t - 1, trial_point)

    def backward_benders_matrix(self, iteration: int, samples: list, stage: int = None, storage_flag: bool = False) -> None:
        i = iteration
        n_samples = len(samples)
        if stage is None:
            stages = reversed(range(1, self.problem_params.n_stages))  # 倒序计算所有阶段
        else:
            stages = [stage]  # 仅计算一个阶段
        for t in stages:
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
                    uc_fw, X_vector, A_eq, b_eq, A_ub, b_ub, A_ub_cut, b_ub_cut = (
                        self.add_problem_constraints_matrix(uc_fw, t, n)
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

                    if storage_flag:
                        # 获取对偶值
                        # 标准对偶值应该是从采样中选一个就行，驻点条件中并不包含场景信息，因此对于不同的n，得到所有的对偶值都是满足驻点条件的
                        dual_values_eq = []  # 平等约束
                        dual_values_ub = []  # 不等式约束
                        dual_values_cut = []  # 割约束
                        # 1. 获取平等约束的对偶值
                        for row in range(A_eq.shape[0]):
                            constr = uc_fw.model.getConstrByName(f"eq_constrains_{row}")
                            dual_values_eq.append(constr.Pi)
                        # 2. 获取不等式约束对偶值
                        for row in range(A_ub.shape[0]):
                            constr = uc_fw.model.getConstrByName(f"ub_constrains_{row}")
                            dual_values_ub.append(constr.Pi)
                        # 3. 获取不等式割约束对偶值
                        for row in range(A_ub_cut.shape[0]):
                            constr = uc_fw.model.getConstrByName(f"ub_cut_constrains_{row}")
                            dual_values_cut.append(constr.Pi)
                        self.dual_values_dict[(i, n, t)] = [dual_values_eq, dual_values_ub, dual_values_cut]

                    dm = []

                    copy_constrs = uc_fw.sddip_copy_constrs

                    try:
                        for constr in copy_constrs:
                            dm.append(constr.getAttr(gp.GRB.attr.Pi))
                    except AttributeError:
                        uc_fw.model.write("model.lp")
                        uc_fw.model.computeIIS()
                        uc_fw.model.write("model.ilp")
                        raise

                    dual_multipliers.append(dm)

                    # 输出解
                    # self.logger.info(f"stage: {t}  n: {n}")
                    # self.logger.info(f"x: {[x.x for x in uc_fw.x]}")
                    # self.logger.info(f"y: {[y.x for y in uc_fw.y]}")
                    # self.logger.info(f"x_bs: {[x.x for x_bs in uc_fw.x_bs for x in x_bs]}")
                    # self.logger.info(f"soc: {[soc.x for soc in uc_fw.soc]}")
                    # self.logger.info(f"socs_p: {[soc.x for soc in uc_fw.socs_p]}")
                    # self.logger.info(f"socs_n: {[soc.x for soc in uc_fw.socs_n]}")
                    # self.logger.info(f"theta: {uc_fw.theta.x}")
                    # self.logger.info(f"delta: {uc_fw.delta.x}")
                    #
                    # self.logger.info(f"dual: {dm}")

                    # if t == 1 and k == 0:
                    #     uc_fw.model.write("IFR_model.lp")

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
