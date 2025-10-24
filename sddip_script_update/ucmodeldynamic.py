from abc import ABC, abstractmethod

import gurobipy as gp
import numpy as np
from scipy import linalg
from scipy import sparse



class ModelBuilder(ABC):
    def __init__(
        self,
        n_buses: int,
        n_lines: int,
        n_generators: int,
        n_storages: int,
        generators_at_bus: list,
        storages_at_bus: list,
        backsight_periods: list,
        lp_relax: bool = False,
    ) -> None:
        self.n_buses = n_buses
        self.n_lines = n_lines
        self.n_generators = n_generators
        self.n_storages = n_storages
        self.generators_at_bus = generators_at_bus
        self.storages_at_bus = storages_at_bus
        self.backsight_periods = backsight_periods
        self.model = gp.Model("MILP: Unit commitment")
        self.model.setParam("OutputFlag", 0)
        self.model.setParam("IntFeasTol", 10 ** (-9))
        self.model.setParam("NumericFocus", 3)

        # Commitment decision
        self.x = []
        # Dispatch decision
        self.y = []
        # Generator state backsight variables
        # Given current stage t, x_bs[g][k] is the state of generator g at stage (t-k-1)
        self.x_bs = []
        self.x_bs_p = []
        self.x_bs_n = []
        # Storage charge/discharge
        self.ys_c = []
        self.ys_dc = []
        # Switch variable
        self.u_c_dc = []
        # SOC and slack
        self.soc = []
        self.socs_p = []
        self.socs_n = []
        # Copy variables
        self.z_x = []
        self.z_y = []
        self.z_x_bs = []
        self.z_soc = []
        # Startup decsision
        self.s_up = []
        # Shutdown decision
        self.s_down = []
        # Expected value function approximation
        self.theta = None
        # Positive slack
        self.ys_p = None
        # Negative slack
        self.ys_n = None

        # Objective
        self.objective_terms = None
        # Balance constraints
        self.balance_constraints = None
        # Copy constraints
        self.copy_constraints_x = None
        self.copy_constraints_y = None
        # Cut constraints
        self.cut_constraints = None
        # Cut lower bound
        self.cut_lower_bound = None
        # 变量偏移量dict
        self.var_offsets = None
        # 变量长度
        self.var_len = None

        self.bin_type = gp.GRB.CONTINUOUS if lp_relax else gp.GRB.BINARY

        self.initialize_variables()
        self.initialize_copy_variables()

    def initialize_variables(self) -> None:
        for g in range(self.n_generators):
            self.x.append(
                self.model.addVar(
                    vtype=self.bin_type, lb=0, ub=1, name="x_%i" % (g + 1)
                )
            )
            self.y.append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=0, name="y_%i" % (g + 1)
                )
            )
            self.x_bs.append(
                [
                    self.model.addVar(
                        vtype=self.bin_type,
                        lb=0,
                        ub=1,
                        name="x_bs_%i_%i" % (g + 1, k + 1),
                    )
                    for k in range(self.backsight_periods[g])
                ]
            )
            self.s_up.append(
                self.model.addVar(
                    vtype=self.bin_type, lb=0, ub=1, name="s_up_%i" % (g + 1)
                )
            )
            self.x_bs_p.append(
                [
                    self.model.addVar(
                        vtype=gp.GRB.CONTINUOUS,
                        lb=0,
                        ub=1,
                        name="x_bs_p_%i_%i" % (g + 1, k + 1),
                    )
                    for k in range(self.backsight_periods[g])
                ]
            )
            self.x_bs_n.append(
                [
                    self.model.addVar(
                        vtype=gp.GRB.CONTINUOUS,
                        lb=0,
                        ub=1,
                        name="x_bs_n_%i_%i" % (g + 1, k + 1),
                    )
                    for k in range(self.backsight_periods[g])
                ]
            )
            self.s_down.append(
                self.model.addVar(
                    vtype=self.bin_type, lb=0, ub=1, name="s_down_%i" % (g + 1)
                )
            )
        for s in range(self.n_storages):
            self.ys_c.append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=0, name=f"y_c_{s+1}"
                )
            )
            self.ys_dc.append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=0, name=f"y_dc_{s+1}"
                )
            )
            self.u_c_dc.append(
                self.model.addVar(
                    vtype=self.bin_type, lb=0, ub=1, name=f"u_{s+1}"
                )
            )
            self.soc.append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=0, name=f"soc_{s+1}"
                )
            )
            self.socs_p.append(
                self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name="socs_p")
            )
            self.socs_n.append(
                self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name="socs_n")
            )
        self.theta = self.model.addVar(
            vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="theta"
        )
        self.ys_p = self.model.addVar(
            vtype=gp.GRB.CONTINUOUS, lb=0, name="ys_p"
        )
        self.ys_n = self.model.addVar(
            vtype=gp.GRB.CONTINUOUS, lb=0, name="ys_n"
        )
        self.delta = self.model.addVar(
            vtype=gp.GRB.CONTINUOUS, lb=0, name="delta"
        )
        self.model.update()
        # self.model.addConstr(self.delta == 0)
        # self.model.addConstrs(self.socs_p[s] == 0 for s in range(self.n_storages))
        # self.model.addConstrs(self.socs_n[s] == 0 for s in range(self.n_storages))

    def initialize_copy_variables(self) -> None:
        for g in range(self.n_generators):
            self.z_x.append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS,lb=0, ub=1, name="z_x_%i" % (g + 1)
                )
            )
            self.z_y.append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=0, name="z_y_%i" % (g + 1)
                )
            )
            self.z_x_bs.append(
                [
                    self.model.addVar(
                        vtype=gp.GRB.CONTINUOUS,lb=0, ub=1,
                        name="z_x_bs_%i_%i" % (g + 1, k + 1),
                    )
                    for k in range(self.backsight_periods[g])
                ]
            )
        for s in range(self.n_storages):
            self.z_soc.append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=0, name="z_soc_%i" % (s + 1)
                )
            )
        self.model.update()


    def add_objective(self, coefficients: list):
        # x_bs_p = []
        # x_bs_n = []
        # for g in range(self.n_generators):
        #     x_bs_p += self.x_bs_p[g]
        x_bs_p = [x for g in range(self.n_generators) for x in self.x_bs_p[g]]
        x_bs_n = [x for g in range(self.n_generators) for x in self.x_bs_n[g]]

        penalty = coefficients[-1]

        coefficients = (
            coefficients
            + [penalty] * (2 * self.n_storages + 2 * len(x_bs_p) + 1)
            + [1]
        )

        variables = (
            self.y
            + self.s_up
            + self.s_down
            + [self.ys_p, self.ys_n]
            + self.socs_p
            + self.socs_n
            + x_bs_p
            + x_bs_n
            + [self.delta]
            + [self.theta]
        )
        self.objective_terms = gp.LinExpr(coefficients, variables)

        self.model.setObjective(self.objective_terms)
        self.update_model()

    def add_balance_constraints(
        self,
        total_demand: float,
        total_renewable_generation: float,
        discharge_eff: list,
    ) -> None:
        self.balance_constraints = self.model.addConstr(
            gp.quicksum(self.y)
            + gp.quicksum(
                discharge_eff[s] * self.ys_dc[s] - self.ys_c[s]
                for s in range(self.n_storages)
            )
            + self.ys_p
            - self.ys_n
            == total_demand - total_renewable_generation,
            "balance",
        )
        self.update_model()

    def add_generator_constraints(
        self, min_generation: list, max_generation: list
    ) -> None:
        self.model.addConstrs(
            (
                self.y[g] >= min_generation[g] * self.x[g] - self.delta
                for g in range(self.n_generators)
            ),
            "min-generation",
        )
        self.model.addConstrs(
            (
                self.y[g] <= max_generation[g] * self.x[g] + self.delta
                for g in range(self.n_generators)
            ),
            "max-generation",
        )

        self.update_model()

    def add_storage_constraints(
        self, max_charge_rate: list, max_discharge_rate: list, max_soc: list
    ) -> None:
        self.model.addConstrs(
            (
                self.ys_c[s] <= max_charge_rate[s] * self.u_c_dc[s]
                for s in range(self.n_storages)
            ),
            "max-charge-rate",
        )

        self.model.addConstrs(
            (
                self.ys_dc[s] <= max_discharge_rate[s] * (1 - self.u_c_dc[s])
                for s in range(self.n_storages)
            ),
            "max-discharge-rate",
        )

        self.model.addConstrs(
            (
                self.soc[s] <= max_soc[s] + self.delta
                for s in range(self.n_storages)
            ),
            "max-soc",
        )



    def add_soc_transfer(self, charge_eff: list) -> None:
        self.model.addConstrs(
            (
                self.soc[s]
                == self.z_soc[s]
                + charge_eff[s] * self.ys_c[s]
                - self.ys_dc[s]
                + self.socs_p[s]
                - self.socs_n[s]
                for s in range(self.n_storages)
            ),
            "soc",
        )

    def add_final_soc_constraints(self, final_soc: list) -> None:
        self.model.addConstrs(
            (
                self.soc[s] >= final_soc[s] - self.delta
                for s in range(self.n_storages)
            ),
            "final soc",
        )

    def add_power_flow_constraints(
        self,
        ptdf,
        max_line_capacities: list,
        demand: list,
        renewable_generation: list,
        discharge_eff: list,
    ) -> None:
        line_flows = [
            gp.quicksum(
                ptdf[l, b]
                * (
                    gp.quicksum(self.y[g] for g in self.generators_at_bus[b])
                    + gp.quicksum(
                        discharge_eff[s] * self.ys_dc[s] - self.ys_c[s]
                        for s in self.storages_at_bus[b]
                    )
                    - demand[b]
                    + renewable_generation[b]
                )
                for b in range(self.n_buses)
            )
            for l in range(self.n_lines)
        ]
        self.model.addConstrs(
            (
                line_flows[l] <= max_line_capacities[l] + self.delta
                for l in range(self.n_lines)
            ),
            "power-flow(1)",
        )
        self.model.addConstrs(
            (
                -line_flows[l] <= max_line_capacities[l] + self.delta
                for l in range(self.n_lines)
            ),
            "power-flow(2)",
        )
        self.update_model()

    def add_startup_shutdown_constraints(self) -> None:
        self.model.addConstrs(
            (
                self.x[g] - self.z_x[g] <= self.s_up[g] + self.delta
                for g in range(self.n_generators)
            ),
            "start-up",
        )
        self.model.addConstrs(
            (
                self.z_x[g] - self.x[g] <= self.s_down[g] + self.delta
                for g in range(self.n_generators)
            ),
            "shut-down",
        )
        self.update_model()

    def add_ramp_rate_constraints(
        self,
        max_rate_up: list,
        max_rate_down: list,
        startup_rate: list,
        shutdown_rate: list,
    ) -> None:
        self.model.addConstrs(
            (
                self.y[g] - self.z_y[g]
                <= max_rate_up[g] * self.z_x[g]
                + startup_rate[g] * self.s_up[g]
                + self.delta
                for g in range(self.n_generators)
            ),
            "rate-up",
        )
        self.model.addConstrs(
            (
                self.z_y[g] - self.y[g]
                <= max_rate_down[g] * self.x[g]
                + shutdown_rate[g] * self.s_down[g]
                + self.delta
                for g in range(self.n_generators)
            ),
            "rate-down",
        )

    def add_up_down_time_constraints(
        self, min_up_times: list, min_down_times: list
    ) -> None:
        self.model.addConstrs(
            (
                gp.quicksum(self.z_x_bs[g])
                >= min_up_times[g] * self.s_down[g] - self.delta
                for g in range(self.n_generators)
            ),
            "up-time",
        )

        self.model.addConstrs(
            (
                len(self.z_x_bs[g]) - gp.quicksum(self.z_x_bs[g])
                >= min_down_times[g] * self.s_up[g] - self.delta
                for g in range(self.n_generators)
            ),
            "down-time",
        )

        self.model.addConstrs(
            (
                self.z_x_bs[g][k]
                == self.x_bs[g][k] + self.x_bs_p[g][k] - self.x_bs_n[g][k]
                for g in range(self.n_generators)
                for k in range(self.backsight_periods[g])
            ),
            "backsight",
        )

    # TODO Adjust copy constraints to suit up- and down-time constraints
    @abstractmethod
    def add_copy_constraints(
        self, x_trial_point: list, y_trial_point: list, soc_trial_point: list
    ):
        pass

    def add_cut_lower_bound(self, lower_bound: float) -> None:
        self.cut_lower_bound = self.model.addConstr(
            (self.theta >= lower_bound), "cut-lb"
        )

    def _compute_var_offsets(self):
        """根据每个变量的长度计算偏移量，赋值到self.var_offsets"""
        var_len = {
            # 变量
            'x': self.n_generators,
            'y': self.n_generators,
            'x_bs': [self.backsight_periods[g] for g in range(len(self.backsight_periods))],
            'soc': self.n_storages,
            # 局部变量
            'x_bs_p': [self.backsight_periods[g] for g in range(len(self.backsight_periods))],
            'x_bs_n': [self.backsight_periods[g] for g in range(len(self.backsight_periods))],
            'ys_c': self.n_storages,
            'ys_dc': self.n_storages,
            'u_c_dc': self.n_storages,
            'socs_p': self.n_storages,
            'socs_n': self.n_storages,
            's_up': self.n_generators,
            's_down': self.n_generators,
            'ys_p': 1,
            'ys_n': 1,
            'delta': 1,
            # 复制变量
            'z_x': self.n_generators,
            'z_y': self.n_generators,
            'z_x_bs': [self.backsight_periods[g] for g in range(len(self.backsight_periods))],
            'z_soc': self.n_storages,
            # theta
            'theta': 1,
        }

        current_offset = 0
        self.var_offsets = {}
        for name, length in var_len.items():
            if isinstance(length, int):
                # 普通整数变量
                self.var_offsets[name] = current_offset
                current_offset += length

            elif isinstance(length, list):
                # 列表类型变量（如x_bs）
                offsets = []
                for sub_len in length:
                    offsets.append(current_offset)
                    current_offset += sub_len
                self.var_offsets[name] = offsets

            else:
                raise TypeError(f"Unsupported type for {name}: {type(length)}")

        self.var_len = current_offset

    def get_var_column_index(self):
        """获取变量所占用的列索引，使用这些变量可以从问题参数矩阵中抽取需要的部分"""
        var_column_dict = {}
        # [起始位置，结束位置） 注意结束位置应该是开区间
        var_column_dict['X'] = [self._get_var_index('x', 0), self._get_var_index('soc', self.n_storages - 1) + 1]
        var_column_dict['Y'] = [self._get_var_index('x_bs_p', 0, 0), self._get_var_index('delta', 0) + 1]
        var_column_dict['Z_X'] = [self._get_var_index('z_x', 0), self._get_var_index('z_soc', self.n_storages - 1) + 1]
        var_column_dict['theta'] = [self._get_var_index('theta', 0), self._get_var_index('theta', 0) + 1]

        return var_column_dict

    def _get_var_index(self, var_type, index=0, g=None)->int:
        """获取变量对应在向量中的索引位置"""
        if self.var_offsets is None:
            self._compute_var_offsets()

        if g is None:
            return self.var_offsets[var_type] + index
        else:
            return self.var_offsets[var_type][g] + index

    def _get_var_len(self):
        if self.var_len is None:
            self._compute_var_offsets()
        return self.var_len

    def get_problem_constrains_matrix(
            self,
            # balance_constraints
            total_demand: float,
            total_renewable_generation: float,
            discharge_eff: list,
            # generator_constraints
            min_generation: list,
            max_generation: list,
            # storage_constraints
            max_charge_rate: list,
            max_discharge_rate: list,
            max_soc: list,
            # soc_transfer
            charge_eff: list,
            # final_soc_constraints
            final_soc: list,  # 可能None
            # power_flow_constraints
            ptdf,
            max_line_capacities: list,
            demand: list,
            renewable_generation: list,
            # discharge_eff

            # startup_shutdown_constraints

            # ramp_rate_constraints
            max_rate_up: list,
            max_rate_down: list,
            startup_rate: list,
            shutdown_rate: list,
            # up_down_time_constraints
            min_up_times: list,
            min_down_times: list,
            # cut_lower_bound
            # lower_bound: float,
            cuts_list: list  # 可能None
    ):
        X_vector = self.get_var_vector()
        A_eq, b_eq = self.get_equality_constraints_matrix(
            total_demand,
            total_renewable_generation,
            discharge_eff,
            charge_eff,
        )
        A_ub, b_ub = self.get_inequality_constraints(
            min_generation,
            max_generation,
            max_charge_rate,
            max_discharge_rate,
            max_soc,
            final_soc,
            ptdf,
            max_line_capacities,
            demand,
            renewable_generation,
            discharge_eff,
            max_rate_up,
            max_rate_down,
            startup_rate,
            shutdown_rate,
            min_up_times,
            min_down_times,
            # lower_bound
        )
        A_ub_cut, b_ub_cut = self.get_inequality_cut_constrains(cuts_list)
        return X_vector, A_eq, b_eq, A_ub, b_ub, A_ub_cut, b_ub_cut

    def get_var_vector(self):
        """获取变量向量"""
        X_vector = []
        X_vector.extend(self.x)
        X_vector.extend(self.y)
        X_vector.extend([x for x_bs in self.x_bs for x in x_bs])
        X_vector.extend(self.soc)

        X_vector.extend([x for x_bs_p in self.x_bs_p for x in x_bs_p])
        X_vector.extend([x for x_bs_n in self.x_bs_n for x in x_bs_n])
        X_vector.extend(self.ys_c)
        X_vector.extend(self.ys_dc)
        X_vector.extend(self.u_c_dc)
        X_vector.extend(self.socs_p)
        X_vector.extend(self.socs_n)
        X_vector.extend(self.s_up)
        X_vector.extend(self.s_down)
        X_vector.append(self.ys_p)
        X_vector.append(self.ys_n)
        X_vector.append(self.delta)

        X_vector.extend(self.z_x)
        X_vector.extend(self.z_y)
        X_vector.extend([x for z_x_bs in self.z_x_bs for x in z_x_bs])
        X_vector.extend(self.z_soc)

        X_vector.append(self.theta)

        # print("X_vector: ", len(X_vector))  # 53
        # print(self._get_var_len())  # 53

        return X_vector

    def get_obj_coefficients(self, coefficients: list):
        """获取目标函数参数向量"""
        penalty = coefficients[-1]
        x_bs_p = [x for g in range(self.n_generators) for x in self.x_bs_p[g]]

        coefficients = (
                coefficients
                + [penalty] * (2 * self.n_storages + 2 * len(x_bs_p) + 1)
                + [1]
        )
        # variables = (
        #         self.y
        #         + self.s_up
        #         + self.s_down
        #         + [self.ys_p, self.ys_n]
        #         + self.socs_p
        #         + self.socs_n
        #         + x_bs_p
        #         + x_bs_n
        #         + [self.delta]
        #         + [self.theta]
        # )
        # 初始化系数矩阵和右侧向量
        A_obj_data = []
        A_obj_rows = []
        A_obj_cols = []
        # 约束对应行数
        current_row = 0
        index = 0  # coefficients索引
        for g in range(self.n_generators):
            A_obj_data.append(coefficients[index])
            A_obj_rows.append(current_row)
            A_obj_cols.append(self._get_var_index('y', g))
            index += 1
        for g in range(self.n_generators):
            A_obj_data.append(coefficients[index])
            A_obj_rows.append(current_row)
            A_obj_cols.append(self._get_var_index('s_up', g))
            index += 1
        for g in range(self.n_generators):
            A_obj_data.append(coefficients[index])
            A_obj_rows.append(current_row)
            A_obj_cols.append(self._get_var_index('s_down', g))
            index += 1

        A_obj_data.append(coefficients[index])
        A_obj_rows.append(current_row)
        A_obj_cols.append(self._get_var_index('ys_p', 0))
        index += 1

        A_obj_data.append(coefficients[index])
        A_obj_rows.append(current_row)
        A_obj_cols.append(self._get_var_index('ys_n', 0))
        index += 1

        for s in range(self.n_storages):
            A_obj_data.append(coefficients[index])
            A_obj_rows.append(current_row)
            A_obj_cols.append(self._get_var_index('socs_p', s))
            index += 1
        for s in range(self.n_storages):
            A_obj_data.append(coefficients[index])
            A_obj_rows.append(current_row)
            A_obj_cols.append(self._get_var_index('socs_n', s))
            index += 1
        for g in range(self.n_generators):
            for k in range(self.backsight_periods[g]):
                A_obj_data.append(-1.0)
                A_obj_rows.append(current_row)
                A_obj_cols.append(self._get_var_index('x_bs_p', k, g))
                index += 1
        for g in range(self.n_generators):
            for k in range(self.backsight_periods[g]):
                A_obj_data.append(-1.0)
                A_obj_rows.append(current_row)
                A_obj_cols.append(self._get_var_index('x_bs_n', k, g))
                index += 1

        A_obj_data.append(coefficients[index])
        A_obj_rows.append(current_row)
        A_obj_cols.append(self._get_var_index('delta', 0))
        index += 1

        A_obj_data.append(coefficients[index])
        A_obj_rows.append(current_row)
        A_obj_cols.append(self._get_var_index('theta', 0))
        index += 1

        current_row += 1

        # 创建稀疏矩阵
        A_obj = sparse.csr_matrix((A_obj_data, (A_obj_rows, A_obj_cols)),
                                 shape=(current_row, self._get_var_len()))
        A_obj = A_obj.toarray()
        return A_obj



    def get_equality_constraints_matrix(
            self,
            # balance_constraints
            total_demand,
            total_renewable_generation,
            discharge_eff,
            # soc_transfer
            charge_eff,

    ):
        """等式约束的矩阵形式：A_eq * v = b_eq"""



        # 初始化系数矩阵和右侧向量
        A_eq_data = []
        A_eq_rows = []
        A_eq_cols = []
        b_eq = []

        # 约束对应行数
        current_row = 0

        # 1. 平衡约束: gp.quicksum(self.y) +
        #             gp.quicksum(discharge_eff[s] * self.ys_dc[s] - self.ys_c[s] for s in range(self.n_storages))
        #             + self.ys_p
        #             - self.ys_n
        #             == total_demand - total_renewable_generation
        for g in range(self.n_generators):
            # y[g] 系数为 1
            A_eq_data.append(1.0)
            A_eq_rows.append(current_row)
            A_eq_cols.append(self._get_var_index('y', g))

        for s in range(self.n_storages):
            # ys_dc[s] 系数为 discharge_eff[s]
            A_eq_data.append(discharge_eff[s])
            A_eq_rows.append(current_row)
            A_eq_cols.append(self._get_var_index('ys_dc', s))

            # ys_c[s] 系数为 -1
            A_eq_data.append(-1.0)
            A_eq_rows.append(current_row)
            A_eq_cols.append(self._get_var_index('ys_c', s))

        # ys_p 系数为 1
        A_eq_data.append(1.0)
        A_eq_rows.append(current_row)
        A_eq_cols.append(self._get_var_index('ys_p'))

        # ys_n 系数为 -1
        A_eq_data.append(-1.0)
        A_eq_rows.append(current_row)
        A_eq_cols.append(self._get_var_index('ys_n'))

        b_eq.append(total_demand - total_renewable_generation)
        current_row += 1

        # 2. SOC转移约束:
        # self.soc[s] == self.z_soc[s]
        #             + charge_eff[s] * self.ys_c[s]
        #             - self.ys_dc[s]
        #             + self.socs_p[s]
        #             - self.socs_n[s]
        #             for s in range(self.n_storages)
        for s in range(self.n_storages):
            # soc[s] 系数为 1
            A_eq_data.append(1.0)
            A_eq_rows.append(current_row)
            A_eq_cols.append(self._get_var_index('soc', s))

            # z_soc[s] 系数为 -1
            A_eq_data.append(-1.0)
            A_eq_rows.append(current_row)
            A_eq_cols.append(self._get_var_index('z_soc', s))

            # ys_c[s] 系数为 -charge_eff[s]
            A_eq_data.append(-charge_eff[s])
            A_eq_rows.append(current_row)
            A_eq_cols.append(self._get_var_index('ys_c', s))

            # ys_dc[s] 系数为 1
            A_eq_data.append(1.0)
            A_eq_rows.append(current_row)
            A_eq_cols.append(self._get_var_index('ys_dc', s))

            # socs_p[s] 系数为 -1
            A_eq_data.append(-1.0)
            A_eq_rows.append(current_row)
            A_eq_cols.append(self._get_var_index('socs_p', s))

            # socs_n[s] 系数为 1
            A_eq_data.append(1.0)
            A_eq_rows.append(current_row)
            A_eq_cols.append(self._get_var_index('socs_n', s))

            b_eq.append(0.0)
            current_row += 1

        # 3. up_down_time_constraints中的后视变量约束:
        # self.z_x_bs[g][k]
        #        == self.x_bs[g][k] + self.x_bs_p[g][k] - self.x_bs_n[g][k]
        #        for g in range(self.n_generators)
        #        for k in range(self.backsight_periods[g])
        for g in range(self.n_generators):
            for k in range(self.backsight_periods[g]):
                # z_x_bs[g][k] 系数为 1
                A_eq_data.append(1.0)
                A_eq_rows.append(current_row)
                A_eq_cols.append(self._get_var_index('z_x_bs', k, g))

                # x_bs[g][k] 系数为 -1
                A_eq_data.append(-1.0)
                A_eq_rows.append(current_row)
                A_eq_cols.append(self._get_var_index('x_bs', k, g))

                # x_bs_p[g][k] 系数为 -1
                A_eq_data.append(-1.0)
                A_eq_rows.append(current_row)
                A_eq_cols.append(self._get_var_index('x_bs_p', k, g))

                # x_bs_n[g][k] 系数为 1
                A_eq_data.append(1.0)
                A_eq_rows.append(current_row)
                A_eq_cols.append(self._get_var_index('x_bs_n', k, g))

                b_eq.append(0.0)
                current_row += 1

        # 创建稀疏矩阵
        A_eq = sparse.csr_matrix((A_eq_data, (A_eq_rows, A_eq_cols)),
                                 shape=(current_row, self._get_var_len()))

        return A_eq, b_eq

    def get_inequality_constraints(
            self,
            # generator_constraints
            min_generation: list,
            max_generation: list,
            # storage_constraints
            max_charge_rate: list,
            max_discharge_rate: list,
            max_soc: list,
            # final_soc_constraints
            final_soc: list,
            # power_flow_constraints
            ptdf,
            max_line_capacities: list,
            demand: list,
            renewable_generation: list,
            discharge_eff,
            # startup_shutdown_constraints
            # ramp_rate_constraints
            max_rate_up: list,
            max_rate_down: list,
            startup_rate: list,
            shutdown_rate: list,
            # up_down_time_constraints
            min_up_times: list,
            min_down_times: list,
            # cut_lower_bound
            # lower_bound: float
    ):
        """不等式约束的矩阵形式：A_ub * v <= b_ub"""

        # 初始化系数矩阵和右侧向量
        A_ub_data = []
        A_ub_rows = []
        A_ub_cols = []
        b_ub = []

        current_row = 0

        # 1. generator_constraints 发电机最小发电量约束:
        # self.y[g] >= min_generation[g] * self.x[g] - self.delta
        #                 for g in range(self.n_generators)
        # 转换为: -y + min_generation * x - delta <= 0
        for g in range(self.n_generators):
            A_ub_data.append(-1.0)
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('y', g))

            A_ub_data.append(min_generation[g])
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('x', g))

            A_ub_data.append(-1.0)
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('delta'))

            b_ub.append(0)
            current_row += 1

        # 2. generator_constraints 发电机最大发电量约束
        # self.y[g] <= max_generation[g] * self.x[g] + self.delta
        #                 for g in range(self.n_generators)
        # 转换为: y - max_generation * x - delta <= 0
        for g in range(self.n_generators):
            A_ub_data.append(1.0)
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('y', g))

            A_ub_data.append(-max_generation[g])
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('x', g))

            A_ub_data.append(-1.0)
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('delta'))

            b_ub.append(0.0)
            current_row += 1

        # 3. storage_constraints 储能最大充电速率约束
        # self.ys_c[s] <= max_charge_rate[s] * self.u_c_dc[s]
        #                 for s in range(self.n_storages)
        # 转换为: ys_c - max_charge_rate * u_c_dc <= 0
        for s in range(self.n_storages):
            A_ub_data.append(1.0)  # ys_c[s]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('ys_c', s))

            A_ub_data.append(-max_charge_rate[s])  # -max_charge_rate[s] * u_c_dc[s]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('u_c_dc', s))

            b_ub.append(0.0)
            current_row += 1

        # 4. storage_constraints 储能最大放电速率约束
        # self.ys_dc[s] <= max_discharge_rate[s] * (1 - self.u_c_dc[s])
        #                 for s in range(self.n_storages)
        # 转换为: ys_dc + max_discharge_rate * u_c_dc <= max_discharge_rate
        for s in range(self.n_storages):
            A_ub_data.append(1.0)  # ys_dc[s]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('ys_dc', s))

            A_ub_data.append(max_discharge_rate[s])  # max_discharge_rate[s] * u_c_dc[s]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('u_c_dc', s))

            b_ub.append(max_discharge_rate[s])
            current_row += 1

        # 5. storage_constraints 最大SOC约束
        # self.soc[s] <= max_soc[s] + self.delta
        #                 for s in range(self.n_storages)
        for s in range(self.n_storages):
            A_ub_data.append(1.0)  # soc[s]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('soc', s))

            A_ub_data.append(-1.0)
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('delta'))

            b_ub.append(max_soc[s])
            current_row += 1

        # 6. final_soc_constraints 最终SOC约束
        # self.soc[s] >= final_soc[s] - self.delta
        #                 for s in range(self.n_storages)
        # 转换为: -soc - delta <= -final_soc
        if final_soc is not None:  # 只有在最后一个阶段才需要这个约束
            for s in range(self.n_storages):
                A_ub_data.append(-1.0)  # -soc[s]
                A_ub_rows.append(current_row)
                A_ub_cols.append(self._get_var_index('soc', s))

                A_ub_data.append(-1.0)
                A_ub_rows.append(current_row)
                A_ub_cols.append(self._get_var_index('delta'))

                b_ub.append(-final_soc[s])
                current_row += 1

        # 7. power_flow_constraints 功率流约束
        # max_line_capacities[l] + self.delta <= gp.quicksum(
        #                 ptdf[l, b]
        #                 * (
        #                     gp.quicksum(self.y[g] for g in self.generators_at_bus[b])
        #                     + gp.quicksum(
        #                         discharge_eff[s] * self.ys_dc[s] - self.ys_c[s]
        #                         for s in self.storages_at_bus[b]
        #                     )
        #                     - demand[b]
        #                     + renewable_generation[b]
        #                 )
        #                 for b in range(self.n_buses)
        #             ) <= max_line_capacities[l] + self.delta

        # (上界)  sum_b(
        #      sum_g(ptdf[l,b]*y_g)
        #   + sum_s(ptdf[l,b]*discharge_eff[s]*ys_dc_s)
        #   - sum_s(ptdf[l,b]*ys_c_s)
        #                   )
        #   - delta <= sum_b(ptdf[l, b] *demand[b]) - sum_b(ptdf[l,b]*renewable_generation[b]) + max_line_capacities[l]
        for l in range(self.n_lines):
            # -------- 上界约束 --------
            b_ub.append(0.0)
            for b in range(self.n_buses):
                ptdf_coeff = ptdf[l, b]

                # sum_g(ptdf[l,b]*y_g)
                for g in self.generators_at_bus[b]:
                    A_ub_data.append(ptdf_coeff)
                    A_ub_rows.append(current_row)
                    A_ub_cols.append(self._get_var_index('y', g))

                # + sum_s(ptdf[l,b]*discharge_eff[s]*ys_dc_s)
                # - sum_s(ptdf[l, b] * ys_c_s)
                for s in self.storages_at_bus[b]:
                    A_ub_data.append(ptdf_coeff * discharge_eff[s])
                    A_ub_rows.append(current_row)
                    A_ub_cols.append(self._get_var_index('ys_dc', s))

                    A_ub_data.append(-ptdf_coeff)
                    A_ub_rows.append(current_row)
                    A_ub_cols.append(self._get_var_index('ys_c', s))

                # sum_b(ptdf[l, b] *demand[b]) - sum_b(ptdf[l,b]*renewable_generation[b])
                b_ub[current_row] += ptdf_coeff * (demand[b] - renewable_generation[b])

            # -delta
            A_ub_data.append(-1.0)
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('delta'))

            # + max_line_capacities[l]
            b_ub[current_row] += max_line_capacities[l]
            current_row += 1

            # (下界)
            b_ub.append(0.0)
            for b in range(self.n_buses):
                ptdf_coeff = -ptdf[l, b]  # 取负号，后面都不用改
                # sum_g(ptdf[l, b] * y_g)
                for g in self.generators_at_bus[b]:
                    A_ub_data.append(ptdf_coeff)
                    A_ub_rows.append(current_row)
                    A_ub_cols.append(self._get_var_index('y', g))

                # + sum_s(ptdf[l,b]*discharge_eff[s]*ys_dc_s)
                # - sum_s(ptdf[l, b] * ys_c_s)
                for s in self.storages_at_bus[b]:
                    A_ub_data.append(ptdf_coeff * discharge_eff[s])
                    A_ub_rows.append(current_row)
                    A_ub_cols.append(self._get_var_index('ys_dc', s))

                    A_ub_data.append(-ptdf_coeff)
                    A_ub_rows.append(current_row)
                    A_ub_cols.append(self._get_var_index('ys_c', s))

                # sum_b(ptdf[l, b] *demand[b]) - sum_b(ptdf[l,b]*renewable_generation[b])
                b_ub[current_row] += ptdf_coeff * (demand[b] - renewable_generation[b])

            # -delta
            A_ub_data.append(-1.0)
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('delta'))

            # + max_line_capacities[l]
            b_ub[current_row] += max_line_capacities[l]
            current_row += 1

        # 8. startup_shutdown_constraints 启动约束
        # self.x[g] - self.z_x[g] <= self.s_up[g] + self.delta
        #                 for g in range(self.n_generators)
        # 转换为: x - z_x - s_up - delta <= 0
        for g in range(self.n_generators):
            A_ub_data.append(1.0)  # x[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('x', g))

            A_ub_data.append(-1.0)  # -z_x[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('z_x', g))

            A_ub_data.append(-1.0)  # -s_up[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('s_up', g))

            A_ub_data.append(-1.0)
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('delta'))

            b_ub.append(0.0)
            current_row += 1

        # 9. startup_shutdown_constraints 关闭约束
        # self.z_x[g] - self.x[g] <= self.s_down[g] + self.delta
        #                 for g in range(self.n_generators)
        # 转换为: -x + z_x - s_down <= delta
        for g in range(self.n_generators):
            A_ub_data.append(-1.0)  # -x[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('x', g))

            A_ub_data.append(1.0)  # z_x[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('z_x', g))

            A_ub_data.append(-1.0)  # -s_down[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('s_down', g))

            A_ub_data.append(-1.0)
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('delta'))

            b_ub.append(0.0)
            current_row += 1

        # 10. ramp_rate_constraints 上升速率约束
        # self.y[g] - self.z_y[g]
        #                 <= max_rate_up[g] * self.z_x[g]
        #                 + startup_rate[g] * self.s_up[g]
        #                 + self.delta
        #                 for g in range(self.n_generators)
        # 转换为: y - z_y - max_rate_up * z_x - startup_rate * s_up - delta <= 0
        for g in range(self.n_generators):
            A_ub_data.append(1.0)  # y[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('y', g))

            A_ub_data.append(-1.0)  # -z_y[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('z_y', g))

            A_ub_data.append(-max_rate_up[g])  # -max_rate_up[g] * z_x[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('z_x', g))

            A_ub_data.append(-startup_rate[g])  # -startup_rate[g] * s_up[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('s_up', g))

            A_ub_data.append(-1.0)
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('delta'))

            b_ub.append(0.0)
            current_row += 1

        # 11. ramp_rate_constraints 下降速率约束:
        # self.z_y[g] - self.y[g]
        #                 <= max_rate_down[g] * self.x[g]
        #                 + shutdown_rate[g] * self.s_down[g]
        #                 + self.delta
        #                 for g in range(self.n_generators)
        # 转换为: -y + z_y - max_rate_down * x - shutdown_rate * s_down - delta <= 0
        for g in range(self.n_generators):
            A_ub_data.append(-1.0)  # -y[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('y', g))

            A_ub_data.append(1.0)  # z_y[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('z_y', g))

            A_ub_data.append(-max_rate_down[g])  # -max_rate_down[g] * x[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('x', g))

            A_ub_data.append(-shutdown_rate[g])  # -shutdown_rate[g] * s_down[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('s_down', g))

            A_ub_data.append(-1.0)
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('delta'))

            b_ub.append(0.0)
            current_row += 1

        # 12. up_down_time_constraints 启动时间约束
        # gp.quicksum(self.z_x_bs[g])
        #                 >= min_up_times[g] * self.s_down[g] - self.delta
        #                 for g in range(self.n_generators)
        # 转换为: -sum(z_x_bs) + min_up_times * s_down - delta <= 0
        for g in range(self.n_generators):
            for k in range(self.backsight_periods[g]):
                A_ub_data.append(-1.0)  # -z_x_bs[g][k]
                A_ub_rows.append(current_row)
                A_ub_cols.append(self._get_var_index('z_x_bs', k, g))

            A_ub_data.append(min_up_times[g])  # min_up_times[g] * s_down[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('s_down', g))

            A_ub_data.append(-1.0)
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('delta'))

            b_ub.append(0.0)
            current_row += 1

        # 13. up_down_time_constraints 停机时间约束
        # len(self.z_x_bs[g]) - gp.quicksum(self.z_x_bs[g])
        #                 >= min_down_times[g] * self.s_up[g] - self.delta
        #                 for g in range(self.n_generators)
        # 转换为: sum(z_x_bs) + min_down_times * s_up - delta <= len(self.z_x_bs[g])
        for g in range(self.n_generators):
            for k in range(self.backsight_periods[g]):
                A_ub_data.append(1.0)  # z_x_bs[g][k]
                A_ub_rows.append(current_row)
                A_ub_cols.append(self._get_var_index('z_x_bs', k, g))

            A_ub_data.append(min_down_times[g])  # min_down_times[g] * s_up[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('s_up', g))

            A_ub_data.append(-1.0)
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('delta'))


            b_ub.append(len(self.z_x_bs[g]))
            current_row += 1

        # 14. 割下界约束: theta >= lower_bound
        # 转换为: -theta <= -lower_bound
        # 简单约束直接写，也不用放到lagrangian对偶问题里
        # A_ub_data.append(-1.0)  # -theta
        # A_ub_rows.append(current_row)
        # A_ub_cols.append(self._get_var_index('theta'))
        # b_ub.append(-lower_bound)
        # current_row += 1

        # 打印稀疏矩阵的shape
        # print(len(A_ub_data))  # 237
        # print(len(A_ub_rows))  # 237
        # print(len(A_ub_cols))  # 237
        # print(len(b_ub))  # 51
        # print(current_row)  # 51



        # 创建稀疏矩阵
        A_ub = sparse.csr_matrix((A_ub_data, (A_ub_rows, A_ub_cols)),
                                 shape=(current_row, self._get_var_len()))

        return A_ub, b_ub

    def get_inequality_cut_constrains(self, cuts_list):
        # 初始化系数矩阵和右侧向量
        A_ub_cut_data = []
        A_ub_cut_rows = []
        A_ub_cut_cols = []
        b_ub_cut = []

        current_row = 0
        # state_variables = (
        #             self.x
        #             + self.y
        #             + [var for gen_bs in self.x_bs for var in gen_bs]
        #             + self.soc
        #     )
        #     for id, cut in enumerate(cuts_list):
        #         pi = cut[:-1]
        #         intercept = cut[-1]
        #         self.model.addConstr(
        #             (
        #                     self.theta
        #                     >= intercept + gp.LinExpr(pi, state_variables)
        #             ),
        #             f"cut_{id}",
        #         )
        # 转换为：
        # -theta + sum(pi * state_variables) <= -intercept
        if cuts_list is not None:  # 最后一个阶段没有cut约束
            for cut in cuts_list:
                pi = cut[:-1]
                intercept = cut[-1]
                # theta
                A_ub_cut_data.append(-1.0)
                A_ub_cut_rows.append(current_row)
                A_ub_cut_cols.append(self._get_var_index('theta'))

                pi_index = 0
                # x
                for g in range(self.n_generators):
                    A_ub_cut_data.append(pi[pi_index])
                    A_ub_cut_rows.append(current_row)
                    A_ub_cut_cols.append(self._get_var_index('x', g))
                    pi_index += 1
                # y
                for g in range(self.n_generators):
                    A_ub_cut_data.append(pi[pi_index])
                    A_ub_cut_rows.append(current_row)
                    A_ub_cut_cols.append(self._get_var_index('y', g))
                    pi_index += 1
                # x_bs
                for g in range(self.n_generators):
                    for k in range(self.backsight_periods[g]):
                        A_ub_cut_data.append(pi[pi_index])
                        A_ub_cut_rows.append(current_row)
                        A_ub_cut_cols.append(self._get_var_index('x_bs', k, g))
                        pi_index += 1
                # soc
                for s in range(self.n_storages):
                    A_ub_cut_data.append(pi[pi_index])
                    A_ub_cut_rows.append(current_row)
                    A_ub_cut_cols.append(self._get_var_index('soc', s))
                    pi_index += 1

                b_ub_cut.append(-intercept)
                current_row += 1

        # 创建稀疏矩阵
        A_ub_cut = sparse.csr_matrix((A_ub_cut_data, (A_ub_cut_rows, A_ub_cut_cols)),
                                 shape=(current_row, self._get_var_len()))
        return A_ub_cut, b_ub_cut


    # 下面没有用到
    def add_benders_cuts(
        self,
        cut_intercepts: list,
        cut_gradients: list,
        trial_points: list,
    ) -> None:
        for intercept, gradient, trial_point in zip(
            cut_intercepts, cut_gradients, trial_points, strict=False
        ):
            self.add_benders_cut(intercept, gradient, trial_point)

    def add_benders_cut(
        self, cut_intercept: float, cut_gradient: list, trial_point: list
    ) -> None:
        state_variables = (
            self.x
            + self.y
            + [var for gen_bs in self.x_bs for var in gen_bs]
            + self.soc
        )
        n_state_variables = len(state_variables)

        if n_state_variables != len(trial_point):
            msg = "Number of state variables must be equal to the number of trial points."
            raise ValueError(msg)

        self.model.addConstr(
            (
                self.theta
                >= cut_intercept
                + gp.quicksum(
                    cut_gradient[i] * (state_variables[i] - trial_point[i])
                    for i in range(n_state_variables)
                )
            ),
            "benders-cut",
        )

    def add_cut_constraints(
        self,
        cut_intercepts: list,
        cut_gradients: list,
        y_binary_multipliers: list,
        soc_binary_multipliers: list,
        big_m: float,
        sos: bool = False,
    ) -> None:
        r = 0
        for intercept, gradient, y_multipliers, soc_multipliers in zip(
            cut_intercepts,
            cut_gradients,
            y_binary_multipliers,
            soc_binary_multipliers,
            strict=False,
        ):
            self.add_cut(
                intercept,
                gradient,
                y_multipliers,
                soc_multipliers,
                r,
                big_m,
                sos,
            )
            r += 1
        self.update_model()

    def add_cut(
        self,
        cut_intercept: float,
        cut_gradient: list,
        y_binary_multipliers: np.array,
        soc_binary_multipliers: np.array,
        id: int,
        big_m: float,
        sos: bool = False,
    ) -> None:
        x_binary_multipliers = linalg.block_diag(*[1] * len(self.x))
        n_total_backsight_variables = sum(self.backsight_periods)
        x_bs_binary_multipliers = linalg.block_diag(
            *[1] * n_total_backsight_variables
        )

        binary_multipliers = linalg.block_diag(
            x_binary_multipliers, y_binary_multipliers
        )

        if x_bs_binary_multipliers.size > 0:
            binary_multipliers = linalg.block_diag(
                binary_multipliers, x_bs_binary_multipliers
            )
        if soc_binary_multipliers.size > 0:
            binary_multipliers = linalg.block_diag(
                binary_multipliers, soc_binary_multipliers
            )

        n_var_approximations, n_binaries = binary_multipliers.shape

        ny = self.model.addVars(
            n_binaries, vtype=gp.GRB.CONTINUOUS, lb=0, name=f"ny_{id}"
        )
        my = self.model.addVars(
            n_binaries, vtype=gp.GRB.CONTINUOUS, lb=0, name=f"my_{id}"
        )
        eta = self.model.addVars(
            n_var_approximations,
            vtype=gp.GRB.CONTINUOUS,
            lb=-gp.GRB.INFINITY,
            name=f"eta_{id}",
        )
        lmda = self.model.addVars(
            n_binaries,
            vtype=gp.GRB.CONTINUOUS,
            lb=0,
            ub=1,
            name=f"lambda_{id}",
        )

        state_vars = (
            self.x
            + self.y
            + [variable for bs_vars in self.x_bs for variable in bs_vars]
            + self.soc
        )

        w = self.model.addVars(n_binaries, vtype=self.bin_type, name=f"w_{id}")
        u = self.model.addVars(n_binaries, vtype=self.bin_type, name=f"u_{id}")

        m2 = [big_m] * n_binaries
        m4 = [big_m] * n_binaries

        # Cut constraint
        self.model.addConstr(
            (
                self.theta
                >= cut_intercept
                + gp.quicksum(
                    lmda[i] * cut_gradient[i] for i in range(n_binaries)
                )
            ),
            f"cut_{id}",
        )

        # KKT conditions
        self.model.addConstrs(
            (
                -cut_gradient[j]
                - ny[j]
                + my[j]
                + gp.quicksum(
                    binary_multipliers[i, j] * eta[i]
                    for i in range(n_var_approximations)
                )
                == 0
                for j in range(n_binaries)
            ),
            f"KKT(1)_{id}",
        )

        self.model.addConstrs(
            (
                gp.quicksum(
                    binary_multipliers[i, j] * lmda[j]
                    for j in range(n_binaries)
                )
                - state_vars[i]
                == 0
                for i in range(n_var_approximations)
            ),
            f"KKT(2)_{id}",
        )

        if sos:
            for i in range(n_binaries):
                self.model.addGenConstrIndicator(w[i], True, ny[i] == 0)
                self.model.addGenConstrIndicator(u[i], True, my[i] == 0)
            # self.model.addConstrs(
            #     ((w[i] == 1) >> (ny[i] == 0) for i in range(n_binaries)), f"KKT(4)_{id}"
            # )
            # self.model.addConstrs(
            #     ((u[i] == 1) >> (my[i] == 0) for i in range(n_binaries)), f"KKT(6)_{id}"
            # )
        else:
            self.model.addConstrs(
                (ny[i] <= m2[i] * (1 - w[i]) for i in range(n_binaries)),
                f"KKT(4)_{id}",
            )
            self.model.addConstrs(
                (my[i] <= m4[i] * (1 - u[i]) for i in range(n_binaries)),
                f"KKT(6)_{id}",
            )

        self.model.addConstrs(
            (lmda[i] <= w[i] for i in range(n_binaries)), f"KKT(3)_{id}"
        )

        self.model.addConstrs(
            (lmda[i] - 1 >= -u[i] for i in range(n_binaries)), f"KKT(5)_{id}"
        )

    def remove(self, gurobi_objects) -> None:
        # Remove if gurobi_objects not None or not empty
        if gurobi_objects:
            self.model.remove(gurobi_objects)
            self.update_model

    def update_model(self) -> None:
        self.model.update()

    def disable_output(self) -> None:
        self.model.setParam("OutputFlag", 0)

    def enable_output(self) -> None:
        self.model.setParam("OutputFlag", 1)


class ForwardModelBuilder(ModelBuilder):
    def __init__(
        self,
        n_buses: int,
        n_lines: int,
        n_generators: int,
        n_storages: int,
        generators_at_bus: list,
        storages_at_bus: list,
        backsight_periods: list,
        lp_relax: bool = False,
    ) -> None:
        super().__init__(
            n_buses,
            n_lines,
            n_generators,
            n_storages,
            generators_at_bus,
            storages_at_bus,
            backsight_periods,
            lp_relax,
        )

    def add_copy_constraints(
        self,
        x_trial_point: list,
        y_trial_point: list,
        x_bs_trial_point: list[list],
        soc_trial_point: list,
    ) -> None:
        self.copy_constraints_x = self.model.addConstrs(
            (
                self.z_x[g] == x_trial_point[g]
                for g in range(self.n_generators)
            ),
            "copy-x",
        )
        self.copy_constraints_y = self.model.addConstrs(
            (
                self.z_y[g] == y_trial_point[g]
                for g in range(self.n_generators)
            ),
            "copy-y",
        )
        self.copy_constraints_x_bs = self.model.addConstrs(
            (
                self.z_x_bs[g][k] == x_bs_trial_point[g][k]
                for g in range(self.n_generators)
                for k in range(self.backsight_periods[g])
            ),
            "copy-x-bs",
        )
        self.copy_constraints_soc = self.model.addConstrs(
            (
                self.z_soc[s] == soc_trial_point[s]
                for s in range(self.n_storages)
            ),
            "copy-soc",
        )
        self.update_model()

    def get_copy_terms(
        self,
        x_trial_point: list,
        y_trial_point: list,
        x_bs_trial_point: list[list],
        soc_trial_point: list,
    ):
        copy_terms = [
            x_trial_point[g] - self.z_x[g] for g in range(self.n_generators)
        ]
        copy_terms += [
            y_trial_point[g] - self.z_y[g] for g in range(self.n_generators)
        ]
        copy_terms += [
            x_bs_trial_point[g][k] - self.z_x_bs[g][k]
            for g in range(self.n_generators)
            for k in range(self.backsight_periods[g])
        ]
        copy_terms += [
            soc_trial_point[s] - self.z_soc[s] for s in range(self.n_storages)
        ]

        return copy_terms


class BackwardModelBuilder(ModelBuilder):
    def __init__(
        self,
        n_buses: int,
        n_lines: int,
        n_generators: int,
        n_storages: int,
        generators_at_bus: list,
        storages_at_bus: list,
        backsight_periods: list,
        lp_relax: bool = False,
    ) -> None:
        super().__init__(
            n_buses,
            n_lines,
            n_generators,
            n_storages,
            generators_at_bus,
            storages_at_bus,
            backsight_periods,
            lp_relax,
        )

        self.n_x_trial_binaries = None
        self.n_y_trial_binaries = None
        self.n_x_bs_trial_binaries = None
        self.n_soc_trial_binaries = None

        self.relaxed_terms = []

        # Copy variable for binary variables
        self.x_bin_copy_vars = []
        self.y_bin_copy_vars = []
        self.x_bs_bin_copy_vars = []
        self.soc_bin_copy_vars = []

        # Copy constraints
        self.copy_constraints_x = None
        self.copy_constraints_y = None
        self.copy_constraints_x_bs = None
        self.copy_constraints_soc = None

    def add_relaxation(
        self,
        x_binary_trial_point: list,
        y_binary_trial_point: list,
        x_bs_binary_trial_point: list[list],
        soc_binary_trial_point: list,
    ) -> None:
        self.bin_copy_vars = []
        self.n_x_trial_binaries = len(x_binary_trial_point)
        self.n_y_trial_binaries = len(y_binary_trial_point)
        self.n_x_bs_trial_binaries = [
            len(trial_point) for trial_point in x_bs_binary_trial_point
        ]
        self.n_soc_trial_binaries = len(soc_binary_trial_point)

        for j in range(self.n_x_trial_binaries):
            self.x_bin_copy_vars.append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS,
                    lb=0,
                    ub=1,
                    name="x_bin_copy_var_%i" % (j + 1),
                )
            )

        for j in range(self.n_y_trial_binaries):
            self.y_bin_copy_vars.append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS,
                    lb=0,
                    ub=1,
                    name="y_bin_copy_var_%i" % (j + 1),
                )
            )

        j = 0
        for n_vars in self.n_x_bs_trial_binaries:
            self.x_bs_bin_copy_vars.append(
                [
                    self.model.addVar(
                        vtype=gp.GRB.CONTINUOUS,
                        lb=0,
                        ub=1,
                        name="x_bs_bin_copy_var_%i" % (k + 1),
                    )
                    for k in range(j, j + n_vars)
                ]
            )
            j += n_vars

        for j in range(self.n_soc_trial_binaries):
            self.soc_bin_copy_vars.append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS,
                    lb=0,
                    ub=1,
                    name="soc_bin_copy_var_%i" % (j + 1),
                )
            )

        self.relax(
            x_binary_trial_point,
            y_binary_trial_point,
            x_bs_binary_trial_point,
            soc_binary_trial_point,
        )

    def relax(
        self,
        x_binary_trial_point: list,
        y_binary_trial_point: list,
        x_bs_binary_trial_point: list[list],
        soc_binary_trial_point: list,
    ) -> None:
        self.check_bin_copy_vars_not_empty()

        self.relaxed_terms = [
            x_binary_trial_point[j] - self.x_bin_copy_vars[j]
            for j in range(self.n_x_trial_binaries)
        ]

        self.relaxed_terms += [
            y_binary_trial_point[j] - self.y_bin_copy_vars[j]
            for j in range(self.n_y_trial_binaries)
        ]

        self.relaxed_terms += [
            x_bs_binary_trial_point[g][k] - self.x_bs_bin_copy_vars[g][k]
            for g in range(len(self.n_x_bs_trial_binaries))
            for k in range(self.n_x_bs_trial_binaries[g])
        ]

        self.relaxed_terms += [
            soc_binary_trial_point[j] - self.soc_bin_copy_vars[j]
            for j in range(self.n_soc_trial_binaries)
        ]

    def add_copy_constraints(
        self,
        y_binary_trial_multipliers: np.array,
        soc_binary_trial_multipliers: np.array,
    ) -> None:
        self.check_bin_copy_vars_not_empty()

        n_y_var_approximations, n_y_binaries = y_binary_trial_multipliers.shape
        n_soc_var_approximations = 0
        n_soc_binaries = 0
        if soc_binary_trial_multipliers.size > 0:
            (
                n_soc_var_approximations,
                n_soc_binaries,
            ) = soc_binary_trial_multipliers.shape

        self.copy_constraints_y = self.model.addConstrs(
            (
                self.z_y[i]
                == gp.quicksum(
                    y_binary_trial_multipliers[i, j] * self.y_bin_copy_vars[j]
                    for j in range(n_y_binaries)
                )
                for i in range(n_y_var_approximations)
            ),
            "copy-y",
        )

        self.copy_constraints_x = self.model.addConstrs(
            (
                self.z_x[i] == self.x_bin_copy_vars[i]
                for i in range(self.n_x_trial_binaries)
            ),
            "copy-x",
        )

        self.copy_constraints_x_bs = self.model.addConstrs(
            (
                self.z_x_bs[g][k] == self.x_bs_bin_copy_vars[g][k]
                for g in range(len(self.n_x_bs_trial_binaries))
                for k in range(self.n_x_bs_trial_binaries[g])
            ),
            "copy-x-bs",
        )

        self.copy_constraints_soc = self.model.addConstrs(
            (
                self.z_soc[i]
                == gp.quicksum(
                    soc_binary_trial_multipliers[i, j]
                    * self.soc_bin_copy_vars[j]
                    for j in range(n_soc_binaries)
                )
                for i in range(n_soc_var_approximations)
            ),
            "copy-soc",
        )
        self.update_model()

    def check_bin_copy_vars_not_empty(self) -> None:
        if not (
            self.x_bin_copy_vars
            and self.y_bin_copy_vars
            and self.x_bs_bin_copy_vars
        ):
            msg = "Copy variable does not exist. Call add_relaxation first."
            raise ValueError(msg)
