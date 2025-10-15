from abc import ABC, abstractmethod

import gurobipy as gp
import numpy as np
from scipy import linalg


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


    def add_objective(self, coefficients: list) -> None:
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


    def _get_var_index(self, var_type, index=None):

        n_x_bs = sum(self.backsight_periods)
        var_len = {
            'x': self.n_generators,
            'y': self.n_generators,
            'x_bs': n_x_bs,
            'x_bs_p': n_x_bs,
            'x_bs_n': n_x_bs,
            'ys_c': self.n_storages,
            'ys_dc': self.n_storages,
            'u_c_dc': self.n_storages,
            'z_y': 4 * self.n_generators + 2 * sum(self.backsight_periods),
            'z_x_bs': 4 * self.n_generators,
            'soc': 6 * self.n_generators + self.n_storage,
            'ys_P': self.ys_p,
        }

        var_offsets = {}



        """
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
        """






        # if index is None:
        #     return var_offsets[var_type]
        # else:
        #     return var_offsets[var_type] + index

    def get_equality_constraints(self):
        """等式约束的矩阵形式：A_eq * v = b_eq"""

        # 获取变量维度
        n_g = self.n_generators
        n_s = self.n_storages
        n_bp_total = sum(self.backsight_periods)

        # 计算总变量数
        total_vars = (n_g * 4 +  # x, y, z_x, z_y
                      n_s * 8 +  # ys_c, ys_dc, u_c_dc, soc, socs_p, socs_n, z_soc, z_soc
                      n_bp_total * 4 +  # x_bs, x_bs_p, x_bs_n, z_x_bs
                      n_g * 2 +  # s_up, s_down
                      3)  # theta, ys_p, ys_n

        # 初始化约束计数器
        eq_constraint_count = 0

        # 1. 平衡约束 (1个)
        eq_constraint_count += 1

        # 2. SOC转移约束 (n_s个)
        eq_constraint_count += n_s

        # 3. 后视变量约束 (n_bp_total个)
        eq_constraint_count += n_bp_total

        # 初始化系数矩阵和右侧向量
        A_eq_data = []
        A_eq_rows = []
        A_eq_cols = []
        b_eq = np.zeros(eq_constraint_count)

        current_row = 0

        # 1. 平衡约束: sum(y) + sum(discharge_eff * ys_dc - ys_c) + ys_p - ys_n = total_demand - total_renewable
        for g in range(n_g):
            # y[g] 系数为 1
            A_eq_data.append(1.0)
            A_eq_rows.append(current_row)
            A_eq_cols.append(self._get_var_index('y', g))

        for s in range(n_s):
            # ys_dc[s] 系数为 discharge_eff[s]
            A_eq_data.append(self.discharge_eff[s])
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

        b_eq[current_row] = self.total_demand - self.total_renewable_generation
        current_row += 1

        # 2. SOC转移约束: soc = z_soc + charge_eff * ys_c - ys_dc + socs_p - socs_n
        for s in range(n_s):
            # soc[s] 系数为 1
            A_eq_data.append(1.0)
            A_eq_rows.append(current_row)
            A_eq_cols.append(self._get_var_index('soc', s))

            # z_soc[s] 系数为 -1
            A_eq_data.append(-1.0)
            A_eq_rows.append(current_row)
            A_eq_cols.append(self._get_var_index('z_soc', s))

            # ys_c[s] 系数为 -charge_eff[s]
            A_eq_data.append(-self.charge_eff[s])
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

            b_eq[current_row] = 0
            current_row += 1

        # 3. 后视变量约束: z_x_bs = x_bs + x_bs_p - x_bs_n
        bs_index = 0
        for g in range(n_g):
            for k in range(self.backsight_periods[g]):
                # z_x_bs[g][k] 系数为 1
                A_eq_data.append(1.0)
                A_eq_rows.append(current_row)
                A_eq_cols.append(self._get_var_index('z_x_bs', bs_index))

                # x_bs[g][k] 系数为 -1
                A_eq_data.append(-1.0)
                A_eq_rows.append(current_row)
                A_eq_cols.append(self._get_var_index('x_bs', bs_index))

                # x_bs_p[g][k] 系数为 -1
                A_eq_data.append(-1.0)
                A_eq_rows.append(current_row)
                A_eq_cols.append(self._get_var_index('x_bs_p', bs_index))

                # x_bs_n[g][k] 系数为 1
                A_eq_data.append(1.0)
                A_eq_rows.append(current_row)
                A_eq_cols.append(self._get_var_index('x_bs_n', bs_index))

                b_eq[current_row] = 0
                current_row += 1
                bs_index += 1

        # 创建稀疏矩阵
        A_eq = sparse.csr_matrix((A_eq_data, (A_eq_rows, A_eq_cols)),
                                 shape=(eq_constraint_count, total_vars))

        return A_eq, b_eq

    def get_inequality_constraints(self):
        """不等式约束的矩阵形式：A_ub * v <= b_ub"""

        # 获取变量维度
        n_g = self.n_generators
        n_s = self.n_storages
        n_lines = self.n_lines
        n_bp_total = sum(self.backsight_periods)

        # 计算总变量数
        total_vars = (n_g * 4 +  # x, y, z_x, z_y
                      n_s * 8 +  # ys_c, ys_dc, u_c_dc, soc, socs_p, socs_n, z_soc, z_soc
                      n_bp_total * 4 +  # x_bs, x_bs_p, x_bs_n, z_x_bs
                      n_g * 2 +  # s_up, s_down
                      3)  # theta, ys_p, ys_n

        # 初始化约束计数器
        ub_constraint_count = 0

        # 1. 发电机最小发电量约束 (n_g个)
        ub_constraint_count += n_g

        # 2. 发电机最大发电量约束 (n_g个)
        ub_constraint_count += n_g

        # 3. 储能最大充电速率约束 (n_s个)
        ub_constraint_count += n_s

        # 4. 储能最大放电速率约束 (n_s个)
        ub_constraint_count += n_s

        # 5. 最大SOC约束 (n_s个)
        ub_constraint_count += n_s

        # 6. 最终SOC约束 (n_s个)
        ub_constraint_count += n_s

        # 7. 功率流约束 (2 * n_lines个)
        ub_constraint_count += 2 * n_lines

        # 8. 启动约束 (n_g个)
        ub_constraint_count += n_g

        # 9. 关闭约束 (n_g个)
        ub_constraint_count += n_g

        # 10. 上升速率约束 (n_g个)
        ub_constraint_count += n_g

        # 11. 下降速率约束 (n_g个)
        ub_constraint_count += n_g

        # 12. 上线时间约束 (n_g个)
        ub_constraint_count += n_g

        # 13. 下线时间约束 (n_g个)
        ub_constraint_count += n_g

        # 14. 割下界约束 (1个)
        ub_constraint_count += 1

        # 初始化系数矩阵和右侧向量
        A_ub_data = []
        A_ub_rows = []
        A_ub_cols = []
        b_ub = np.zeros(ub_constraint_count)

        current_row = 0

        # 1. 发电机最小发电量约束: y >= min_generation * x - delta
        # 转换为: -y + min_generation * x <= delta
        for g in range(n_g):
            A_ub_data.append(-1.0)  # -y[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('y', g))

            A_ub_data.append(self.min_generation[g])  # min_generation[g] * x[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('x', g))

            b_ub[current_row] = self.delta
            current_row += 1

        # 2. 发电机最大发电量约束: y <= max_generation * x + delta
        # 转换为: y - max_generation * x <= delta
        for g in range(n_g):
            A_ub_data.append(1.0)  # y[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('y', g))

            A_ub_data.append(-self.max_generation[g])  # -max_generation[g] * x[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('x', g))

            b_ub[current_row] = self.delta
            current_row += 1

        # 3. 储能最大充电速率约束: ys_c <= max_charge_rate * u_c_dc
        # 转换为: ys_c - max_charge_rate * u_c_dc <= 0
        for s in range(n_s):
            A_ub_data.append(1.0)  # ys_c[s]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('ys_c', s))

            A_ub_data.append(-self.max_charge_rate[s])  # -max_charge_rate[s] * u_c_dc[s]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('u_c_dc', s))

            b_ub[current_row] = 0
            current_row += 1

        # 4. 储能最大放电速率约束: ys_dc <= max_discharge_rate * (1 - u_c_dc)
        # 转换为: ys_dc + max_discharge_rate * u_c_dc <= max_discharge_rate
        for s in range(n_s):
            A_ub_data.append(1.0)  # ys_dc[s]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('ys_dc', s))

            A_ub_data.append(self.max_discharge_rate[s])  # max_discharge_rate[s] * u_c_dc[s]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('u_c_dc', s))

            b_ub[current_row] = self.max_discharge_rate[s]
            current_row += 1

        # 5. 最大SOC约束: soc <= max_soc + delta
        for s in range(n_s):
            A_ub_data.append(1.0)  # soc[s]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('soc', s))

            b_ub[current_row] = self.max_soc[s] + self.delta
            current_row += 1

        # 6. 最终SOC约束: soc >= final_soc - delta
        # 转换为: -soc <= -final_soc + delta
        for s in range(n_s):
            A_ub_data.append(-1.0)  # -soc[s]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('soc', s))

            b_ub[current_row] = -self.final_soc[s] + self.delta
            current_row += 1

        # 7. 功率流约束 (需要PTDF计算，这里简化表示)
        for l in range(n_lines):
            # 正向功率流约束
            # 这里需要根据PTDF矩阵计算系数，简化表示
            A_ub_data.append(1.0)  # 简化系数
            A_ub_rows.append(current_row)
            A_ub_cols.append(0)  # 简化位置

            b_ub[current_row] = self.max_line_capacities[l] + self.delta
            current_row += 1

            # 反向功率流约束
            A_ub_data.append(-1.0)  # 简化系数
            A_ub_rows.append(current_row)
            A_ub_cols.append(0)  # 简化位置

            b_ub[current_row] = self.max_line_capacities[l] + self.delta
            current_row += 1

        # 8. 启动约束: x - z_x <= s_up + delta
        # 转换为: x - z_x - s_up <= delta
        for g in range(n_g):
            A_ub_data.append(1.0)  # x[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('x', g))

            A_ub_data.append(-1.0)  # -z_x[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('z_x', g))

            A_ub_data.append(-1.0)  # -s_up[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('s_up', g))

            b_ub[current_row] = self.delta
            current_row += 1

        # 9. 关闭约束: z_x - x <= s_down + delta
        # 转换为: -x + z_x - s_down <= delta
        for g in range(n_g):
            A_ub_data.append(-1.0)  # -x[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('x', g))

            A_ub_data.append(1.0)  # z_x[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('z_x', g))

            A_ub_data.append(-1.0)  # -s_down[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('s_down', g))

            b_ub[current_row] = self.delta
            current_row += 1

        # 10. 上升速率约束: y - z_y <= max_rate_up * z_x + startup_rate * s_up + delta
        # 转换为: y - z_y - max_rate_up * z_x - startup_rate * s_up <= delta
        for g in range(n_g):
            A_ub_data.append(1.0)  # y[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('y', g))

            A_ub_data.append(-1.0)  # -z_y[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('z_y', g))

            A_ub_data.append(-self.max_rate_up[g])  # -max_rate_up[g] * z_x[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('z_x', g))

            A_ub_data.append(-self.startup_rate[g])  # -startup_rate[g] * s_up[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('s_up', g))

            b_ub[current_row] = self.delta
            current_row += 1

        # 11. 下降速率约束: z_y - y <= max_rate_down * x + shutdown_rate * s_down + delta
        # 转换为: -y + z_y - max_rate_down * x - shutdown_rate * s_down <= delta
        for g in range(n_g):
            A_ub_data.append(-1.0)  # -y[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('y', g))

            A_ub_data.append(1.0)  # z_y[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('z_y', g))

            A_ub_data.append(-self.max_rate_down[g])  # -max_rate_down[g] * x[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('x', g))

            A_ub_data.append(-self.shutdown_rate[g])  # -shutdown_rate[g] * s_down[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('s_down', g))

            b_ub[current_row] = self.delta
            current_row += 1

        # 12. 上线时间约束: sum(z_x_bs) >= min_up_times * s_down - delta
        # 转换为: -sum(z_x_bs) + min_up_times * s_down <= delta
        bs_index = 0
        for g in range(n_g):
            for k in range(self.backsight_periods[g]):
                A_ub_data.append(-1.0)  # -z_x_bs[g][k]
                A_ub_rows.append(current_row)
                A_ub_cols.append(self._get_var_index('z_x_bs', bs_index))
                bs_index += 1

            A_ub_data.append(self.min_up_times[g])  # min_up_times[g] * s_down[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('s_down', g))

            b_ub[current_row] = self.delta
            current_row += 1

        # 13. 下线时间约束: (periods - sum(z_x_bs)) >= min_down_times * s_up - delta
        # 转换为: sum(z_x_bs) - min_down_times * s_up <= periods - delta
        bs_index = 0
        for g in range(n_g):
            for k in range(self.backsight_periods[g]):
                A_ub_data.append(1.0)  # z_x_bs[g][k]
                A_ub_rows.append(current_row)
                A_ub_cols.append(self._get_var_index('z_x_bs', bs_index))
                bs_index += 1

            A_ub_data.append(-self.min_down_times[g])  # -min_down_times[g] * s_up[g]
            A_ub_rows.append(current_row)
            A_ub_cols.append(self._get_var_index('s_up', g))

            b_ub[current_row] = self.backsight_periods[g] - self.delta
            current_row += 1

        # 14. 割下界约束: theta >= lower_bound
        # 转换为: -theta <= -lower_bound
        A_ub_data.append(-1.0)  # -theta
        A_ub_rows.append(current_row)
        A_ub_cols.append(self._get_var_index('theta'))

        b_ub[current_row] = -self.lower_bound

        # 创建稀疏矩阵
        # A_ub = sparse.csr_matrix((A_ub_data, (A_ub_rows, A_ub_cols)),
        #                          shape=(ub_constraint_count, total_vars))

        # return A_ub, b_ub





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
