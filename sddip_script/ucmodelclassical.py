import logging
import gurobipy as gp

from .ucmodeldynamic import BackwardModelBuilder

logger = logging.getLogger(__name__)


class ClassicalModel(BackwardModelBuilder):
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
        copy_lp_relax: bool = True,  # Z设置为默认连续

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
            copy_lp_relax,
        )

        self.lp_relax = lp_relax

        # inner_model need
        self.X_trial_point = None
        self.theta_trial_point = None
        # self.pi_hat = None
        # self.pi0_hat = None
        # 令复制变量和trial_point值相等的约束
        self.sddip_copy_constrs = []

    # 计算松弛 z - x_hat
    def calculate_relaxed_terms(
        self,
        x_trial_point: list,
        y_trial_point: list,
        x_bs_trial_point: list[list],
        soc_trial_point: list,
    ):
        self.relaxed_terms = []
        self.relaxed_terms += [
            self.z_x[j] -x_trial_point[j]
            for j in range(len(x_trial_point))
        ]
        self.relaxed_terms += [
            self.z_y[j] - y_trial_point[j]
            for j in range(len(y_trial_point))
        ]
        self.relaxed_terms += [
            self.z_x_bs[g][k] - x_bs_trial_point[g][k]
            for g in range(len(x_bs_trial_point))
            for k in range(len(x_bs_trial_point[g]))
        ]
        self.relaxed_terms += [
            self.z_soc[j] - soc_trial_point[j]
            for j in range(len(soc_trial_point))
        ]

    def zero_relaxed_terms(self):
        for i, term in enumerate(self.relaxed_terms):
            self.sddip_copy_constrs.append(
                self.model.addConstr(term == 0, f"zero_relaxed_terms{i+1}")
            )
        self.model.update()
        return
    # dual_solver使用，获取松弛项
    def get_relaxed_terms(self, coefficients: list, theta_trail_point):
        x_bs_p = [x for g in range(self.n_generators) for x in self.x_bs_p[g]]
        x_bs_n = [x for g in range(self.n_generators) for x in self.x_bs_n[g]]

        penalty = coefficients[-1]
        coefficients = (
                coefficients + [penalty] * (2 * self.n_storages + 2 * len(x_bs_p) + 1) + [1]
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
        )  # theta对应cut
        self.theta_relaxed_terms = gp.LinExpr(coefficients, variables) - theta_trail_point
        return self.relaxed_terms + [self.theta_relaxed_terms]

    def add_inner_objective(self, coefficients: list, pi_hat, pi0_hat):
        x_bs_p = [x for g in range(self.n_generators) for x in self.x_bs_p[g]]
        x_bs_n = [x for g in range(self.n_generators) for x in self.x_bs_n[g]]

        penalty = coefficients[-1]
        coefficients = (
                coefficients + [penalty] * (2 * self.n_storages + 2 * len(x_bs_p) + 1) + [1]
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
        )  # theta对应cut
        theta = gp.LinExpr(coefficients, variables)

        z_X = (
                self.z_x
                + self.z_y
                + [val for bs in self.z_x_bs for val in bs]
                + self.z_soc
        )

        self.model.setObjective(gp.LinExpr(pi_hat, z_X) + pi0_hat * theta)
        self.model.update()

    # def add_sddip_copy_constraints(
    #     self,
    #     x_trial_point: list,
    #     y_trial_point: list,
    #     x_bs_trial_point: list[list],
    #     soc_trial_point: list,
    #     pg_max: list,
    #     soc_max: list,
    # ):
    #     self.add_relaxation(
    #         x_trial_point,
    #         y_trial_point,
    #         x_bs_trial_point,
    #         soc_trial_point,
    #         pg_max,
    #         soc_max,
    #     )
    #     self.sddip_copy_constrs = []
    #     #二进制变量与松弛变量差值为0
    #     for term in self.relaxed_terms:
    #         self.sddip_copy_constrs.append(
    #             self.model.addConstr(term == 0, "sddip-copy")
    #         )
    #
    # def relax_sddip_copy_constraints(
    #     self,
    #     x_binary_trial_point: list,
    #     y_binary_trial_point: list,
    #     x_bs_binary_trial_point: list[list],
    #     soc_binary_trial_point: list,
    # ):
    #     self.add_relaxation(
    #         x_binary_trial_point,
    #         y_binary_trial_point,
    #         x_bs_binary_trial_point,
    #         soc_binary_trial_point,
    #     )

    def add_lag_cut_constrains(
        self,
        lag_cuts_list
    ):
        state_variables = (
                self.x
                + self.y
                + [var for gen_bs in self.x_bs for var in gen_bs]
                + self.soc
            )

        for id, lag_cut in enumerate(lag_cuts_list):
            lag_pi = lag_cut[:-1]
            lag_intercept = lag_cut[-1]
            self.model.addConstr(
                (
                    self.theta
                    >= lag_intercept + gp.LinExpr(lag_pi, state_variables)
                ),
                f"lag_cut_{id}",
            )
        self.model.update()

    def add_benders_cut_constrains(
            self,
            benders_cuts_list
    ):
        state_variables = (
            self.x
            + self.y
            + [var for gen_bs in self.x_bs for var in gen_bs]
            + self.soc
        )

        for id, benders_cut in enumerate(benders_cuts_list):
            benders_pi = benders_cut[:-1]
            benders_intercept = benders_cut[-1]
            self.model.addConstr(
                (
                    self.theta >= benders_intercept + gp.LinExpr(benders_pi, state_variables)
                ),
                f"lag_cut_{id}",
            )
        self.model.update()






