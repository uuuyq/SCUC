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
        self.lp_relax = lp_relax

    def binary_approximation(
        self, y_bin_multipliers, soc_bin_multipliers
    ) -> None:
        self.y_bin_states = []
        self.soc_bin_states = []
        n_y_bin_vars = [len(bin_mult) for bin_mult in y_bin_multipliers]
        n_soc_bin_vars = [len(bin_mult) for bin_mult in soc_bin_multipliers]

        var_type = gp.GRB.CONTINUOUS if self.lp_relax else gp.GRB.BINARY

        g = 1
        for n_y in n_y_bin_vars:
            self.y_bin_states.append(
                self.model.addVars(
                    n_y, vtype=var_type, lb=0, ub=1, name=f"y_bin_{g}"
                )
            )
            g += 1

        s = 1
        for n_soc in n_soc_bin_vars:
            self.soc_bin_states.append(
                self.model.addVars(
                    n_soc, vtype=var_type, lb=0, ub=1, name=f"soc_bin_{s}"
                )
            )
            s += 1

        self.y_bin_states_flattened = [
            y_bin_var
            for y_tuple_dict in self.y_bin_states
            for y_bin_var in y_tuple_dict.values()
        ]

        self.soc_bin_states_flattened = [
            soc_bin_var
            for soc_tuple_dict in self.soc_bin_states
            for soc_bin_var in soc_tuple_dict.values()
        ]

        self.model.addConstrs(
            (
                gp.LinExpr(
                    y_bin_multipliers[g], self.y_bin_states[g].select("*")
                )
                == self.y[g]
                for g in range(self.n_generators)
            ),
            name="y_bin_appr",
        )

        self.model.addConstrs(
            (
                gp.LinExpr(
                    soc_bin_multipliers[s], self.soc_bin_states[s].select("*")
                )
                == self.soc[s]
                for s in range(self.n_storages)
            ),
            name="soc_bin_appr",
        )
        self.update_model()

    def add_sddip_copy_constraints(
        self,
        x_binary_trial_point: list,
        y_binary_trial_point: list,
        x_bs_binary_trial_point: list[list],
        soc_binary_trial_point: list,
    ) -> None:
        self.add_relaxation(
            x_binary_trial_point,
            y_binary_trial_point,
            x_bs_binary_trial_point,
            soc_binary_trial_point,
        )
        self.sddip_copy_constrs = []
        for term in self.relaxed_terms:
            self.sddip_copy_constrs.append(
                self.model.addConstr(term == 0, "sddip-copy")
            )

    def relax_sddip_copy_constraints(
        self,
        x_binary_trial_point: list,
        y_binary_trial_point: list,
        x_bs_binary_trial_point: list[list],
        soc_binary_trial_point: list,
    ) -> None:
        self.add_relaxation(
            x_binary_trial_point,
            y_binary_trial_point,
            x_bs_binary_trial_point,
            soc_binary_trial_point,
        )

    def add_cut_constraints(
        self,
        cut_intercepts: list,
        cut_gradients: list,
    ) -> None:
        state_variables = (
            self.x
            + self.y_bin_states_flattened
            + [var for gen_bs in self.x_bs for var in gen_bs]
            + self.soc_bin_states_flattened
        )

        id = 0
        for intercept, gradient in zip(
            cut_intercepts, cut_gradients, strict=False
        ):
            # Cut constraint
            self.model.addConstr(
                (
                    self.theta
                    >= intercept + gp.LinExpr(gradient, state_variables)
                ),
                f"cut_{id}",
            )
            id += 1

    def add_benders_cuts(
        self, cut_intercepts: list, cut_gradients: list, trial_points: list
    ) -> None:
        state_variables = (
            self.x
            + self.y_bin_states_flattened
            + [var for gen_bs in self.x_bs for var in gen_bs]
            + self.soc_bin_states_flattened
        )

        n_state_variables = len(state_variables)

        for intercept, gradient, trial_point in zip(
            cut_intercepts, cut_gradients, trial_points, strict=False
        ):
            if n_state_variables != len(trial_point):
                logger.warning("Trial point: %s", trial_point)
                msg = "Number of state variables must be equal to the number of trial points."
                raise ValueError(msg)

            self.model.addConstr(
                (
                    self.theta
                    >= intercept
                    + gp.quicksum(
                        gradient[i] * (state_variables[i] - trial_point[i])
                        for i in range(n_state_variables)
                    )
                ),
                "cut",
            )
    """
    add some no binary methods
    """

    def calculate_relaxed_terms(
            self,
            x_trial_point: list,
            y_trial_point: list,
            x_bs_trial_point: list[list],
            soc_trial_point: list,
    ):
        self.relaxed_terms = []
        self.relaxed_terms += [
            x_trial_point[j] - self.z_x[j]
            for j in range(len(x_trial_point))
        ]
        self.relaxed_terms += [
            y_trial_point[j] - self.z_y[j]
            for j in range(len(y_trial_point))
        ]
        self.relaxed_terms += [
            x_bs_trial_point[g][k] - self.z_x_bs[g][k]
            for g in range(len(x_bs_trial_point))
            for k in range(len(x_bs_trial_point[g]))
        ]
        self.relaxed_terms += [
            soc_trial_point[j] - self.z_soc[j]
            for j in range(len(soc_trial_point))
        ]

    def zero_relaxed_terms(self):
        self.sddip_copy_constrs = []
        for term in self.relaxed_terms:
            self.sddip_copy_constrs.append(
                self.model.addConstr(term == 0, "sddip-copy-without-binary")
            )
        self.model.update()


    def add_benders_cuts_without_binary(
        self, cut_intercepts: list, cut_gradients: list, trial_points: list
    ) -> None:
        state_variables = (
            self.x
            + self.y
            + [var for gen_bs in self.x_bs for var in gen_bs]
            + self.soc
        )

        n_state_variables = len(state_variables)

        for intercept, gradient, trial_point in zip(
            cut_intercepts, cut_gradients, trial_points, strict=False
        ):
            if n_state_variables != len(trial_point):
                logger.warning("Trial point: %s", trial_point)
                msg = "Number of state variables must be equal to the number of trial points."
                raise ValueError(msg)

            self.model.addConstr(
                (
                    self.theta
                    >= intercept
                    + gp.quicksum(
                        gradient[i] * (state_variables[i] - trial_point[i])
                        for i in range(n_state_variables)
                    )
                ),
                "cut",
            )

    def set_inner_objective(self, coefficients: list, pi_hat, pi0_hat):
        x_bs_p = [x for g in range(self.n_generators) for x in self.x_bs_p[g]]
        x_bs_n = [x for g in range(self.n_generators) for x in self.x_bs_n[g]]

        penalty = coefficients[-1]
        coefficients = (
                coefficients + [penalty] * (2 * self.n_storages + 2 * len(x_bs_p) + 1) + [1]
        )
        variables = (
                self.y  # 0 max
                + self.s_up  # 0 1
                + self.s_down  # 0 1
                + [self.ys_p, self.ys_n]  # 0
                + self.socs_p  # 0
                + self.socs_n  # 0
                + x_bs_p  # 0 1
                + x_bs_n  # 0 1
                + [self.delta]  # 0
                + [self.theta]  # -inf
        )  # theta对应cut
        theta = gp.LinExpr(coefficients, variables)

        z_X = (
                self.z_x
                + self.z_y
                + [val for bs in self.z_x_bs for val in bs]
                + self.z_soc
        )
        # for i in range(len(self.z_y)):
        #     self.model.addConstr(self.z_y[i] <= 100000)
        # for i in range(len(self.z_soc)):
        #     self.model.addConstr(self.z_soc[i] <= 100000)
        # for i in range(len(self.z_soc)):
        #     self.model.addConstr(self.z_soc[i] <= 100000)
        # for i in range(len(self.y)):
        #     self.model.addConstr(self.y[i] <= 100000)
        #
        # self.model.addConstr(self.delta<= 100000)
        # self.model.addConstr(self.theta<= 100000)
        # self.model.addConstr(self.ys_p<= 100000)
        # self.model.addConstr(self.ys_n<= 100000)
        #
        # for i in range(len(self.socs_p)):
        #     self.model.addConstr(self.socs_p[i] <= 100000)
        # for i in range(len(self.socs_n)):
        #     self.model.addConstr(self.socs_n[i] <= 100000)

        self.model.setObjective(gp.LinExpr(pi_hat, z_X) + pi0_hat * theta, gp.GRB.MINIMIZE)
        self.model.update()

    def add_z_var_constrains(self, max_soc, min_generation, max_generation):
        # TODO: 增加了z_soc约束，与soc的约束保持一致  需要用delta吗？
        self.model.addConstrs(
            (
                self.z_soc[s] <= max_soc[s]
                for s in range(self.n_storages)
            ), "max-soc_z",
        )

        # TODO: z_y约束，与y的约束保持一致  需要用delta吗？
        self.model.addConstrs(
            (
                self.z_y[g] >= min_generation[g] * self.z_x[g]
                for g in range(self.n_generators)
            ),
            "min-generation_z",
        )
        self.model.addConstrs(
            (
                self.z_y[g] <= max_generation[g] * self.z_x[g]
                for g in range(self.n_generators)
            ),
            "max-generation_z",
        )

        self.model.update()




    # 统一添加cut，不区分benders和lag
    def add_cut_constrains(
            self,
            cuts_list
    ):
        state_variables = (
                self.x
                + self.y
                + [var for gen_bs in self.x_bs for var in gen_bs]
                + self.soc
        )

        for id, cut in enumerate(cuts_list):
            pi = cut[:-1]
            intercept = cut[-1]
            self.model.addConstr(
                (
                        self.theta
                        >= intercept + gp.LinExpr(pi, state_variables)
                ),
                f"cut_{id}",
            )
        self.model.update()

