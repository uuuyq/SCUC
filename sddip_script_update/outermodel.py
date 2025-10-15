import gurobipy as gp

class OuterModel:
    def __init__(self, dim_pi, X_trial, theta_trial, benders_pi_list=None, alpha=1):
        self.dim_pi = dim_pi
        self.benders_pi_list = benders_pi_list
        self.subgradient = None
        self.L = None
        self.pi = []
        self.pi0 = None
        self.level_obj = None
        self.lower_bound_const = None
        self.abs_pi = []
        self.beta = []
        self.CL_constrain = []
        self.X_trial = X_trial
        self.theta_trial = theta_trial
        self.alpha = alpha
        self.model = gp.Model("outer_model")
        self.model.setParam("OutputFlag", 0)

        self.init_model(self.alpha)

    def init_model(self, alpha):
        for i in range(self.dim_pi):
            self.pi.append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="pi_%i" % (i + 1)
                )
            )
        self.pi0 = self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name="pi0")

        self.add_l1_norm_constrains(alpha)
        # if self.benders_pi_list is not None:
        #     self.add_CL_constrains()

        self.L = self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="L")
        self.model.update()

    def set_objective(self):
        self.model.setObjective(self.L - gp.quicksum(self.X_trial[i] * self.pi[i] for i in range(len(self.pi))) - self.pi0 * self.theta_trial,
            # - self.penalty / 2 * gp.quicksum((self.pi[i] - pi_star[i]) ** 2 for i in range(len(self.pi)))
            # - self.penalty / 2 * (self.pi0 - pi0_star) ** 2,
            gp.GRB.MAXIMIZE)
        self.model.update()


    def add_constrains(self, coefficient: list):
        self.model.addConstr(self.L <= gp.LinExpr(coefficient[:-1], self.pi) + self.pi0 * coefficient[-1])
        self.model.update()

    def set_lower_bound(self, lower_bound:float):
        self.lower_bound_const = self.model.addConstr(
            self.L - gp.LinExpr(self.X_trial, self.pi) - self.pi0 * self.theta_trial >= lower_bound,
            name="lower_bound"
        )
        self.model.update()

    def set_level_obj(
        self,
        pi_hat,
        pi0_hat
    ):
        self.level_obj = (gp.quicksum([(self.pi[i] - pi_hat[i]) ** 2 for i in range(len(self.pi))])
                          + (self.pi0 - pi0_hat) ** 2)
        self.model.setObjective(self.level_obj, gp.GRB.MINIMIZE)
        self.model.update()

    def recover(self):
        try:
            self.model.remove(self.lower_bound_const)
        except Exception as e:
            print("recover:", e)
        self.set_objective()

    # 无穷范数约束
    def add_l8_norm_constrains(self):
        if self.abs_pi:  # 检查是否已经初始化
            print("L1 norm constraints already added. Skipping.")
            return
        for i in range(self.dim_pi):
            self.abs_pi.append(
                self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name="abs_pi_%i" % (i + 1))
            )
        for i in range(self.dim_pi):
            self.model.addConstr(self.abs_pi[i] == gp.abs_(self.pi[i]))
        self.model.addConstr(self.pi0 <= 1, "pi0_cons")
        self.model.update()

    def add_l1_norm_constrains(self, alpha):
        for i in range(self.dim_pi):
            self.abs_pi.append(
                self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name="abs_pi_%i" % (i + 1))
            )
        for i in range(self.dim_pi):
            self.model.addConstr(self.abs_pi[i] == gp.abs_(self.pi[i]))
        self.model.addConstr(gp.quicksum(self.abs_pi[i] for i in range(self.dim_pi)) + alpha *self.pi0 <= 1)
        self.model.update()

    def add_CL_constrains(self):
        k = len(self.benders_pi_list)
        for i in range(k):
            self.beta.append(
                self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="beta_%i" % (i + 1))
            )
        # 添加benders pi与lag pi之间的约束，即CL
        for i in range(self.dim_pi):
            self.CL_constrain.append(
                self.model.addConstr(self.pi[i] == gp.quicksum(self.beta[j] * self.benders_pi_list[j][i] for j in range(k)))
            )
        self.model.update()


