class Config:

    def __init__(self,
                 num_data,
                 num_stage,
                 n_vars,
                 feat_dim,
                 single_cut_dim,
                 n_pieces,
                 train_data_path,
                 tensor_data_path,
                 result_path,
                 N_EPOCHS,
                 batch_size,
                 weight_decay,
                 LEARNING_RATE,
                 hidden_arr,
                 gamma,
                 n_realizations,
                 scenario_flag=False,
                 x_input_flag=True,
                 additional_data=None,
                 standard_flag=False
                 ):
        """
        :param num_data: 数据集样本数
        :param num_stage:
        :param n_vars: 只是变量的个数，在NN模型中会加上截距
        :param feat_dim:
        :param single_cut_dim:
        :param n_pieces:  cuts的个数
        :param train_data_path: train_data的位置
        :param tensor_data_path: tensor_data的位置  训练数据
        :param result_path: model训练等数据保存位置
        :param N_EPOCHS:
        :param weight_decay:
        :param LEARNING_RATE:
        :param hidden_arr:
        :param gamma:
        :param n_realizations: SDDiP算法中使用的采样个数
        :param scenario_flag: 是否使用场景作为模型的输入（默认使用parameters分布参数作为输入）
        :param x_input_flag:
        """
        self.num_data = num_data
        self.num_stage = num_stage
        self.n_pieces = n_pieces
        self.hidden_arr = hidden_arr
        self.LEARNING_RATE = LEARNING_RATE
        self.n_vars = n_vars
        self.feat_dim = feat_dim
        self.single_cut_dim = single_cut_dim
        self.N_EPOCHS = N_EPOCHS
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.scenario_flag = scenario_flag
        self.x_input_flag = x_input_flag
        self.gamma = gamma
        self.n_realizations = n_realizations
        self.scaler_part1 = None
        self.scaler_part2 = None
        self.train_data_path = train_data_path
        self.tensor_data_path = tensor_data_path
        self.result_path = result_path
        self.additional_data = additional_data
        self.standard_flag = standard_flag
        self.compare_path = None
