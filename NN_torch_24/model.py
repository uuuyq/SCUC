import torch
import torch.nn as nn
import numpy as np

class NN_Model(nn.Module):
    def __init__(self, num_stages, hidden_arr, num_vars, num_pieces):
        super().__init__()
        self.num_stages = num_stages
        self.num_vars = num_vars
        self.num_pieces = num_pieces

        self.initial_dense = nn.LazyLinear(hidden_arr[0])

        self.stage_embedding = nn.Embedding(
            num_embeddings=num_stages - 1,  # 因为索引从0开始
            embedding_dim=hidden_arr[0]
        )

        mlp_layers = []
        for in_dim, out_dim in zip(hidden_arr, hidden_arr[1:]):  # hidden_arr(512, 512)(input1, output1(input2), output2)
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(hidden_arr[-1], (num_vars + 1) * num_pieces))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, feat, x, scenarios=None, return_output=False, x_input_flag=False):
        """
        feats = torch.reshape(feats, [-1, 4])  [batch, 4]
        scenarios = torch.reshape(scenarios, [-1, self.num_scenarios, 12])  [batch, 6, 12]
        cuts = torch.reshape(cuts, [-1, self.n_pieces, 14])  [batch, n_pieces, 14]
        x = torch.reshape(x, [-1, self.n_pieces, 13])  [batch, n_pieces, 13]
        """

        # 分离stage索引和其他特征
        batch_size = feat.size(0)
        stage_idx = feat[:, 1].long() - 2  # 数据中的stage从2开始
        if scenarios is None:
            cond_feat = feat[:, 2:]
        else:
            cond_feat = scenarios.reshape(scenarios.size(0), -1)
            # print("scenario", cond_feat.shape)
        if x_input_flag:
            # print(cond_feat.shape)  # torch.Size([80, 72])
            # print(x.shape)  # torch.Size([80, 15, 13])
            cond_feat = torch.cat((cond_feat, x.reshape(batch_size, -1)), dim=1)
        cond_feat = self.initial_dense(cond_feat)
        time_embed = self.stage_embedding(stage_idx)
        cond_feat += time_embed

        output = self.mlp(cond_feat)

        output = output.reshape(batch_size, self.num_pieces, self.num_vars + 1)
        # 还要预测出顺序
        # x.shape  [batch, self.num_pieces, self.num_vars]
        # result_storage_temp.shape  []
        result = (x * output[:, :, :x.size(-1)]).sum(dim=-1, keepdims=True) + output[:, :, -1:]

        if return_output:
            return result, output
        else:
            return result

class ApproxEMDLoss(nn.Module):
    def __init__(self, temperature_scales=None):
        """
        EMD 近似损失函数
        参数：
            temperature_scales: 多尺度温度参数列表，默认使用原论文配置
        """
        super().__init__()
        # self.temperature_scales = temperature_scales or [-4.0 ** j for j in range(8, -3, -1)]
        self.temperature_scales = temperature_scales or np.arange(8, -3, -0.25).tolist()
        self.temperature_scales[-1] = 0.0  # 最后一轮温度归零

    def _pdist2(self, x, y):
        """计算平方欧氏距离矩阵"""
        x_norm = torch.sum(x ** 2, dim=2, keepdim=True)  # [B, N, 1]
        y_norm = torch.sum(y ** 2, dim=2, keepdim=True)  # [B, M, 1]
        cross = torch.bmm(x, y.transpose(1, 2))  # [B, N, M]
        return x_norm - 2 * cross + y_norm.transpose(1, 2)

    def _compute_match(self, points_x, points_y):
        """核心匹配计算"""
        batch_size, n, _ = points_x.shape
        _, m, _ = points_y.shape

        # 初始化容量约束
        max_size = max(n, m)
        factorl = max_size / n
        factorr = max_size / m

        pairwise_dist2 = self._pdist2(points_x, points_y)  # [B, N, M]

        # 批量处理初始化
        saturatedl = torch.full((batch_size, n), factorl,
                                dtype=points_x.dtype,
                                device=points_x.device)
        saturatedr = torch.full((batch_size, m), factorr,
                                dtype=points_x.dtype,
                                device=points_x.device)

        match = torch.zeros_like(pairwise_dist2)

        for level in self.temperature_scales:
            # 计算对数权重
            log_sr = torch.log(saturatedr.unsqueeze(1) + 1e-30)  # [B, 1, M]
            log_weight = pairwise_dist2 * level + log_sr

            # Softmax归一化
            weight = torch.softmax(log_weight, dim=-1)  # [B, N, M]

            # 应用发送端约束
            weight = weight * saturatedl.unsqueeze(-1)

            # 接收端容量调整
            ss = torch.sum(weight, dim=1, keepdim=True) + 1e-9  # [B, 1, M]
            ss = torch.minimum(saturatedr.unsqueeze(1) / ss,
                               torch.tensor(1.0, device=ss.device))
            weight = weight * ss

            # 更新剩余容量
            s = torch.sum(weight, dim=2)  # [B, N]
            ss2 = torch.sum(weight, dim=1)  # [B, M]
            saturatedl = torch.clamp_min(saturatedl - s, 0.0)
            saturatedr = torch.clamp_min(saturatedr - ss2, 0.0)

            match = match + weight

        return match.detach()  # 阻断匹配矩阵的梯度

    def forward(self, points_x, points_y):
        """
        前向计算
        参数：
            points_x: 形状 [B, N, D] 的预测点云
            points_y: 形状 [B, M, D] 的目标点云
        返回：
            loss: 标量损失值
        """
        # 计算匹配矩阵
        match_matrix = self._compute_match(points_x, points_y)

        # 计算距离矩阵
        dist_matrix = self._pdist2(points_x, points_y)

        # 计算EMD损失
        loss = torch.sum(match_matrix * dist_matrix, dim=(1, 2)).mean()

        return loss

# 使用示例
if __name__ == "__main__":
    # 参数配置
    num_stages = 5
    hidden_dims = [512, 512]
    num_vars = 3
    num_pieces = 4

    # 初始化模型
    NN = NN_Model(
        num_stages=num_stages,
        hidden_arr=hidden_dims,
        num_vars=num_vars,
        num_pieces=num_pieces
    )
    input = torch.ones((10,10))
    output = NN(input)
    print(output.shape)

