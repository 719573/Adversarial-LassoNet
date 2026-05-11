from itertools import islice
import torch
from torch import nn
from torch.nn import functional as F
from lassonet.prox import inplace_group_prox, inplace_prox, prox
from itertools import islice

# class LassoNet(nn.Module):
#     def __init__(self, *dims, groups=None, dropout=None):
#         """
#         first dimension is input
#         last dimension is output
#         `groups` is a list of list such that `groups[i]`
#         contains the indices of the features in the i-th group
#
#         """
#         assert len(dims) > 2
#         if groups is not None:
#             n_inputs = dims[0]
#             all_indices = []
#             for g in groups:
#                 for i in g:
#                     all_indices.append(i)
#             assert len(all_indices) == n_inputs and set(all_indices) == set(
#                 range(n_inputs)
#             ), f"Groups must be a partition of range(n_inputs={n_inputs})"
#
#         self.groups = groups
#
#         super().__init__()
#
#         self.dropout = nn.Dropout(p=dropout) if dropout is not None else None
#         self.layers = nn.ModuleList(
#             [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
#         )
#         self.skip = nn.Linear(dims[0], dims[-1], bias=False)
#
#     def forward(self, inp):
#         current_layer = inp
#         result = self.skip(inp)
#         for theta in self.layers:
#             current_layer = theta(current_layer)
#             if theta is not self.layers[-1]:
#                 if self.dropout is not None:
#                     current_layer = self.dropout(current_layer)
#                 current_layer = F.relu(current_layer)
#         return result + current_layer
#
#     def prox(self, *, lambda_, lambda_bar=0, M=1):
#         if self.groups is None:
#             with torch.no_grad():
#                 inplace_prox(
#                     beta=self.skip,
#                     theta=self.layers[0],
#                     lambda_=lambda_,
#                     lambda_bar=lambda_bar,
#                     M=M,
#                 )
#         else:
#             with torch.no_grad():
#                 inplace_group_prox(
#                     groups=self.groups,
#                     beta=self.skip,
#                     theta=self.layers[0],
#                     lambda_=lambda_,
#                     lambda_bar=lambda_bar,
#                     M=M,
#                 )
#
#     def lambda_start(
#         self,
#         M=1,
#         lambda_bar=0,
#         factor=2,
#     ):
#         """Estimate when the model will start to sparsify."""
#
#         def is_sparse(lambda_):
#             with torch.no_grad():
#                 beta = self.skip.weight.data
#                 theta = self.layers[0].weight.data
#
#                 for _ in range(10000):
#                     new_beta, theta = prox(
#                         beta,
#                         theta,
#                         lambda_=lambda_,
#                         lambda_bar=lambda_bar,
#                         M=M,
#                     )
#                     if torch.abs(beta - new_beta).max() < 1e-5:
#                         break
#                     beta = new_beta
#                 return (torch.norm(beta, p=2, dim=0) == 0).sum()
#
#         start = 1e-6
#         while not is_sparse(factor * start):
#             start *= factor
#         return start
#
#     def l2_regularization(self):
#         """
#         L2 regulatization of the MLP without the first layer
#         which is bounded by the skip connection
#         """
#         ans = 0
#         for layer in islice(self.layers, 1, None):
#             ans += (
#                 torch.norm(
#                     layer.weight.data,
#                     p=2,
#                 )
#                 ** 2
#             )
#         return ans
#
#     def l1_regularization_skip(self):
#         return torch.norm(self.skip.weight.data, p=2, dim=0).sum()
#
#     def l2_regularization_skip(self):
#         return torch.norm(self.skip.weight.data, p=2)
#
#     def input_mask(self):
#         with torch.no_grad():
#             return torch.norm(self.skip.weight.data, p=2, dim=0) != 0
#
#     def selected_count(self):
#         return self.input_mask().sum().item()
#
#     def cpu_state_dict(self):
#         return {k: v.detach().clone().cpu() for k, v in self.state_dict().items()}




class LassoNet(nn.Module):
    def __init__(self, *dims, groups=None, dropout=None):
        """
        dims: 输入维度, 隐藏层1, 隐藏层2, ..., 输出维度
        """
        assert len(dims) > 2
        if groups is not None:
            n_inputs = dims[0]
            all_indices = []
            for g in groups:
                for i in g:
                    all_indices.append(i)
            assert len(all_indices) == n_inputs and set(all_indices) == set(
                range(n_inputs)
            ), f"Groups must be a partition of range(n_inputs={n_inputs})"

        self.groups = groups
        super().__init__()

        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None

        # 1. 创建全连接层 (Linear) - 保持原样
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        #  修改点 1：创建 Batch Normalization 层
        # 逻辑：有多少个隐藏层，就建多少个 BN 层。
        # 最后一层是输出层，不需要 BN。
        self.bn_layers = nn.ModuleList()
        for i in range(len(dims)-2):  # 减2是因为排除输入层和输出层
            # dims[i+1] 是第 i 个隐藏层的输出节点数
            self.bn_layers.append(nn.BatchNorm1d(dims[i + 1]))

        # 2. Skip 连接 - 保持原样
        self.skip = nn.Linear(dims[0], dims[-1], bias=False)

    def forward(self, inp):
        current_layer = inp
        result = self.skip(inp)

        # 遍历每一层
        for i, theta in enumerate(self.layers):
            current_layer = theta(current_layer)  # 跑全连接 Linear

            # # 如果不是最后一层（即：如果是隐藏层）
            # if theta is not self.layers[-1]:

            #  修改点 2：在 ReLU 之前插入 BN
            # 确保 i 没有越界（虽然后面 logic 保证了不会，但安全第一）
            if i < len(self.bn_layers):
                current_layer = self.bn_layers[i](current_layer)

            if self.dropout is not None:
                current_layer = self.dropout(current_layer)

            # 最后才跑 ReLU
            current_layer = F.relu(current_layer)

        return result + current_layer

    def prox(self, *, lambda_, lambda_bar=0, M=1):
        if self.groups is None:
            with torch.no_grad():
                inplace_prox(
                    beta=self.skip,
                    theta=self.layers[0],
                    lambda_=lambda_,
                    lambda_bar=lambda_bar,
                    M=M,
                )
        else:
            with torch.no_grad():
                inplace_group_prox(
                    groups=self.groups,
                    beta=self.skip,
                    theta=self.layers[0],
                    lambda_=lambda_,
                    lambda_bar=lambda_bar,
                    M=M,
                )

    def lambda_start(self, M=1, lambda_bar=0, factor=2):
        def is_sparse(lambda_):
            with torch.no_grad():
                beta = self.skip.weight.data
                theta = self.layers[0].weight.data
                for _ in range(10000):
                    new_beta, theta = prox(
                        beta, theta, lambda_=lambda_, lambda_bar=lambda_bar, M=M
                    )
                    if torch.abs(beta - new_beta).max() < 1e-5:
                        break
                    beta = new_beta
                return (torch.norm(beta, p=2, dim=0) == 0).sum()

        start = 1e-6
        while not is_sparse(factor * start):
            start *= factor
        return start

    def l2_regularization(self):
        ans = 0
        for layer in islice(self.layers, 1, None):
            ans += (torch.norm(layer.weight.data, p=2) ** 2)
        return ans

    def l1_regularization_skip(self):
        return torch.norm(self.skip.weight.data, p=2, dim=0).sum()

    def l2_regularization_skip(self):
        return torch.norm(self.skip.weight.data, p=2)

    def input_mask(self):
        with torch.no_grad():
            return torch.norm(self.skip.weight.data, p=2, dim=0) != 0

    def selected_count(self):
        return self.input_mask().sum().item()

    def cpu_state_dict(self):
        return {k: v.detach().clone().cpu() for k, v in self.state_dict().items()}



# class LassoNet(nn.Module):
#     def __init__(self, *dims, groups=None, dropout=None):
#         """
#         Method 2 定制版 LassoNet
#         结构：Linear编码 -> Conv1d解码 -> 输出
#         注意：dims[1] (第一个隐藏层维度) 必须能被 4 整除，否则卷积层计算会报错。
#         """
#         assert len(dims) > 2, "必须要指定输入维度、至少一个隐藏维度(作为信号宽度)和输出维度"
# 
#         if groups is not None:
#             n_inputs = dims[0]
#             all_indices = []
#             for g in groups:
#                 for i in g:
#                     all_indices.append(i)
#             assert len(all_indices) == n_inputs and set(all_indices) == set(
#                 range(n_inputs)
#             ), f"Groups must be a partition of range(n_inputs={n_inputs})"
# 
#         self.groups = groups
#         super().__init__()
# 
#         self.dropout = nn.Dropout(p=dropout) if dropout is not None else None
# 
#         # =================================================================
#         #  Method 2 核心结构定义
#         # =================================================================
# 
#         # 1. Encoder (编码器): 全连接扩维
#         # 输入 -> 隐藏层宽度 (例如 2560)
#         # 这是 LassoNet 的第一层，必须与 skip 层竞争
#         self.encoder = nn.Linear(dims[0], dims[1])
#         self.bn_enc = nn.BatchNorm1d(dims[1])
# 
#         # 2. Decoder (解码器): 卷积降维 (稀疏连接)
#         # 假设 dims[1] 是信号长度，通道数为 1
#         # Conv1: 长度减半 (2560 -> 1280), 通道 1 -> 4
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5, stride=2, padding=2)
#         self.bn1 = nn.BatchNorm1d(4)
# 
#         # Conv2: 长度再减半 (1280 -> 640), 通道 4 -> 8
#         self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=5, stride=2, padding=2)
#         self.bn2 = nn.BatchNorm1d(8)
# 
#         # 3. Classifier (分类头)
#         # 计算展平后的维度: 
#         # 长度从 dims[1] 经过两次 stride=2 减半: L = dims[1] / 4
#         # 通道数变成 8
#         # 所以 Flatten Dim = 8 * (dims[1] // 4)
#         flatten_dim = 8 * (dims[1] // 4)
#         self.classifier = nn.Linear(flatten_dim, dims[-1])
# 
#         #  关键：将所有层放入 self.layers
#         # 这样 LassoNet 的优化器和正则化函数才能找到它们！
#         # 顺序必须是：[第一层(必须是Linear), ..., 最后一层]
#         self.layers = nn.ModuleList([
#             self.encoder,
#             self.conv1,
#             self.conv2,
#             self.classifier
#         ])
# 
#         # 4. Skip 连接 (LassoNet 的灵魂)
#         self.skip = nn.Linear(dims[0], dims[-1], bias=False)
# 
#     def forward(self, inp):
#         # 1. Skip Path (线性路径)
#         result = self.skip(inp)
# 
#         # 2. Non-linear Path (卷积编码-解码路径)
# 
#         # --- Encoder ---
#         out = self.encoder(inp)
#         out = self.bn_enc(out)
#         if self.dropout is not None: out = self.dropout(out)
#         out = F.relu(out)
# 
#         # --- Reshape for Conv (N, L) -> (N, 1, L) ---
#         out = out.unsqueeze(1)
# 
#         # --- Decoder Conv1 ---
#         out = self.conv1(out)
#         out = self.bn1(out)
#         if self.dropout is not None: out = self.dropout(out)
#         out = F.relu(out)
# 
#         # --- Decoder Conv2 ---
#         out = self.conv2(out)
#         out = self.bn2(out)
#         if self.dropout is not None: out = self.dropout(out)
#         out = F.relu(out)
# 
#         # --- Flatten ---
#         out = out.view(out.size(0), -1)
# 
#         # --- Classifier ---
#         out = self.classifier(out)
# 
#         return result + out
# 
#     # =================================================================
#     # 下面的辅助函数保持原样，确保接口兼容性
#     # =================================================================
# 
#     def prox(self, *, lambda_, lambda_bar=0, M=1):
#         if self.groups is None:
#             with torch.no_grad():
#                 inplace_prox(
#                     beta=self.skip,
#                     theta=self.layers[0],  # 这里指向 self.encoder
#                     lambda_=lambda_,
#                     lambda_bar=lambda_bar,
#                     M=M,
#                 )
#         else:
#             with torch.no_grad():
#                 inplace_group_prox(
#                     groups=self.groups,
#                     beta=self.skip,
#                     theta=self.layers[0],
#                     lambda_=lambda_,
#                     lambda_bar=lambda_bar,
#                     M=M,
#                 )
# 
#     def lambda_start(self, M=1, lambda_bar=0, factor=2):
#         def is_sparse(lambda_):
#             with torch.no_grad():
#                 beta = self.skip.weight.data
#                 theta = self.layers[0].weight.data
#                 for _ in range(10000):
#                     new_beta, theta = prox(
#                         beta,
#                         theta,
#                         lambda_=lambda_,
#                         lambda_bar=lambda_bar,
#                         M=M,
#                     )
#                     if torch.abs(beta - new_beta).max() < 1e-5:
#                         break
#                     beta = new_beta
#                 return (torch.norm(beta, p=2, dim=0) == 0).sum()
# 
#         start = 1e-6
#         while not is_sparse(factor * start):
#             start *= factor
#         return start
# 
#     def l2_regularization(self):
#         # 计算 L2 正则时，跳过第一层 (encoder)，计算后面所有层
#         # 对于 Conv 层和 Linear 层，weight 属性都是存在的，所以通用
#         ans = 0
#         for layer in islice(self.layers, 1, None):
#             ans += (torch.norm(layer.weight.data, p=2) ** 2)
#         return ans
# 
#     def l1_regularization_skip(self):
#         return torch.norm(self.skip.weight.data, p=2, dim=0).sum()
# 
#     def l2_regularization_skip(self):
#         return torch.norm(self.skip.weight.data, p=2)
# 
#     def input_mask(self):
#         with torch.no_grad():
#             return torch.norm(self.skip.weight.data, p=2, dim=0) != 0
# 
#     def selected_count(self):
#         return self.input_mask().sum().item()
# 
#     def cpu_state_dict(self):
#         return {k: v.detach().clone().cpu() for k, v in self.state_dict().items()}