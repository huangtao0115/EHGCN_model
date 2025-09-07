"""Hyperboloid manifold."""

# 导入torch库，用于张量操作和自动微分
import torch

# 从manifolds.base模块导入Manifold基类
from manifolds.base import Manifold
# 从utils.math_utils模块导入数学工具函数
from utils.math_utils import arcosh, cosh, sinh


class Hyperboloid(Manifold):
    """
    Hyperboloid manifold class.

    We use the following convention: -x0^2 + x1^2 + ... + xd^2 = -K

    c = 1 / K is the hyperbolic curvature.
    """
    # 类初始化方法
    def __init__(self):
        # 调用父类Manifold的初始化方法
        super(Hyperboloid, self).__init__()
        # 设置manifold的名称
        self.name = 'Hyperboloid'
        # 设置不同精度的epsilon值，用于数值稳定性
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        # 设置最小和最大范数值，用于数值稳定性
        self.min_norm = 1e-15
        self.max_norm = 1e6

    # 计算闵可夫斯基点积
    def minkowski_dot(self, x, y, keepdim=True):
        # 计算x和y的点积，然后减去两倍的x和y的第一个元素的乘积
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        # 如果keepdim为True，则在结果的最后一个维度增加一个维度
        if keepdim:
            res = res.view(res.shape + (1,))
        return res

    # 计算闵可夫斯基范数
    def minkowski_norm(self, u, keepdim=True):
        # 计算u与其自身的闵可夫斯基点积
        dot = self.minkowski_dot(u, u, keepdim=keepdim)
        # 计算点积的平方根，并确保其值大于等于eps
        return torch.sqrt(torch.clamp(dot, min=self.eps[u.dtype]))

    # 计算两点之间的平方距离
    def sqdist(self, x, y, c):
        # 计算K值
        K = 1. / c
        # 计算x和y的闵可夫斯基点积
        prod = self.minkowski_dot(x, y)
        # 计算theta值，并确保其大于等于1+eps
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[x.dtype])
        # 计算平方距离
        sqdist = K * arcosh(theta) ** 2
        # 限制距离值以避免在Fermi-Dirac解码器中出现NaN
        return torch.clamp(sqdist, max=50.0)

    # 投影到双曲流形
    def proj(self, x, c):
        K = 1. / c
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2
        mask = torch.ones_like(x)
        mask[:, 0] = 0
        vals = torch.zeros_like(x)
        vals[:, 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[x.dtype]))
        return vals + mask * x

    def proj3(self, x, c):
        # 计算K值
        K = 1. / c
        # 计算维度d
        d = x.size(-1) - 1  # 这里的d应该是127，因为我们要处理的是最后一个维度除了第一个元素外的所有元素
        # 提取x的后d个元素
        y = x.narrow(-1, 1, d)
        # 计算y的欧几里得范数的平方
        y_sqnorm = torch.norm(y, p=2, dim=-1, keepdim=True) ** 2  # 计算每个样本的范数平方

        # 添加保护：确保非负且不小于eps
        K_plus_y = K + y_sqnorm
        K_plus_y = torch.clamp(K_plus_y, min=self.eps[x.dtype], max=1e20)

        # 创建一个全1的掩码
        mask = torch.ones_like(x)
        # 将掩码的第一列设置为0
        mask[..., 0] = 0  # 注意这里的索引方式要适应三维张量
        # 创建一个全0的张量
        vals = torch.zeros_like(x)
        # 设置vals的第一列
        # 添加保护：使用安全平方根
        sqrt_val = torch.sqrt(K_plus_y)
        vals[..., 0:1] = sqrt_val
        # 将x与vals相加，并应用掩码
        return vals + mask * x

    # 切空间投影
    def proj_tan(self, u, x, c):
        # 计算K值
        K = 1. / c
        # 计算维度d
        d = x.size(1) - 1
        # 计算x和u的后d个元素的点积
        ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=1, keepdim=True)
        # 创建一个全1的掩码
        mask = torch.ones_like(u)
        # 将掩码的第一列设置为0
        mask[:, 0] = 0
        # 创建一个全0的张量
        vals = torch.zeros_like(u)
        # 设置vals的第一列
        vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=self.eps[x.dtype])
        # 将u与vals相加，并应用掩码
        return vals + mask * u

    # 切空间原点投影
    def proj_tan0(self, u, c):
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals

    def proj_tan3(self, u, c):
        # 提取u的第一个元素
        narrowed = u.narrow(-1, 0, 1)
        # 创建一个全0的张量，类似U
        vals = torch.zeros_like(u)
        # 设置vals的第一列
        vals[..., 0:1] = narrowed
        # 返回u减去vals
        return u - vals
    # 指数映射
    def expmap(self, u, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, max=self.max_norm)
        theta = normu / sqrtK
        theta = torch.clamp(theta, min=self.min_norm)
        result = cosh(theta) * x + sinh(theta) * u / theta
        return self.proj(result, c)

    # 对数映射，参考点是x
    def logmap(self, x, y, c):
        K = 1. / c
        xy = torch.clamp(self.minkowski_dot(x, y) + K, max=-self.eps[x.dtype]) - K
        u = y + xy * x * c
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, min=self.min_norm)
        dist = self.sqdist(x, y, c) ** 0.5
        result = dist * u / normu
        return self.proj_tan(result, x, c)

    # 原点指数映射
    def expmap0(self, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d).view(-1, d)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm / sqrtK
        res = torch.ones_like(u)
        res[:, 0:1] = sqrtK * cosh(theta)
        res[:, 1:] = sqrtK * sinh(theta) * x / x_norm
        return self.proj(res, c)

    def expmap3(self, u, c):
        # 计算K值
        K = 1. / c
        # 计算K的平方根
        sqrtK = K ** 0.5
        # 计算维度d
        d = u.size(-1) - 1 # 注意这里d应该是128，因为我们要处理的是最后一个维度
        # 提取u的后d个元素，并调整形状
        x = u.narrow(-1, 1, d)  # 这里我们直接提取从第2个元素到最后一个元素
        # 计算x的欧几里得范数，并限制其最小值
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)  # 计算每个样本的范数
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        # 计算theta值
        theta = x_norm / sqrtK
        # 创建一个全1的张量
        res = torch.ones_like(u)
        # 设置res的第一列
        res[..., 0:1] = sqrtK * torch.cosh(theta)  # 使用torch.cosh
        # 设置res的后d个元素
        # 添加保护：防止除零
        safe_x_norm = torch.where(x_norm > self.min_norm, x_norm, torch.ones_like(x_norm))
        res[..., 1:] = sqrtK * torch.sinh(theta) * x / safe_x_norm
        return self.proj3(res, c)

    def logmap3(self, x, c):
        # 计算K值
        K = 1. / c
        # 计算K的平方根
        sqrtK = K ** 0.5
        # 计算维度d
        d = x.size(-1) - 1  # 注意这里d应该是128，因为我们要处理的是最后一个维度
        # 提取x的后d个元素，并调整形状
        y = x.narrow(-1, 1, d)  # 这里我们直接提取从第2个元素到最后一个元素
        # 计算y的欧几里得范数，并限制其最小值
        y_norm = torch.norm(y, p=2, dim=-1, keepdim=True)  # 计算每个样本的范数
        y_norm = torch.clamp(y_norm, min=self.min_norm)
        # 创建一个全0的张量
        res = torch.zeros_like(x)

        # 添加更严格的数值保护
        x0 = torch.clamp(x[..., 0:1], min=sqrtK + self.eps[x.dtype])  # 确保大于 sqrtK
        theta = x0 / sqrtK
        theta = torch.clamp(theta, min=1.0 + self.eps[x.dtype], max=1e10)

        # 设置res的后d个元素
        res[..., 1:] = sqrtK * torch.acosh(theta) * y / y_norm  # 使用acosh代替arcosh
        return res

    # 原点对数映射
    def logmap0(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d).view(-1, d)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=self.min_norm)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[:, 0:1] / sqrtK, min=1.0 + self.eps[x.dtype])
        res[:, 1:] = sqrtK * arcosh(theta) * y / y_norm
        return res

    # 莫比乌斯加法
    def mobius_add(self, x, y, c):
        u = self.logmap0(y, c)
        v = self.ptransp0(x, u, c)
        return self.expmap(v, x, c)

    # def mobius_add3(self, x, y, c):
    #     # 计算y的原点对数映射
    #     u = self.logmap0(y, c)
    #     # 计算x到u的平行传输
    #     # 得到的特征是切空间的特征
    #     v = self.ptransp0(x, u, c)
    #     # 计算v的指数映射
    #     return self.expmap(v, x, c)


    # 莫比乌斯加法，不是偏置相加，而是两个相同维度的向量进行相加
    def mobius_add0(self, x, y, c):
        # 计算y到以x为参考点的对数映射
        y = self.logmap0(y, c)
        x = self.logmap0(x, c)
        add = x + y
        # 计算v到以x为参考点的指数映射
        return self.expmap0(add, c)

    def mobius_add1(self, x, y, c):
        # x为参考点的对数映射
        y = self.logmap(x,y, c)
        # x = self.logmap0(x, c)
        add = x + y
        # 计算v到以x为参考点的指数映射
        return self.expmap(add,x, c)

    # 对特征进行聚合相加
    def mobius_featureadd(self, x, y, c):
        # 计算y的原点对数映射
        u = self.logmap0(y, c)
        # 计算x到u的平行传输
        # 得到的特征是切空间的特征
        v = self.ptransp0(x, u, c)
        # 计算v的指数映射
        return self.expmap(v, x, c)

    # 莫比乌斯矩阵向量乘法
    def mobius_matvec(self, m, x, c):
        # 计算x的原点对数映射
        u = self.logmap0(x, c)
        # 计算m与u的转置的矩阵乘法
        mu = u @ m.transpose(-1, -2)
        # 计算mu的原点指数映射
        return self.expmap0(mu, c)

    # 双曲的元素相乘
    def mobius_matvec1(self, m, x, c):
        # 计算x的原点对数映射
        u = self.logmap0(x, c)
        # 计算m与u的转置的矩阵乘法
        mu = torch.mm(u, m)
        # 计算mu的原点指数映射
        return self.expmap0(mu, c)

    def mobius_matvec2(self, m, x, c):
        # 计算x的原点对数映射
        u = self.logmap0(x, c)
        # 计算m与u的转置的矩阵乘法
        mu = torch.matmul(m, u)
        # mu = torch.mm(u, m)
        # 计算mu的原点指数映射
        return self.expmap0(mu, c)

    # 这是包含batch_size 的操作
    def mobius_matvec3(self, m, x, c):
        # 计算x的原点对数映射
        u = self.logmap3(x, c)
        # 计算m与u的转置的矩阵乘法
        mu = u * m
        # 计算mu的原点指数映射
        return self.expmap3(mu, c)

    def mobius_matvec3_3(self, m, x, c):
        # 计算x的原点对数映射
        u = self.logmap3(x, c)
        # 计算m与u的转置的矩阵乘法
        mu = u @ m
        # 计算mu的原点指数映射
        return self.expmap3(mu, c)

    # 逐元素乘法
    def mobius_matvec0(self, m, x, c):
        # 计算x的原点对数映射
        u = self.logmap0(x, c)
        # 计算m与u的转置的矩阵乘法
        mu = u * m
        # 计算mu的原点指数映射
        return self.expmap0(mu, c)

    # 平行传输
    def ptransp(self, x, y, u, c):
        # 计算x到y的对数映射
        logxy = self.logmap(x, y, c)
        # 计算y到x的对数映射
        logyx = self.logmap(y, x, c)
        # 计算x和y之间的平方距离，并限制其最小值
        sqdist = torch.clamp(self.sqdist(x, y, c), min=self.min_norm)
        # 计算alpha值
        alpha = self.minkowski_dot(logxy, u) / sqdist
        # 计算平行传输的结果
        res = u - alpha * (logxy + logyx)
        # 将结果投影到切空间
        return self.proj_tan(res, y, c)

    # 原点平行传输
    def ptransp0(self, x, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        x0 = x.narrow(-1, 0, 1)
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_norm = torch.clamp(torch.norm(y, p=2, dim=1, keepdim=True), min=self.min_norm)
        y_normalized = y / y_norm
        v = torch.ones_like(x)
        v[:, 0:1] = - y_norm
        v[:, 1:] = (sqrtK - x0) * y_normalized
        alpha = torch.sum(y_normalized * u[:, 1:], dim=1, keepdim=True) / sqrtK
        res = u - alpha * v
        return self.proj_tan(res, x, c)

    # 转换到泊松圆盘
    def to_poincare(self, x, c):
        # 计算K值
        K = 1. / c
        # 计算K的平方根
        sqrtK = K ** 0.5
        # 计算维度d
        d = x.size(-1) - 1
        # 返回转换后的泊松圆盘坐标
        return sqrtK * x.narrow(-1, 1, d) / (x[:, 0:1] + sqrtK)