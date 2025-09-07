import plotly.graph_objects as go
import numpy as np

# 定义u和v的范围
u = np.linspace(-5, 5, 100)
v = np.linspace(-5, 5, 100)
u, v = np.meshgrid(u, v)

# 计算双曲体的坐标
x = u
y = v
z = np.sqrt(1 + 2*x**2 + 2*y**2)

# Viridis：一种从黄绿色到深蓝色的渐变色，适合于大多数数据集。
# Plasma：一种从黄色到深紫色的渐变色，强调高对比度。
# Inferno：一种从黄色到深紫色的渐变色，强调颜色的区分度。
# Magma：一种从黄色到深红色的渐变色，适合于强调暖色调。
# Cividis：一种从浅蓝色到深灰色的渐变色，适合于强调亮度变化。
# Turbo：一种从绿色到红色的渐变色，适合于强调高对比度和亮度变化。
# Blues：一种从浅蓝色到深蓝色的渐变色，适合于强调冷色调。
# Greens：一种从浅绿色到深绿色的渐变色，适合于强调自然色调。
# Oranges：一种从浅橙色到深橙色的渐变色，适合于强调暖色调。
# Reds：一种从浅红色到深红色的渐变色，适合于强调警告或重要性。
# Greys：一种从浅灰色到深灰色的渐变色，适合于强调亮度变化。
# colorrs = 'Viridis'      # 黄绿-深蓝渐变（默认推荐）
# colorrs = 'Plasma'       # 黄-紫渐变（高对比度）
# colorrs = 'Inferno'      # 黄-黑渐变（强对比）
# colorrs = 'Magma'        # 黄-红渐变（暖色系）
# colorrs = 'Cividis'      # 蓝-灰渐变（亮度突出）
# colorrs = 'Turbo'        # 彩虹色环（最大对比）
# colorrs = 'Blues'        # 浅蓝到深蓝（冷静）
# colorrs = 'Greens'       # 浅绿到深绿（自然）
# colorrs = 'Reds'         # 浅红到深红（警示）
# colorrs = 'Oranges'      # 浅橙到深橙（温暖）
# colorrs = 'Purples'      # 浅紫到深紫（神秘）
colorrs = 'RdBu'         # 红-蓝对比（冷热）可行
# colorrs = 'PiYG'         # 粉-绿对比（生物）
# colorrs = 'BrBG'         # 棕-绿对比（地形）
# colorrs = [[0, 'rgb(255,0,0)'], [1, 'rgb(0,0,255)']]  # 红到蓝渐变
#
#
# colorrs='Blues'

# 创建3D图形
fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, surfacecolor=z, colorscale=colorrs)])

# 定义圆盘的半径和中心
radius = 5
center_x, center_y, center_z = 0, 0, 0  # 将圆盘移至双曲体下方

# 生成圆盘的坐标
theta = np.linspace(0, 2 * np.pi, 100)
r = np.linspace(0, radius, 100)
theta, r = np.meshgrid(theta, r)
x_disk = center_x + r * np.cos(theta)
y_disk = center_y + r * np.sin(theta)
z_disk = center_z * np.ones_like(x_disk)  # 圆盘位于z=0平面

# 计算每个点到圆盘中心的距离
distance_from_center = np.sqrt(x_disk**2 + y_disk**2)
# 归一化距离值到[0, 1]区间
normalized_distance = distance_from_center / radius

# 添加圆盘到图形
fig.add_trace(go.Surface(x=x_disk, y=y_disk, z=z_disk, surfacecolor=normalized_distance, colorscale=colorrs))
#
# # 设置布局参数，去除背景、坐标轴数字和颜色条
fig.update_layout(
    scene=dict(
        xaxis=dict(
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title=''  # 删除x轴标签
        ),
        yaxis=dict(
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title=''  # 删除y轴标签
        ),
        zaxis=dict(
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title=''  # 删除z轴标签
        ),
        bgcolor="rgba(0,0,0,0)"  # 设置背景颜色为透明
    ),
    paper_bgcolor="rgba(0,0,0,0)",  # 设置整个图形背景为透明
)

# 显示图形
fig.show()