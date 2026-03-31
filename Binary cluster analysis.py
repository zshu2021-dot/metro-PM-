import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.stats import chi2

# 设置全局字体为 Arial
plt.rcParams['font.family'] = 'Arial'

# ============================================
# 1. 读取数据
# ============================================
df = pd.read_excel(r"C:\\Users\\shuzh\\Desktop\\散点图聚类分析\outdoor.xlsx")
df = df.dropna(subset=['Davg', 'Density'])

X = df['Davg'].values       # 原始粒径值（μm）
Y = df['Density'].values    # 原始密度值（g/cm³）

print(f"Davg    范围：{X.min():.4f} ~ {X.max():.4f} μm")
print(f"Density 范围：{Y.min():.4f} ~ {Y.max():.4f} g/cm³")

# ============================================
# 2. ▼▼▼ 手动设置分界直线参数 ▼▼▼
# 在对数坐标轴上显示为直线
# 方程：Density = k * log10(Davg) + b
# ============================================
k = 0.5    # 斜率（负值=右下倾斜，正值=右上倾斜）← 手动调整
b =  3.2   # 截距（控制直线上下位置）          ← 手动调整

# 分类依据：用 log10(X) 计算，保证与对数轴一致
log_X = np.log10(X)
mask0 = Y >= k * log_X + b    # 直线上方 → Cluster 1
mask1 = Y <  k * log_X + b    # 直线下方 → Cluster 2

df['Cluster'] = np.where(mask0, 0, 1)
print(f"\n分界线方程：Density = {k} × log₁₀(Davg) + {b}")
print(f"Cluster 1（上方）数量：{mask0.sum()}")
print(f"Cluster 2（下方）数量：{mask1.sum()}")

# ============================================
# 3. 置信椭圆函数（仅边框，无填充）
# ============================================
def draw_confidence_ellipse(ax, x, y, coverage=0.75, color='blue', linewidth=2.5, 
                           fill_alpha=0.75, edge_alpha=0.8):
    """
    在对数坐标轴上绘制置信椭圆（半透明填充+边框）
    
    参数:
        x, y       : 原始数据（Davg, Density）
        coverage   : 椭圆覆盖的数据点比例（0.75 = 75%）← 手动调整
        color      : 椭圆颜色 ← 手动调整
        linewidth  : 椭圆边框线宽 ← 手动调整
        fill_alpha : 椭圆填充透明度（0.1-0.3）← 手动调整
        edge_alpha : 椭圆边框透明度（0-1）← 手动调整
    """
    # 转换到对数空间
    log_x = np.log10(x)
    
    # 计算协方差矩阵（在对数空间）
    data = np.vstack([log_x, y])
    cov = np.cov(data)
    
    # 计算椭圆参数
    eigenvalues, eigenvectors = linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # 根据覆盖率计算缩放因子
    n_std = np.sqrt(chi2.ppf(coverage, df=2))
    
    # 椭圆中心（对数空间）
    center_log_x = np.mean(log_x)
    center_y = np.mean(y)
    
    # 生成椭圆边界点（在对数空间）
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])
    
    # 参数化椭圆
    t = np.linspace(0, 2 * np.pi, 300)
    ellipse_x_r = (width / 2) * np.cos(t)
    ellipse_y_r = (height / 2) * np.sin(t)
    
    # 旋转
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    ellipse_coords = np.dot(R, np.array([ellipse_x_r, ellipse_y_r]))
    
    # 平移到中心（对数空间）
    ellipse_log_x = ellipse_coords[0, :] + center_log_x
    ellipse_y = ellipse_coords[1, :] + center_y
    
    # 转回原始空间（只对x坐标）
    ellipse_x = 10 ** ellipse_log_x
    
    # 绘制半透明填充
    ax.fill(ellipse_x, ellipse_y,
            color=color,
            alpha=fill_alpha,   # 填充透明度 ← 手动调整
            zorder=2)
    
    # 绘制虚线边框
    ax.plot(ellipse_x, ellipse_y,
            color=color, 
            linewidth=linewidth,
            linestyle='--',
            alpha=edge_alpha,   # 边框透明度 ← 手动调整
            zorder=5)
    
    # 返回中心坐标
    return 10 ** center_log_x, center_y

# ============================================
# 4. 绘图
# ============================================
fig, ax = plt.subplots(figsize=(9, 9))  # 正方形画布，保证1:1比例

# ▼▼▼ 颜色配置 ▼▼▼
ellipse_colors = ["#E35717", "#095271"]  # Cluster 1 / Cluster 2 椭圆颜色 ← 手动调整
cluster_names  = ['Cluster 1', 'Cluster 2']
masks          = [mask0, mask1]

# ▼▼▼ 散点样式配置 ▼▼▼
scatter_facecolor = "#8d8c8f"   # 散点填充颜色（更浅）← 手动调整
scatter_edgecolor = 'black'        # 散点边缘颜色（更浅）← 手动调整
scatter_alpha     = 0.15           # 散点透明度（更透明）← 手动调整
scatter_size      = 50           # 散点大小 ← 手动调整
scatter_linewidth = 0.1           # 散点边缘线宽 ← 手动调整

# ▼▼▼ 椭圆配置 ▼▼▼
ellipse_coverage  = 0.95          # 椭圆覆盖率（0.75 = 75%）← 手动调整
ellipse_linewidth = 2.0           # 椭圆边框线宽 ← 手动调整

for i, (mask, ellipse_color, name) in enumerate(zip(masks, ellipse_colors, cluster_names)):
    xi = X[mask]
    yi = Y[mask]
    n  = mask.sum()

    if n == 0:
        print(f"⚠️  {name} 为空，请调整 k 和 b")
        continue

    # ① 散点（浅灰色填充 + 灰色边缘）
    ax.scatter(
        xi, yi,
        facecolors=scatter_facecolor,
        edgecolors=scatter_edgecolor,
        alpha=scatter_alpha,
        s=scatter_size,
        linewidths=scatter_linewidth,
        label=f'{name}  (n={n})',
        zorder=1
    )

    # ② 椭圆（半透明填充+虚线边框）
    if n >= 3:
        cx, cy = draw_confidence_ellipse(
            ax, xi, yi,
            coverage=0.75,
            color=ellipse_color,
            linewidth=ellipse_linewidth,
            fill_alpha=0.0,    # 填充透明度 ← 手动调整
            edge_alpha=0.8      # 边框透明度 ← 手动调整
        )

# ============================================
# 5. ▼▼▼ 对数坐标轴设置 ▼▼▼
# ============================================
ax.set_xscale('log')
ax.set_xlim(0.2, 10)     # 横坐标范围 ← 手动调整

# 主刻度设置
major_ticks = [0.2, 0.5, 1, 2, 5, 10]   # 主刻度位置 ← 手动调整
ax.set_xticks(major_ticks)
ax.xaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())

# 去掉所有次刻度
ax.xaxis.set_minor_locator(plt.matplotlib.ticker.NullLocator())
ax.yaxis.set_minor_locator(plt.matplotlib.ticker.NullLocator())

# ▼▼▼ 设置坐标轴显示范围比例为1:1 ▼▼▼
# 计算合适的纵轴范围，使得在对数坐标下视觉比例为1:1
x_range_log = np.log10(10) - np.log10(0.2)  # 对数空间的x范围
y_center = (Y.max() + Y.min()) / 2
y_half_range = x_range_log / 2

ax.set_ylim(y_center - y_half_range, y_center + y_half_range)  # 纵轴范围自动匹配

# 或者手动设置纵轴范围使比例为1:1 ← 手动调整
ax.set_ylim(1, 10)  # 根据实际数据调整

# ============================================
# 6. 坐标轴标签 & 标题
# ============================================
ax.set_xlabel('Particle size (μm)', fontsize=24, labelpad=21)
ax.set_ylabel('Density (g/cm³)',     fontsize=24, labelpad=21)
ax.set_title('OFP Data — Linear Boundary Clustering',
             fontsize=24, pad=21)

# ============================================
# 7. 图形美化
# ============================================
#ax.legend(fontsize=10, loc='best',
#          framealpha=0.9, edgecolor='lightgray')

# 保留外围四条框线
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)      # 框线宽度 ← 手动调整
    spine.set_color('black')      # 框线颜色 ← 手动调整

ax.grid(False)

# 刻度设置
ax.tick_params(
    axis='both',
    which='major',
    direction='out',      # 刻度朝外 ← 可改为 'in'
    length=5,            # 刻度长度 ← 手动调整
    width=1,             # 刻度宽度 ← 手动调整
    labelsize=21         # 刻度标签字号 ← 手动调整
)

plt.tight_layout()

# ============================================
# 8. 保存
# ============================================
save_path = r"C:\Users\shuzh\Desktop\散点图聚类分析\cluster_line.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n✅ 图片已保存：{save_path}")

# 保存聚类结果到Excel
excel_save = r"C:\Users\shuzh\Desktop\散点图聚类分析\OFP_clustered.xlsx"
df.to_excel(excel_save, index=False)
print(f"✅ 聚类结果已保存：{excel_save}")

plt.show()

