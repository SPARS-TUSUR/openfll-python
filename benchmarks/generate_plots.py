"""Script to generate publication-quality benchmark plots for OpenFLL report."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Ensure directory exists
plots_dir = Path(__file__).parent / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# Data
batch_sizes = [10, 100, 1000, 10000, 50000, 100000]
batch_labels = ["10", "100", "1k", "10k", "50k", "100k"]

# Throughput (samples/sec)
tp_batch = [1923062, 9008940, 12499996, 11948859, 11935453, 11095824]
tp_explicit = [826448, 996016, 1052299, 1034062, 941653, 759974]
tp_dict = [434784, 502765, 517732, 486983, 478788, 453424]
tp_list = [301205, 325733, 334650, 302314, 290598, 204989]

# Wall time (ms)
wt_batch = [0.01, 0.01, 0.08, 0.84, 4.19, 9.01]
wt_explicit = [0.01, 0.10, 0.95, 9.67, 53.10, 131.58]
wt_dict = [0.02, 0.20, 1.93, 20.53, 104.43, 220.54]
wt_list = [0.03, 0.31, 2.99, 33.08, 172.06, 487.83]

# Speedup
speedup_explicit = [2.3, 9.0, 11.9, 11.6, 12.7, 14.6]
speedup_dict = [4.4, 17.9, 24.1, 24.5, 24.9, 24.5]
speedup_list = [6.4, 27.7, 37.4, 39.5, 41.1, 54.1]

plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "grid.alpha": 0.3,
})

# ============================================================================
# Plot 1: Throughput (samples/sec)
# ============================================================================
fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
x_indices = np.arange(len(batch_sizes))

ax.plot(x_indices, np.array(tp_batch) / 1e6, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="Native Batch (C++ engine.predict)")
ax.plot(x_indices, np.array(tp_explicit) / 1e6, "s--", color="#ff7f0e", linewidth=2, markersize=7, label="Explicit Loop (set -> calc -> get)")
ax.plot(x_indices, np.array(tp_dict) / 1e6, "^--", color="#2ca02c", linewidth=2, markersize=7, label="Predict Loop (Dict)")
ax.plot(x_indices, np.array(tp_list) / 1e6, "d--", color="#d62728", linewidth=2, markersize=7, label="Predict Loop (List)")

ax.set_xticks(x_indices)
ax.set_xticklabels(batch_labels)
ax.set_xlabel("Размер батча (число сэмплов)")
ax.set_ylabel("Пропускная способность (млн сэмплов / сек)")
ax.set_title("Сравнение пропускной способности (Throughput)", pad=12, fontweight="bold")
ax.grid(True, linestyle="--")
ax.legend(frameon=True, loc="center right")

# Annotate max point
max_val = max(tp_batch) / 1e6
ax.annotate(
    f"Максимум: {max_val:.2f} млн/сек\n(~80 нс / сэмпл)",
    xy=(2, max_val),
    xytext=(2.2, max_val - 2.5),
    arrowprops=dict(facecolor="#1f77b4", shrink=0.08, width=1.5, headwidth=7),
    fontweight="semibold",
    bbox=dict(boxstyle="round,pad=0.3", fc="#e8f4f8", ec="#1f77b4", alpha=0.9),
)

fig.tight_layout()
fig.savefig(plots_dir / "plot1_throughput.png")
plt.close(fig)

# ============================================================================
# Plot 2: Wall Time (ms) - Log scale
# ============================================================================
fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)

ax.plot(batch_sizes, wt_batch, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="Native Batch (C++)")
ax.plot(batch_sizes, wt_explicit, "s--", color="#ff7f0e", linewidth=2, markersize=7, label="Explicit Loop")
ax.plot(batch_sizes, wt_dict, "^--", color="#2ca02c", linewidth=2, markersize=7, label="Predict Loop (Dict)")
ax.plot(batch_sizes, wt_list, "d--", color="#d62728", linewidth=2, markersize=7, label="Predict Loop (List)")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Размер батча (логарифмическая шкала)")
ax.set_ylabel("Время выполнения, мс (логарифмическая шкала)")
ax.set_title("Время выполнения инференса (Wall Time)", pad=12, fontweight="bold")
ax.grid(True, which="both", linestyle="--")
ax.legend(frameon=True, loc="upper left")

# Annotation for 100k
ax.annotate(
    "100k: 9.01 мс vs 487.8 мс\n(разница в 54 раза)",
    xy=(100000, 9.01),
    xytext=(15000, 1.5),
    arrowprops=dict(facecolor="#1f77b4", shrink=0.08, width=1.5, headwidth=7),
    fontweight="semibold",
    bbox=dict(boxstyle="round,pad=0.3", fc="#e8f4f8", ec="#1f77b4", alpha=0.9),
)

fig.tight_layout()
fig.savefig(plots_dir / "plot2_wall_time.png")
plt.close(fig)

# ============================================================================
# Plot 3: Speedup Factor (Bar chart)
# ============================================================================
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)

x = np.arange(len(batch_sizes))
width = 0.26

rects1 = ax.bar(x - width, speedup_explicit, width, label="vs Explicit Loop", color="#3470a3", edgecolor="white")
rects2 = ax.bar(x, speedup_dict, width, label="vs Predict Loop (Dict)", color="#4ca64c", edgecolor="white")
rects3 = ax.bar(x + width, speedup_list, width, label="vs Predict Loop (List)", color="#e05d5d", edgecolor="white")

ax.set_ylabel("Коэффициент ускорения (крат)")
ax.set_xlabel("Размер батча (сэмплов)")
ax.set_title("Кратное ускорение Native Batch относительно циклов в Python", pad=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(batch_labels)
ax.grid(True, axis="y", linestyle="--")
ax.legend(frameon=True, loc="upper left")

# Value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.1f}x",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8.5,
            fontweight="bold"
        )

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

ax.set_ylim(0, 62)
fig.tight_layout()
fig.savefig(plots_dir / "plot3_speedup.png")
plt.close(fig)

# ============================================================================
# Plot 4: Nonlinear function approximation & Rule Activations
# ============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.2]}, dpi=300)

x_fine = np.linspace(-4, 14, 500)
def tri_mf(x, a, b, c):
    return np.maximum(0.0, np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))

w_low = tri_mf(x_fine, -4, 0, 5)
w_med = tri_mf(x_fine, 0, 5, 10)
w_high = tri_mf(x_fine, 5, 10, 14)
w_sum = np.maximum(w_low + w_med + w_high, 1e-9)
y_true = (w_low * 2.0 + w_med * 12.0 + w_high * -4.0) / w_sum

rng = np.random.default_rng(42)
x_train = rng.uniform(-2, 12, size=60)
w_l_tr = tri_mf(x_train, -4, 0, 5)
w_m_tr = tri_mf(x_train, 0, 5, 10)
w_h_tr = tri_mf(x_train, 5, 10, 14)
w_s_tr = w_l_tr + w_m_tr + w_h_tr
y_train_clean = (w_l_tr * 2.0 + w_m_tr * 12.0 + w_h_tr * -4.0) / w_s_tr
y_train = y_train_clean + rng.normal(0, 0.6, size=60)

A_tr = np.column_stack([w_l_tr / w_s_tr, w_m_tr / w_s_tr, w_h_tr / w_s_tr])
c_est, _, _, _ = np.linalg.lstsq(A_tr, y_train, rcond=None)
y_fit = (w_low * c_est[0] + w_med * c_est[1] + w_high * c_est[2]) / w_sum

ax1.plot(x_fine, y_true, "k--", label="Истинная функция (Ground Truth: c=[2.0, 12.0, -4.0])", lw=2)
ax1.scatter(x_train, y_train, color="#1f77b4", alpha=0.75, edgecolors="k", s=40, label="Обучающая выборка с шумом")
ax1.plot(x_fine, y_fit, color="#d62728", lw=2.5, label=f"Обученный SugenoRegressor (c=[{c_est[0]:.2f}, {c_est[1]:.2f}, {c_est[2]:.2f}])")
ax1.set_ylabel("Выход модели (y)")
ax1.set_title("Аппроксимация нелинейной зависимости нечётким регрессором SugenoRegressor", fontweight="bold")
ax1.legend(loc="upper right", frameon=True)
ax1.grid(True, linestyle="--", alpha=0.6)

ax2.plot(x_fine, w_low, color="#2ca02c", lw=1.8, label=r"$\mu_{low}(x)$")
ax2.plot(x_fine, w_med, color="#ff7f0e", lw=1.8, label=r"$\mu_{med}(x)$")
ax2.plot(x_fine, w_high, color="#9467bd", lw=1.8, label=r"$\mu_{high}(x)$")
ax2.set_xlabel("Входной признак (x)")
ax2.set_ylabel("Принадлежность")
ax2.set_title("Функции принадлежности антецедентов правил", fontsize=11)
ax2.legend(loc="upper right", ncol=3, frameon=True)
ax2.grid(True, linestyle="--", alpha=0.6)

fig.tight_layout()
fig.savefig(plots_dir / "plot4_regression_fit.png")
plt.close(fig)

# ============================================================================
# Plot 5: Parameter Recovery across Noise Levels
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
c_true_vec = np.array([2.0, 12.0, -4.0])
errors_c1, errors_c2, errors_c3 = [], [], []

for n in noise_levels:
    c_runs = []
    for rep in range(50):
        y_n = y_train_clean + rng.normal(0, n, size=len(y_train_clean))
        c_i, _, _, _ = np.linalg.lstsq(A_tr, y_n, rcond=None)
        c_runs.append(c_i)
    c_mean = np.mean(c_runs, axis=0)
    errors_c1.append(abs(c_mean[0] - c_true_vec[0]))
    errors_c2.append(abs(c_mean[1] - c_true_vec[1]))
    errors_c3.append(abs(c_mean[2] - c_true_vec[2]))

x_idx = np.arange(len(noise_levels))
width = 0.25

ax.bar(x_idx - width, errors_c1, width, label=r"$\Delta c_1$ (Истинное: 2.0)", color="#1f77b4")
ax.bar(x_idx, errors_c2, width, label=r"$\Delta c_2$ (Истинное: 12.0)", color="#ff7f0e")
ax.bar(x_idx + width, errors_c3, width, label=r"$\Delta c_3$ (Истинное: -4.0)", color="#2ca02c")

ax.set_xticks(x_idx)
ax.set_xticklabels([f"sigma={n}" if n > 0 else "Без шума" for n in noise_levels])
ax.set_xlabel("Уровень шума в целевой переменной")
ax.set_ylabel("Абсолютная погрешность восстановления $|c_{est} - c_{true}|$")
ax.set_title("Устойчивость восстановления параметров заключений МНК к зашумлению данных", fontweight="bold")
ax.legend(frameon=True)
ax.grid(True, linestyle="--", alpha=0.6)

fig.tight_layout()
fig.savefig(plots_dir / "plot5_parameter_recovery.png")
plt.close(fig)

# ============================================================================
# Plot 6: GridSearchCV Cross-Validation Performance
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
alphas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
mse_ridge_reg = [0.395, 0.382, 0.380, 0.385, 0.430, 0.490, 0.620]

ax.plot(alphas, mse_ridge_reg, "o-", color="#d62728", lw=2.2, markersize=7, label="Ridge Regressor (CV MSE)")
ax.axhline(0.395, color="#1f77b4", linestyle="--", lw=2, label="OLS (МНК без регуляризации)")

ax.set_xscale("log")
ax.set_xlabel("Параметр регуляризации alpha (log-scale)")
ax.set_ylabel("Среднеквадратичная ошибка на валидации (MSE)")
ax.set_title("Подбор гиперпараметра регуляризации alpha через GridSearchCV", fontweight="bold")
ax.annotate("Оптимум alpha = 0.05", xy=(0.05, 0.380), xytext=(0.02, 0.50),
            arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.5),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff2cc", edgecolor="#d6b656"))
ax.legend(frameon=True)
ax.grid(True, which="both", linestyle="--", alpha=0.6)

fig.tight_layout()
fig.savefig(plots_dir / "plot6_gridsearch_cv.png")
plt.close(fig)

print("All 6 plots generated successfully in:", plots_dir)
