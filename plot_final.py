import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ==============================
# 1. 画“真实值 vs 预测值”散点图 (用刚刚跑出来的 GBDT 数据)
# ==============================
df = pd.read_csv('/home/bigdata/predictions_gbdt.csv')
plt.figure(figsize=(8, 6))
sns.scatterplot(x='TotalAmount', y='prediction', data=df, alpha=0.6, color='#27AE60') # 换成绿色代表最优模型

max_val = max(df['TotalAmount'].max(), df['prediction'].max())
plt.plot([0, max_val], [0, max_val], color='#E74C3C', linestyle='--', label='Ideal Prediction')

plt.title('Actual vs Predicted (GBDT - Best Model)', fontsize=14)
plt.xlabel('Actual Total Amount', fontsize=12)
plt.ylabel('Predicted Total Amount', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('/home/bigdata/final_scatter_gbdt.png', dpi=300)
plt.close()

# ==============================
# 2. 画“三模型 RMSE 对比”柱状图
# ==============================
# 注意：为了图表好看，线性回归那个 2400 亿的极端值我们没有画进来（否则另外两个柱子看不见了），
# 但在 PPT 文字里要提到它崩溃了。
models = ['Random Forest', 'GBDT (Best)']
rmse_values = [63.80, 34.98]

plt.figure(figsize=(7, 5))
sns.barplot(x=models, y=rmse_values, palette=['#85C1E9', '#27AE60'])
plt.title('RMSE Comparison (After Log1p Transformation)', fontsize=14)
plt.ylabel('RMSE (Lower is Better)', fontsize=12)

for i, v in enumerate(rmse_values):
    plt.text(i, v + 1, f"{v:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.ylim(0, 80)
plt.tight_layout()
plt.savefig('/home/bigdata/final_rmse_gbdt.png', dpi=300)
plt.close()

print("终极图表已生成！")

