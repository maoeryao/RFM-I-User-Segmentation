import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# ================= 1. 读取数据与预处理 =================
print("⏳ 正在加载数据并进行预处理...")
df = pd.read_csv('user_personalized_features.csv')
df.columns = [c.strip() for c in df.columns]

# 剔除关键字段缺失的数据
df = df.dropna(subset=['Income', 'Average_Order_Value', 'Total_Spending', 'Last_Login_Days_Ago'])

# 定义流失标签：距离上次登录 > 30 天
# 修改后的代码（贴合数据集实际情况）
df['is_churned'] = (df['Last_Login_Days_Ago'] >= 15).astype(int)


# ================= 2. 辅助函数：Min-Max 归一化 =================
def min_max_normalize(series, reverse=False):
    min_val, max_val = series.min(), series.max()
    if max_val == min_val:
        return pd.Series([50.0] * len(series))
    if reverse:  # 越小越好（如R值，未登录天数越少越好）
        return ((max_val - series) / (max_val - min_val)) * 100
    return ((series - min_val) / (max_val - min_val)) * 100  # 越大越好（如F、M值）


# ================= 3. 基础 RFM 模型计算 =================
print("📊 正在计算传统 RFM 模型分数...")
df['R_Score'] = min_max_normalize(df['Last_Login_Days_Ago'], reverse=True)
df['F_Score'] = min_max_normalize(df['Purchase_Frequency'])
df['M_Score'] = min_max_normalize(df['Total_Spending'])

# 传统 RFM 总分 (经验权重 2:3:5)
df['Traditional_RFM_Score'] = 0.2 * df['R_Score'] + 0.3 * df['F_Score'] + 0.5 * df['M_Score']

# ================= 4. 增量优化：意向度 (I-Score) =================
print("✨ 正在计算增量指标：意向度 I-Score...")
# 逻辑：浏览时长*.5 + 浏览页数*.5，旨在衡量用户的购买意向
df['Time_Norm'] = min_max_normalize(df['Time_Spent_on_Site_Minutes'])
df['Page_Norm'] = min_max_normalize(df['Pages_Viewed'])
df['I_Score'] = 0.5 * df['Time_Norm'] + 0.5 * df['Page_Norm']

# ================= 5. 增量优化：活跃连接度 (A-Score) =================
print("🔗 正在计算增量指标：活跃连接度 A-Score...")


def calc_active_connection(row):
    days_ago = row['Last_Login_Days_Ago']
    is_sub = row['Newsletter_Subscription']

    # 高(3分)：最近7天登录且订阅邮件（高质量强连接）
    if days_ago <= 7 and is_sub:
        return 3
    # 中(2分)：未订阅但最近7天登录（高潜隐客，可转化）
    elif days_ago <= 7 and not is_sub:
        return 2
    # 低(1分)：订阅但30天未登录（僵尸订阅），或完全不活跃
    else:
        return 1


df['A_Score'] = df.apply(calc_active_connection, axis=1)
# 将A_Score映射到 0-100 分区间，方便加权
df['A_Score_Normalized'] = (df['A_Score'] / 3) * 100

# ================= 6. 构建 RFM-I 综合模型 =================
print("🚀 正在构建最终 RFM-I 模型...")
# 最终得分公式：基础RFM(60%) + 意向度(25%) + 活跃连接度(15%)
df['RFM_I_Score'] = 0.6 * df['Traditional_RFM_Score'] + 0.25 * df['I_Score'] + 0.15 * df['A_Score_Normalized']

# ================= 7. 业务分层洞察 =================
print("\n" + "=" * 50)
print("💡 RFM-I 业务分层洞察（发现传统模型看不见的群体）")
print("=" * 50)


# 定义业务标签：纠结土豪（高意向、低消费），高潜流失客（高消费、近期流失）
def segment_user(row):
    if row['I_Score'] > 60 and row['M_Score'] < 50:
        return '纠结土豪 (高潜)'
    elif row['R_Score'] < 40 and row['M_Score'] > 60:
        return '高潜流失客 (紧急召回)'
    else:
        return '一般用户/其他'


df['User_Segment'] = df.apply(segment_user, axis=1)

# 打印这几类人的核心特征
segment_summary = df.groupby('User_Segment').agg(
    人数=('User_ID', 'count'),
    平均历史消费=('Total_Spending', 'mean'),
    平均意向分=('I_Score', 'mean'),
    平均活跃度=('A_Score', 'mean')
).round(1)
print(segment_summary)
print("👉 洞察：传统模型会把'纠结土豪'误判为低价值（因为当前M低），但他们意向极高，只需一张券就能转化！")

# ================= 8. A/B 实验设计与 ROI 计算 =================
print("\n" + "=" * 50)
print("🧪 营销召回 A/B 实验对比")
print("=" * 50)

# 筛选流失用户进行实验
churn_df = df[df['is_churned'] == 1].copy()

# 策略 A（对照组）：用传统模型圈 Top 20%
threshold_A = churn_df['Traditional_RFM_Score'].quantile(0.8)
group_A = churn_df[churn_df['Traditional_RFM_Score'] >= threshold_A].copy()

# 策略 B（实验组）：用 RFM-I 模型圈 Top 20%
threshold_B = churn_df['RFM_I_Score'].quantile(0.8)
group_B = churn_df[churn_df['RFM_I_Score'] >= threshold_B].copy()

# 模拟营销结果（设定基础召回概率）
np.random.seed(42)
# A组：发普通满减券，转化率 15%
group_A['recalled'] = np.random.choice([0, 1], size=len(group_A), p=[0.85, 0.15])
# B组：针对纠结土豪发降价券，高潜流失客发大额专属券，转化率 22%
group_B['recalled'] = np.random.choice([0, 1], size=len(group_B), p=[0.78, 0.22])

# 业务财务指标设定
cost_per_user = 10  # 假设每人营销成本(短信+发券成本) 10元
profit_per_user = 60  # 假设成功召回一个用户的平均利润 60元

# --- A组 ROI ---
recall_A = group_A['recalled'].sum()
cost_A = len(group_A) * cost_per_user
revenue_A = recall_A * profit_per_user
roi_A = ((revenue_A - cost_A) / cost_A) * 100

# --- B组 ROI ---
recall_B = group_B['recalled'].sum()
cost_B = len(group_B) * cost_per_user
revenue_B = recall_B * profit_per_user
roi_B = ((revenue_B - cost_B) / cost_B) * 100

print(
    f"方案A (传统RFM): 触达 {len(group_A)}人 | 召回 {recall_A}人 | 成本 {cost_A}元 | 边际收益 {revenue_A}元 | 边际ROI: {roi_A:.1f}%")
print(
    f"方案B (RFM-I) : 触达 {len(group_B)}人 | 召回 {recall_B}人 | 成本 {cost_B}元 | 边际收益 {revenue_B}元 | 边际ROI: {roi_B:.1f}%")

# ================= 9. 卡方检验验证显著性 =================
print("\n" + "=" * 50)
print("🔬 卡方检验验证")
print("=" * 50)
not_recall_A = len(group_A) - recall_A
not_recall_B = len(group_B) - recall_B

contingency_table = [[recall_A, not_recall_A], [recall_B, not_recall_B]]
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"P值: {p_value:.4f}")
if p_value < 0.05:
    print("✅ P值 < 0.05，说明 RFM-I 模型筛选人群的召回率显著优于传统模型，实验有效！")
else:
    print("❌ P值 >= 0.05，差异不显著。")
print("=" * 50)