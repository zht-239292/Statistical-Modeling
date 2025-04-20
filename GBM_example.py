import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns                                                       
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# 加载数据并查看基本信息
df = pd.read_csv('Student Depression Dataset.csv')
print("数据形状：", df.shape)
print("\n前5行数据：")
print(df.head())
print("\n数据基本信息：")
print(df.info())

# 数据预处理
# 删除无用列
df = df.drop(['id', 'City', 'Profession'], axis=1)

# 学位分类
degree_mapping = {
    'BA': 'Undergraduate', 'BSc': 'Undergraduate', 'BCA': 'Undergraduate', 'B.Pharm': 'Undergraduate',
    'M.Tech': 'Master', 'ME': 'Master', 'M.Com': 'Master', 'PhD': 'Doctorate',
    'High School': 'High School', 'Other': 'Other'
}
df['Degree'] = df['Degree'].replace(degree_mapping)

# 区间划分
df['CGPA'] = pd.cut(df['CGPA'], 
                   bins=[0, 2, 4, 6, 8, 10], 
                   labels=['0-2', '2-4', '4-6', '6-8', '8-10'],
                   include_lowest=True)

df['Work/Study Hours'] = pd.cut(df['Work/Study Hours'], 
                               bins=[0, 3, 6, 9, 12], 
                               labels=['0-3', '3-6', '6-9', '9-12'],
                               include_lowest=True)

# 区分数值型和分类变量
categorical_cols = [
    'Gender', 'Study Satisfaction', 'Job Satisfaction',
    'Sleep Duration', 'Dietary Habits', 'Degree',
    'Financial Stress', 'Family History of Mental Illness',
    'Have you ever had suicidal thoughts ?'  
]
numerical_cols = ['CGPA', 'Work/Study Hours', 'Age', 'Academic Pressure', 'Work Pressure']

# 处理缺失值
imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols + numerical_cols] = imputer.fit_transform(df[categorical_cols + numerical_cols])

# 分类变量列表
categorical_cols += ['CGPA', 'Work/Study Hours']
numerical_cols = [col for col in numerical_cols if col not in ['CGPA', 'Work/Study Hours']]

# 创建副本用于相关性分析
df_corr = df.copy()

# 编码分类变量
le = LabelEncoder()
df['Depression'] = le.fit_transform(df['Depression'])
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 相关性分析预处理
# 对副本数据进行标签编码
for col in categorical_cols + ['Depression']:
    df_corr[col] = le.fit_transform(df_corr[col].astype(str))

# 划分特征和目标变量
X = df.drop('Depression', axis=1).astype(float)
y = df['Depression']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建并训练GBM模型
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbm.fit(X_train, y_train)

# 预测和评估
y_pred = gbm.predict(X_test)
print("\n准确率:", accuracy_score(y_test, y_pred))
print("\n分类报告:\n", classification_report(y_test, y_pred))
print("\n混淆矩阵:\n", confusion_matrix(y_test, y_pred))

# 特征重要性分析
feature_importance = gbm.feature_importances_
def plot_vertical_importance(features, importances, title, top_n=20):
    sorted_idx = np.argsort(importances)[::-1][:top_n]  
    sorted_features = np.array(features)[sorted_idx]
    sorted_importances = importances[sorted_idx]
    plt.figure(figsize=(12, 9))
    colors = sns.color_palette("Blues", len(sorted_importances))[::-1]  
    bars = plt.bar(range(len(sorted_importances)), sorted_importances,
                   color=colors, edgecolor='darkblue', linewidth=0.8)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom',
                 fontsize=9, rotation=45)
    plt.xticks(range(len(sorted_features)), sorted_features,
               rotation=60, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel('Feature Importance', fontsize=12, labelpad=10)
    plt.title(title, fontsize=14, pad=20)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# 绘制特征重要性图
nonzero_mask = feature_importance > 0
valid_features = np.array(X.columns)[nonzero_mask]
valid_importances = feature_importance[nonzero_mask]
plot_vertical_importance(valid_features, valid_importances,
                         'Top 20 Feature Importance (Standard)', top_n=20)

# 绘制相关性热力图
correlation_matrix = df_corr.corr()
plt.figure(figsize=(12, 8))  
sns.heatmap(correlation_matrix, 
           cmap='coolwarm',
           annot=True,
           fmt=".2f",
           annot_kws={"size": 8},
           linewidths=0.5,
           cbar_kws={"shrink": 0.8},
           vmin=-0.4, vmax=0.4,
           center=0)

plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()  
plt.show()

# ROC曲线
y_proba = gbm.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('GBM Receiver Operating Characteristic', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()