import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# 加载数据并打印基础信息
data = pd.read_csv('E:/depression research/Student Depression Dataset.csv')
print('Data basic information:')
data.info()

# 查看并打印数据形状相关信息
rows, cols = data.shape
print(('Full data content information:' if rows < 100 and cols < 20
       else 'First few rows of data content information:'))
print(data.head().to_csv(sep='\t', na_rep='nan') if rows >= 100 or cols >= 20 else data.to_csv(sep='\t', na_rep='nan'))

# 删除无用列并填充缺失值
data = data.drop(columns=['Job Satisfaction', 'Work Pressure', 'Profession'])
data['Financial Stress'].fillna(data['Financial Stress'].mean(), inplace=True)

# 数据处理
# 城市分类
city_mapping = {city: 'Developed City' for city in ['Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'Bangalore', 'Hyderabad']}
city_mapping.update({city: 'Developing City' for city in ['Pune', 'Ahmedabad', 'Surat', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur', 'Indore']})
city_mapping.update({city: 'Emerging City' for city in ['Visakhapatnam', 'Varanasi', 'Rajkot', 'Ludhiana', 'Bhopal', 'Meerut', 'Agra', 'Ghaziabad', 'Vasai-Virar', 'Thane', 'Nashik', 'Patna', 'Faridabad']})
data['City'] = data['City'].map(city_mapping).fillna('Unknown')

# 区间划分
data['CGPA'] = pd.cut(data['CGPA'], bins=[0, 2, 4, 6, 8, 10], labels=['0-2', '2-4', '4-6', '6-8', '8-10'])
data['Work/Study Hours'] = pd.cut(data['Work/Study Hours'], bins=[0, 3, 6, 9, 12], labels=['0-3', '3-6', '6-9', '9-12'])

# 学位分类
degree_mapping = {
    'BA': 'Undergraduate', 'BSc': 'Undergraduate', 'BCA': 'Undergraduate', 'B.Pharm': 'Undergraduate',
    'M.Tech': 'Master', 'ME': 'Master', 'M.Com': 'Master', 'PhD': 'Doctorate',
    'High School': 'High School', 'Other': 'Other'
}
data['Degree'] = data['Degree'].map(degree_mapping).fillna('Other')

# 划分特征和目标变量
X, y = data.drop(columns=['id', 'Depression']), data['Depression']

# 区分数值型和分类变量
numeric_features = ['Age', 'Academic Pressure', 'Study Satisfaction', 'Financial Stress']
categorical_features = ['Gender', 'City', 'Degree', 'Have you ever had suicidal thoughts ?',
                        'Family History of Mental Illness', 'Dietary Habits', 'CGPA', 'Work/Study Hours', 'Sleep Duration']

# # 对数值型变量进行对数转换
# for col in numeric_features:
#     # 加一个小常数避免对数中的 0 值
#     X[col] = np.log(X[col] + 0.001)

# 标签编码（合并循环）
for col in categorical_features:
    data[col] = LabelEncoder().fit_transform(data[col])

# 数据预处理
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])
preprocess_pipeline = Pipeline([('preprocessor', preprocessor)])
X_processed = preprocess_pipeline.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# ########### 'C': 0.01, 'class_weight': None, 'solver': 'lbfgs'
# # 定义参数网格
# param_grid = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100],
#     'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
#     'class_weight': [None, 'balanced']
# }

# # 创建逻辑回归模型
# model = LogisticRegression()

# # 使用 GridSearchCV 进行参数搜索
# grid_search = GridSearchCV(model, param_grid, cv=5)
# grid_search.fit(X_train, y_train)

# # 输出最佳参数和最佳准确率
# print(f"最佳参数: {grid_search.best_params_}")
# print(f"最佳准确率: {grid_search.best_score_}")
# ##########

# 创建并训练逻辑回归模型
model = LogisticRegression(C=0.1, class_weight=None, solver='liblinear')
model.fit(X_train, y_train)

# 预测并评估模型
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Model accuracy: {accuracy}')
print(f'Model precision: {precision}')
print(f'Model recall: {recall}')
print(f'Model F1 score: {f1}')

# 分析特征重要性
coefficients = pd.Series(model.coef_[0], index=preprocess_pipeline.get_feature_names_out())

# 计算另一种特征重要性（按原始特征聚合）
feature_importance = {}
for feature in numeric_features:
    prefix = f'num__{feature}'
    relevant_coeffs = coefficients[coefficients.index.str.startswith(prefix)]
    feature_importance[feature] = relevant_coeffs.abs().sum()
for feature in categorical_features:
    sanitized_feature = feature.replace(' ', '_').replace('?', '')
    prefix = f'cat__{sanitized_feature}'
    relevant_coeffs = coefficients[coefficients.index.str.startswith(prefix)]
    feature_importance[feature] = relevant_coeffs.abs().mean()
feature_importance_series = pd.Series(feature_importance).sort_values(ascending=False)

print(coefficients.sort_values(ascending=False))

# 绘制相关性热力图
corr_matrix = data.corr()
plt.figure(figsize=(12, 8))
cmap = sns.diverging_palette(220, 10, s=90, l=50, as_cmap=True)
sns.heatmap(corr_matrix, annot=True, cmap=cmap, vmin=-1, vmax=1,
            annot_kws={"size": 8}, fmt='.2f', linewidths=0.5, alpha=0.9)
plt.title('Correlation Heatmap', fontsize=16)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# 绘制特征重要性柱状图
plt.figure(figsize=(10, 6))
ax = feature_importance_series.plot(kind='bar', color=sns.color_palette("husl", len(feature_importance_series)))
plt.title('Feature Importance', fontsize=16)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Coefficients', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
for p in ax.patches:
    ax.annotate(str(round(p.get_height(), 2)), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=8)
plt.tight_layout()
plt.show()

# 绘制详细特征重要性柱状图（前20）
top_features = coefficients.abs().nlargest(20).index if len(coefficients) > 20 else coefficients.index
top_coefficients = coefficients[top_features]
plt.figure(figsize=(10, 6))
ax = top_coefficients.plot(kind='bar', color=sns.color_palette("husl", len(top_coefficients)))
plt.title('Feature Importance', fontsize=16)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Coefficients', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
for p in ax.patches:
    ax.annotate(str(round(p.get_height(), 2)), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=8)
plt.tight_layout()
plt.show()

# 绘制详细特征重要性柱状图（全部）
sorted_coefficients = coefficients.abs().sort_values(ascending=False).index
sorted_coefficients_with_sign = coefficients[sorted_coefficients]
plt.figure(figsize=(10, 6))
ax = sorted_coefficients_with_sign.plot(kind='bar', color=sns.color_palette("husl", len(sorted_coefficients_with_sign)))
plt.title('Feature Importance', fontsize=16)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Coefficients', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
for p in ax.patches:
    ax.annotate(str(round(p.get_height(), 2)), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=8)
plt.tight_layout()
plt.show()

# 绘制 ROC 曲线
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()