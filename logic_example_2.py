import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据并打印基础信息
data = pd.read_csv('E:/depression research/Student Depression Dataset.csv')
print('Data basic information:'); data.info()

# 查看并打印数据形状相关信息
rows, cols = data.shape
print(('Full data content information:' if rows < 100 and cols < 20
       else 'First few rows of data content information:'))
print(data.head().to_csv(sep='\t', na_rep='nan') if rows >= 100 or cols >= 20 else data.to_csv(sep='\t', na_rep='nan'))

# 数据预处理
data = data.drop(columns=['Job Satisfaction', 'Work Pressure', 'Profession'])
data['Financial Stress'].fillna(data['Financial Stress'].median(), inplace=True)

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

# 数据预处理
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])
X_processed = preprocessor.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 创建并训练逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测并评估模型
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(f'Precision: {precision_score(y_test, y_pred):.2f}')
print(f'Recall: {recall_score(y_test, y_pred):.2f}')
print(f'F1 Score: {f1_score(y_test, y_pred):.2f}')

# 分析特征重要性
coefficients = pd.Series(model.coef_[0], index=preprocessor.get_feature_names_out())

# 特征重要性（按原始特征聚合）
feature_importance = {}
for feature in numeric_features:
    prefix = f'num__{feature}'
    feature_importance[feature] = coefficients[coefficients.index.str.startswith(prefix)].abs().sum()
for feature in categorical_features:
    prefix = f'cat__{feature.replace(" ", "_").replace("?", "")}'
    feature_importance[feature] = coefficients[coefficients.index.str.startswith(prefix)].abs().mean()
feature_importance_series = pd.Series(feature_importance).sort_values(ascending=False)

# 输出特征重要性
print("聚合后的特征重要性：")
print(feature_importance_series)
print("详细特征重要性：")
print(coefficients.abs().sort_values(ascending=False))

# 绘制特征重要性图
def plot_feature_importance(importance, title, top_n=20):
    plt.figure(figsize=(10, 6))
    top_features = importance.nlargest(top_n)
    ax = top_features.plot(kind='bar', color=sns.color_palette("husl", len(top_features)))
    plt.title(title, fontsize=16)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=8)
    plt.tight_layout()
    plt.show()

# 绘制特征重要性图
plot_feature_importance(feature_importance_series, 'Feature Importance (Aggregated)')

# 绘制详细特征重要性图
plot_feature_importance(coefficients.abs(), 'Feature Importance (Detailed)')

# 绘制包含所有特征的特征重要性图
plt.figure(figsize=(10, 6))
ax = coefficients.abs().sort_values(ascending=False).plot(kind='bar', color=sns.color_palette("husl", len(coefficients)))
plt.title('Feature Importance (All Features)', fontsize=16)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Importance', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=8)
plt.tight_layout()
plt.show()

# 对分类变量进行标签编码
data_encoded = data.copy()
for col in categorical_features:
    le = LabelEncoder()
    data_encoded[col] = le.fit_transform(data_encoded[col])

# 绘制相关性热力图
plt.figure(figsize=(12, 8))
# 使用从蓝色到红色的调色板
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(data_encoded.corr(), annot=True, cmap=cmap, vmin=-0.4, vmax=0.4, fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# 绘制 ROC 曲线
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
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

# 绘制小提琴图
for feature in numeric_features + categorical_features:
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Depression', y=feature, data=data_encoded)
    plt.title(f'Distribution of {feature} by Depression Status')
    plt.xlabel('Depression Status (0: No Depression, 1: Depression)')
    plt.ylabel(f'{feature} Value')
    plt.show()
