import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import shap
import matplotlib.pyplot as plt


# 加载数据并打印基础信息
df = pd.read_csv("Student Depression Dataset.csv")
print("数据形状:", df.shape)
print("\n前5行数据:")
print(df.head())
print("\n数据基本信息:")
print(df.info())

# 删除无用列
df = df.drop(['id', 'City'], axis=1)

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

# 处理目标变量
df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0})

# 划分特征和目标变量
X = df.drop('Depression', axis=1)
y = df['Depression']

# 区分数值型和分类变量
numeric_features = ['Age', 'Academic Pressure', 'Work Pressure', 'Financial Stress']
categorical_features = ['Gender', 'Profession', 'CGPA', 'Study Satisfaction', 
                        'Job Satisfaction', 'Sleep Duration', 'Dietary Habits',
                        'Degree', 'Have you ever had suicidal thoughts ?', 
                        'Work/Study Hours', 'Family History of Mental Illness']

# 创建预处理管道
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))  # 确保数值型列填充
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # 分类特征填充
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练GBM模型
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# 创建完整管道（包含缺失值处理）
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', gbm)
])

# 检查预处理后的数据
X_train_processed = preprocessor.fit_transform(X_train)
print("\n预处理后的训练集形状:", X_train_processed.shape)
print("是否存在NaN值:", np.isnan(X_train_processed.toarray()).any())  
pipeline.fit(X_train, y_train)

# 预测并评估模型
y_pred = pipeline.predict(X_test)
print("\n准确率:", accuracy_score(y_test, y_pred))
print("\n混淆矩阵:\n", confusion_matrix(y_test, y_pred))
print("\n分类报告:\n", classification_report(y_test, y_pred))

# SHAP特征重要性图
preprocessor = pipeline.named_steps['preprocessor']
feature_names = (
    numeric_features + 
    list(preprocessor.named_transformers_['cat']
         .named_steps['onehot']
         .get_feature_names_out(categorical_features))
)
X_test_preprocessed = preprocessor.transform(X_test)
if hasattr(X_test_preprocessed, "toarray"):
    X_test_dense = X_test_preprocessed.toarray().astype(np.float64)
else:
    X_test_dense = X_test_preprocessed.astype(np.float64)
assert not np.isnan(X_test_dense).any(), "存在NaN值"
assert not np.isinf(X_test_dense).any(), "存在Inf值"
print("数据维度:", X_test_dense.shape)  


# 计算SHAP值（top20）
explainer = shap.TreeExplainer(gbm, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test_dense, check_additivity=False)
def plot_shap_with_values(shap_values, features, feature_names, max_display=20):
    plt.figure(figsize=(12, 10))
    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame({
        'col': feature_names,
        'importance': vals
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(max_display)
    y_pos = np.arange(len(feature_importance))
    bars = plt.barh(y_pos, feature_importance['importance'], align='center', color='#1f77b4')
    plt.yticks(y_pos, feature_importance['col'])
    plt.ylim(-0.5, len(feature_importance)-0.5) 
    plt.xlabel('Average SHAP Value', fontsize=12)
    plt.title('SHAP Feature Importance (Corrected)', fontsize=14, y=1.02)  
    for idx, (_, row) in enumerate(feature_importance.iterrows()):
        value = row['importance']
        plt.text(
            x = value + 0.001,
            y = idx,
            s = f"{value:.4f}",
            ha = 'left',
            va = 'center',
            fontsize=10
        )
    plt.gca().invert_yaxis()
    plt.subplots_adjust(top=0.95, left=0.3)  
    plt.tight_layout(rect=[0, 0, 1, 0.98]) 
    plt.show()
plot_shap_with_values(shap_values, X_test_dense, feature_names, max_display=20)


# SHAP蜂群图（显示特征值分布与影响方向）
#蜂群图中变量Age:红色表示年龄大的数值（年龄大抑郁症概率越低）
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, 
                 X_test_dense, 
                 feature_names=feature_names,
                 max_display=15,  
                 plot_type="dot", 
                 color=plt.get_cmap("coolwarm"),  
                 show=False)


plt.title("SHAP Feature Impact (Beeswarm Plot)", fontsize=14, pad=20)
plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.gca().xaxis.grid(True, linestyle='--', alpha=0.6) 
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
cbar = plt.gcf().axes[-1]
cbar.set_ylabel('Feature Value', size=10)
cbar.set_yticklabels(['Low', 'High'], size=9)
plt.tight_layout()
plt.show()