import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. 数据加载和基本信息
df = pd.read_csv("Student Depression Dataset.csv")
print("数据形状:", df.shape)
print("\n前5行数据:")
print(df.head())
print("\n缺失值统计:")
print(df.isnull().sum())
print("\n数据类型:")
print(df.dtypes)

# 2. 数据预处理
# 删除无用列
df = df.drop(columns=['id'])

# 分类转换category类型
df['Academic Pressure'] = df['Academic Pressure'].astype('category')
df['Financial Stress'] = df['Financial Stress'].astype('category')
df['Dietary Habits'] = df['Dietary Habits'].astype('category')
df['Sleep Duration'] = df['Sleep Duration'].astype('category')
df['Study Satisfaction'] = pd.to_numeric(df['Study Satisfaction'], errors='coerce')

# 区间划分
df['CGPA'] = pd.cut(df['CGPA'], bins=[0, 2, 4, 6, 8, 10], 
                   labels=['0-2', '2-4', '4-6', '6-8', '8-10'])
df['Work/Study Hours'] = pd.cut(df['Work/Study Hours'], bins=[0, 3, 6, 9, 12],
                               labels=['0-3', '3-6', '6-9', '9-12'])
df['Work/Study Hours'] = pd.Categorical(
    df['Work/Study Hours'],
    categories=['0-3','3-6','6-9','9-12'],
    ordered=True
)

#学位分类
degree_mapping = {
    'BA': 'Undergraduate', 'BSc': 'Undergraduate', 'BCA': 'Undergraduate', 'B.Pharm': 'Undergraduate',
    'M.Tech': 'Master', 'ME': 'Master', 'M.Com': 'Master', 'PhD': 'Doctorate',
    'High School': 'High School', 'Other': 'Other'
}
df['Degree'] = df['Degree'].map(degree_mapping).fillna('Other')

# 处理缺失值
numeric_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(exclude=np.number).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 编码分类变量
numeric_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(include=['category', 'object']).columns
required_categorical = ['Financial Stress', 'Dietary Habits', 'Sleep Duration','Study Satisfaction']
missing = [col for col in required_categorical if col not in categorical_cols]

# 执行编码
le_dict = {}
for col in categorical_cols:
    if col != 'Depression':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str)) 
        le_dict[col] = le

# 划分数据集
X = df.drop(columns=['Depression'])
y = df['Depression']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练GBM模型
gbm = GradientBoostingClassifier(random_state=42)
gbm.fit(X_train, y_train)

# 预测和评估
y_pred = gbm.predict(X_test)
print("\n模型准确率:", accuracy_score(y_test, y_pred))
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 多因素交互热力图

# 自杀倾向+工作/学习时长
plt.figure(figsize=(12, 6))
cross_tab_suicidal_hours = pd.crosstab(
    index=df['Have you ever had suicidal thoughts ?'],
    columns=df['Work/Study Hours'],
    values=df['Depression'],
    aggfunc='mean'
)
suicidal_labels = le_dict['Have you ever had suicidal thoughts ?'].classes_ 
hours_labels = le_dict['Work/Study Hours'].classes_
cross_tab_suicidal_hours.index = [suicidal_labels[i] for i in cross_tab_suicidal_hours.index]
cross_tab_suicidal_hours.columns = [hours_labels[i] for i in cross_tab_suicidal_hours.columns]
sns.heatmap(cross_tab_suicidal_hours, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Depression Rate by Suicidal Thoughts and Work/Study Hours")
plt.xlabel("Work/Study Hours")
plt.ylabel("Have you ever had suicidal thoughts?")
plt.show()

# 自杀倾向 + 学业压力
plt.figure(figsize=(10, 6))
cross_tab_suicidal_academic = pd.crosstab(
    index=df['Have you ever had suicidal thoughts ?'],
    columns=df['Academic Pressure'],
    values=df['Depression'],
    aggfunc='mean'
)
suicidal_labels = le_dict['Have you ever had suicidal thoughts ?'].classes_
academic_labels = le_dict['Academic Pressure'].classes_
cross_tab_suicidal_academic.index = [suicidal_labels[i] for i in cross_tab_suicidal_academic.index]
cross_tab_suicidal_academic.columns = [academic_labels[i] for i in cross_tab_suicidal_academic.columns]
sns.heatmap(cross_tab_suicidal_academic, 
           annot=True, 
           fmt=".2f", 
           cmap="coolwarm",
           vmin=0,
           vmax=1)
plt.title("Depression Rate by Suicidal Thoughts and Academic Pressure")
plt.xlabel("Academic Pressure Level")
plt.ylabel("Had Suicidal Thoughts")
plt.show()

# 自杀倾向 + 家族病史
plt.figure(figsize=(8, 6))
cross_tab_family_suicidal = pd.crosstab(
    index=df['Family History of Mental Illness'],
    columns=df['Have you ever had suicidal thoughts ?'],
    values=df['Depression'],
    aggfunc='mean'
)
family_labels = le_dict['Family History of Mental Illness'].classes_
suicidal_labels = le_dict['Have you ever had suicidal thoughts ?'].classes_
cross_tab_family_suicidal.index = [family_labels[i] for i in cross_tab_family_suicidal.index]
cross_tab_family_suicidal.columns = [suicidal_labels[i] for i in cross_tab_family_suicidal.columns]
sns.heatmap(cross_tab_family_suicidal, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Depression Rate by Family History and Suicidal Thoughts")
plt.xlabel("Have you ever had suicidal thoughts?")
plt.ylabel("Family History of Mental Illness")
plt.show()

# 学业压力 + 财务压力
plt.figure(figsize=(8, 6))
mask = df['Academic Pressure'] != 0  # 排除编码为0的行
cross_tab_academic_financial = pd.crosstab(
    index=df.loc[mask, 'Academic Pressure'],  
    columns=df.loc[mask, 'Financial Stress'],
    values=df.loc[mask, 'Depression'],
    aggfunc='mean'
)
academic_labels = le_dict['Academic Pressure'].classes_
financial_labels = le_dict['Financial Stress'].classes_
cross_tab_academic_financial.index = [academic_labels[i] for i in cross_tab_academic_financial.index]
cross_tab_academic_financial.columns = [financial_labels[i] for i in cross_tab_academic_financial.columns]
sns.heatmap(cross_tab_academic_financial, 
           annot=True, 
           fmt=".2f", 
           cmap="coolwarm",
           vmin=0,
           vmax=1)
plt.title("Depression Rate by Academic Pressure and Financial Stress")
plt.xlabel("Financial Stress Level")
plt.ylabel("Academic Pressure Level")
plt.show()


# 家族病史 + 财务压力
plt.figure(figsize=(8, 6))
cross_tab_family_financial = pd.crosstab(
    index=df['Family History of Mental Illness'],
    columns=df['Financial Stress'],
    values=df['Depression'],
    aggfunc='mean'
)
family_labels = le_dict['Family History of Mental Illness'].classes_
cross_tab_family_financial.index = [family_labels[i] for i in cross_tab_family_financial.index]
cross_tab_family_financial.columns = [financial_labels[i] for i in cross_tab_family_financial.columns]
sns.heatmap(cross_tab_family_financial, 
           annot=True, 
           fmt=".2f", 
           cmap="coolwarm",
           vmin=0,
           vmax=0.9)
plt.title("Depression Rate by Family History and Financial Stress")
plt.xlabel("Financial Stress Level")
plt.ylabel("Family History of Mental Illness")
plt.show()

# 睡眠时长 <5h + 不健康饮食
plt.figure(figsize=(13, 6))
cross_tab_sleep_diet = pd.crosstab(
    index=df['Sleep Duration'],
    columns=df['Dietary Habits'],
    values=df['Depression'],
    aggfunc='mean'
)
# 转换标签
sleep_labels = le_dict['Sleep Duration'].classes_
diet_labels = le_dict['Dietary Habits'].classes_
cross_tab_sleep_diet.index = [sleep_labels[i] for i in cross_tab_sleep_diet.index]
cross_tab_sleep_diet.columns = [diet_labels[i] for i in cross_tab_sleep_diet.columns]
#突出显示关键组合
highlight_mask = np.array([[("Less than 5 hours" in sl) and ("Unhealthy" in dl) 
                          for dl in diet_labels] for sl in sleep_labels])
sns.heatmap(cross_tab_sleep_diet, 
           annot=True, 
           fmt=".2f", 
           cmap="coolwarm",
           vmin=0,
           vmax=1,
           mask=~highlight_mask)  # 仅突出显示目标组合
plt.title("Depression Rate by Sleep Duration and Dietary Habits\n(Highlight: <5h Sleep + Unhealthy Diet)")
plt.xlabel("Dietary Habits")
plt.ylabel("Sleep Duration")
plt.show()

# 学业压力 + 睡眠时长
plt.figure(figsize=(12, 8))
sleep_labels = le_dict['Sleep Duration'].classes_
others_code = list(sleep_labels).index("Others")  
mask = df['Sleep Duration'] != others_code  

cross_tab_AP_Sleep = pd.crosstab(
    index=df.loc[mask, 'Sleep Duration'],     
    columns=df.loc[mask, 'Academic Pressure'], 
    values=df.loc[mask, 'Depression'],
    aggfunc='mean'
)
cross_tab_AP_Sleep.index = [sleep_labels[i] for i in cross_tab_AP_Sleep.index]
cross_tab_AP_Sleep.columns = [academic_labels[i] for i in cross_tab_AP_Sleep.columns]
sns.heatmap(cross_tab_AP_Sleep, 
           annot=True, 
           fmt=".2f", 
           cmap="coolwarm",
           vmin=0,
           vmax=1,
)
plt.title("Depression Rate by Academic Pressure and Sleep Duration", pad=20)
plt.xlabel("Academic Pressure Level ", labelpad=10)
plt.ylabel("Sleep Duration (Filtered)", labelpad=10)  
plt.xticks(rotation=0)  
plt.yticks(rotation=0)  
plt.show()

# 财务压力 + 自杀倾向
plt.figure(figsize=(8, 6))
cross_tab_financial_suicidal = pd.crosstab(
    index=df['Financial Stress'],
    columns=df['Have you ever had suicidal thoughts ?'],
    values=df['Depression'],
    aggfunc='mean'
)
# 转换标签
financial_labels = le_dict['Financial Stress'].classes_
suicidal_labels = le_dict['Have you ever had suicidal thoughts ?'].classes_
cross_tab_financial_suicidal.index = [financial_labels[i] for i in cross_tab_financial_suicidal.index]
cross_tab_financial_suicidal.columns = [suicidal_labels[i] for i in cross_tab_financial_suicidal.columns]
sns.heatmap(cross_tab_financial_suicidal, 
           annot=True, 
           fmt=".2f", 
           cmap="coolwarm",
           vmin=0,
           vmax=1)
plt.title("Depression Rate by Financial Stress and Suicidal Thoughts")
plt.xlabel("Have you ever had suicidal thoughts ?")
plt.ylabel("Financial Stress Level")
plt.show()

# 自杀倾向 + 睡眠时长
plt.figure(figsize=(13, 6))
sleep_labels = le_dict["Sleep Duration"].classes_      
suicidal_labels = le_dict["Have you ever had suicidal thoughts ?"].classes_ 
cross_tab_suicidal_sleep = pd.crosstab(
    index=df["Sleep Duration"],          
    columns=df["Have you ever had suicidal thoughts ?"],  
    values=df["Depression"],
    aggfunc="mean"
)
cross_tab_suicidal_sleep.index = [sleep_labels[i] for i in cross_tab_suicidal_sleep.index]  
cross_tab_suicidal_sleep.columns = [suicidal_labels[i] for i in cross_tab_suicidal_sleep.columns] 
sns.heatmap(
    cross_tab_suicidal_sleep,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=0,
    vmax=0.9
)
plt.title("Depression Rate by Sleep Duration and Suicidal Thoughts")
plt.xlabel("Had Suicidal Thoughts")            
plt.ylabel("Sleep Duration")                  
plt.xticks(rotation=0)          
plt.show()

# 学业压力 + 工作/学习时长 + 学位（分面热力图）
degree_labels = le_dict['Degree'].classes_
plt.figure(figsize=(20, 12))  
for i, degree in enumerate(degree_labels, 1):
    plt.subplot(3, 2, i)
    degree_code = le_dict['Degree'].transform([degree])[0] #排除Academic Pressure=0的无效数据
    mask = (df['Degree'] == degree_code) & (df['Academic Pressure'] != 0)  
    cross_tab = pd.crosstab(
        index=df[mask]['Academic Pressure'],
        columns=df[mask]['Work/Study Hours'],
        values=df[mask]['Depression'],
        aggfunc='mean'
    )
    academic_labels = le_dict['Academic Pressure'].classes_[1:]  
    work_study_labels = le_dict['Work/Study Hours'].classes_
    cross_tab.index = [academic_labels[i-1] for i in cross_tab.index] 
    cross_tab.columns = [work_study_labels[i] for i in cross_tab.columns]
    sns.heatmap(cross_tab,
               annot=True,
               fmt=".2f",
               cmap="coolwarm",
               vmin=0,
               vmax=1)
    plt.title(f"Degree: {degree}", pad=12)
    plt.xlabel("Work/Study Hours", labelpad=10)
    plt.ylabel("Academic Pressure", labelpad=10)
plt.suptitle("Depression Rate by Academic Pressure and Work/Study Hours Across Degrees", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=5, w_pad=2)  
plt.show()

