#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style ='whitegrid')
df = pd.read_excel(r"C:\Users\ngozi\OneDrive\Desktop\Group 2 Capstone Project\Data Science_nurse_attrition.xlsx")
df.head()


# **Creating a New Column - Attrition Binary**
# 
# 
# - This will enable aggregation attrition by department
# - Running machine learning models
# - Correlating with other numeric features

# In[14]:


#Creating new field Attrition_Binary (1 = Employee left; 0 = Employee stayed)

df['Attrition_Binary'] = df['Attrition'].map({'Yes': 1, 'No': 0})
df.head()


# **Descriptive Statistics and Profiling**

# In[15]:


print(df.shape)
print(df.isnull().sum())
print(df.describe(include='all'))


# In[16]:


# Calculate overall attrition rate
total_employees = len(df)
total_attrition = df['Attrition'].value_counts().get('Yes', 0)
attrition_rate = (total_attrition / total_employees) * 100

print(f"Overall Attrition Rate: {attrition_rate:.2f}%")


# **Visualizations**

# In[17]:


#Visualize Using Pie Chart

import matplotlib.pyplot as plt
labels = ['Stayed', 'Left']
sizes = [df['Attrition'].value_counts()['No'], df['Attrition'].value_counts()['Yes']]
colors = ['#1f77b4', '#ff7f0e']
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Overall Nurse Attrition Rate')
plt.axis('equal')  # Equal aspect ratio ensures the pie is a circle
plt.show()


# In[12]:


#Attrition by Department

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dept_attr = df.groupby('Department')['Attrition_Binary'].mean().reset_index()
plt.figure(figsize=(8, 5))
ax = sns.barplot(
    data=dept_attr,
    x='Department',
    y='Attrition_Binary',
    hue='Department',
    palette='Set2',
    dodge=False,
    legend=False
)
for container in ax.containers:
    ax.bar_label(container, labels=[f"{v * 100:.1f}%" for v in container.datavalues], padding=3)
plt.title('Attrition Rate by Department')
plt.ylabel('Attrition Rate (%)')
plt.xlabel('Department')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[18]:


#Attrition Rate by Years at Company

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

years_attr = df.groupby('YearsAtCompany')['Attrition_Binary'].mean().reset_index()
plt.figure(figsize=(10, 5))
ax = sns.lineplot(
    data=years_attr,
    x='YearsAtCompany',
    y='Attrition_Binary',
    marker='o',
    linewidth=2.5,
    color='#1f77b4'
)
ax.set_yticks([i / 10 for i in range(0, 11)])
ax.set_yticklabels([f'{i * 10}%' for i in range(0, 11)])
plt.title('Attrition Rate by Years at Company')
plt.xlabel('Years at Company')
plt.ylabel('Attrition Rate (%)')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[17]:


#Attrition by Age Group

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Defining age group bins and labels
bins = [18, 25, 35, 45, 55, 65]
labels = ['18–25', '26–35', '36–45', '46–55', '56–65']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Calculating attrition rate by age group
age_attr = df.groupby('AgeGroup', observed=True)['Attrition_Binary'].mean().reset_index()
plt.figure(figsize=(8, 5))
ax = sns.barplot(
    data=age_attr,
    x='AgeGroup',
    y='Attrition_Binary',
    hue='AgeGroup',
    palette='coolwarm',
    dodge=False,
    legend=False
)
for container in ax.containers:
    ax.bar_label(container, labels=[f"{v * 100:.1f}%" for v in container.datavalues], padding=3)
plt.title('Attrition Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Attrition Rate (%)')
plt.tight_layout()
plt.show()


# In[19]:


#Attrition Rate by OverTime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Calculating attrition rate by OverTime status
overtime_attr = df.groupby('OverTime', observed=True)['Attrition_Binary'].mean().reset_index()
plt.figure(figsize=(6, 5))
ax = sns.barplot(
    data=overtime_attr,
    x='OverTime',
    y='Attrition_Binary',
    hue='OverTime',           
    palette='Set1',
    dodge=False
)
# Add percentage labels on top of bars
for container in ax.containers:
    ax.bar_label(container, labels=[f"{v * 100:.1f}%" for v in container.datavalues], padding=3)
plt.title('Attrition Rate by Overtime Status')
plt.xlabel('OverTime')
plt.ylabel('Attrition Rate (%)')
plt.tight_layout()
plt.show()


# In[32]:


#Attrition by Monthly Income Bracket

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define income bins and labels
income_bins = [0, 3000, 5000, 7000, 9000, 11000, df['MonthlyIncome'].max()]
income_labels = ['<3k', '3k–5k', '5k–7k', '7k–9k', '9k–11k', '11k+']

# Create income brackets safely (string-based categories)
df['IncomeBracket'] = pd.cut(df['MonthlyIncome'], bins=income_bins, labels=income_labels, right=False)

# Grouping by IncomeBracket 
income_attr = df.groupby('IncomeBracket', observed=True)['Attrition_Binary'].mean().reset_index()
plt.figure(figsize=(8, 5))
ax = sns.barplot(
    data=income_attr,
    x='IncomeBracket',
    y='Attrition_Binary',
    hue='IncomeBracket',
    palette='coolwarm',
    dodge=False
)
for container in ax.containers:
    ax.bar_label(container, labels=[f"{val * 100:.1f}%" for val in container.datavalues], padding=3)
plt.title('Attrition Rate by Monthly Income Bracket')
plt.xlabel('Monthly Income Bracket ($)')
plt.ylabel('Attrition Rate (%)')
plt.tight_layout()
plt.show()


# In[35]:


#Attrition by WorkLifeBalance
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Mapping WorkLifeBalance scale to more readable labels
wlb_map = {1: 'Poor', 2: 'Fair', 3: 'Good', 4: 'Excellent'}
df['WorkLifeBalanceLabel'] = df['WorkLifeBalance'].map(wlb_map)

# Calculating attrition rate by WorkLifeBalance level
wlb_attr = df.groupby('WorkLifeBalanceLabel', observed=True)['Attrition_Binary'].mean().reset_index()
plt.figure(figsize=(7, 5))
ax = sns.barplot(
    data=wlb_attr,
    x='WorkLifeBalanceLabel',
    y='Attrition_Binary',
    hue='WorkLifeBalanceLabel',
    palette='Spectral',
    dodge=False
)
for container in ax.containers:
    ax.bar_label(container, labels=[f"{v * 100:.1f}%" for v in container.datavalues], padding=3)
plt.title('Attrition Rate by Work-Life Balance')
plt.xlabel('Work-Life Balance Rating')
plt.ylabel('Attrition Rate (%)')
plt.tight_layout()
plt.show()


# In[20]:


#Correlation - Heatmap

corr = df.corr(numeric_only=True)
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap of Numeric Features')
plt.show()


# **Machine Learning Model - Random Forest**

# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Making a clean copy of the dataframe
model_df = df.copy()

# Dropping irrelevant or non-numeric columns (e.g., ID and binned ranges)
model_df.drop(columns=[
    'EmployeeID',
    'Attrition',
    'AgeGroup',         
    'IncomeBracket'      
], errors='ignore', inplace=True)

# Encode categorical variables
cat_cols = model_df.select_dtypes(include='object').columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col])
    label_encoders[col] = le

# Prepare features and target
X = model_df.drop(columns='Attrition_Binary')
y = model_df['Attrition_Binary']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluate
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# **Feature Importance**

# In[29]:


importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances.head(15))
importances.head(15).plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Top 15 Feature Importances')
plt.xlabel('Importance')
plt.show()


# In[ ]:




