# Credit Risk Model Building
## Credit Risk Management

### Problem Statement
This dataset contains information about bank customers and their eligibility for loans. The goal is to build a model that detects whether a customer is eligible for a loan or not.

### Data
- Data is present in two different files, both containing the same data about the bank.
- Both files also contain `-99999` values, which are missing values.

### Reading Data
```python
import os
import pandas as pd

file_paths = []
data_folder = "Data"
for filename in os.listdir(data_folder):
    file_paths.append(os.path.join(os.getcwd(), data_folder, filename))
    print(os.path.join(os.getcwd(), data_folder, filename))

df1 = pd.read_excel(file_paths[0])
df2 = pd.read_excel(file_paths[1])
```

### Data Preprocessing
1. **Top records**: Inspect the first few records of the dataset.
2. **Shape of data**: Check the number of rows and columns.
3. **Check columns**: Ensure all columns are loaded correctly.
4. **Merge based on common column**: Use the `PROSPECTID` column to merge the datasets.
5. **Check Datatypes**: Ensure data types are appropriate for analysis.
6. **Null values**: Check for missing values.
7. **Duplicates**: Identify and handle duplicate records.
8. **Unique Values**: Analyze unique values for categorical columns.
9. **Statistical Summary**: Get descriptive statistics for numerical columns.

```python
# Merge df1 and df2
final_df = df1.merge(df2, how='inner', on='PROSPECTID')

# Datatypes
final_df.dtypes

# Handling Missing Values
# Remove columns with more than 10000 -99999 values
columns_to_remove = [col for col in final_df.columns if (final_df[col] == -99999).sum() > 10000]
final_df.drop(columns=columns_to_remove, inplace=True)

# Remove rows with -99999 values
final_df.replace(-99999, pd.NA, inplace=True)
final_df.dropna(inplace=True)
```

### Feature Engineering
1. **Hypothesis Testing**
2. **Variance Inflation Factor (VIF)**
   - Measure multicollinearity in regression analysis.
   - Remove columns with VIF greater than a threshold (e.g., 6).
   
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Apply VIF
num_col = final_df.select_dtypes(include='number').columns
vif_data = final_df[num_col]
col_to_keep = []

for i in range(len(num_col)):
    vif_value = variance_inflation_factor(vif_data.values, i)
    if vif_value <= 6:
        col_to_keep.append(num_col[i])

vif_data = final_df[col_to_keep]
```

3. **ANOVA**
   - Check the relationship between numerical columns and the target column.
   - Drop columns with p-value greater than 0.05.
   
```python
from scipy.stats import f_oneway

# Apply ANOVA
col_to_remain = []
for col in col_to_keep:
    groups = [final_df[col][final_df['Approved_Flag'] == category] for category in final_df['Approved_Flag'].unique()]
    f_stats, p_value = f_oneway(*groups)
    if p_value < 0.05:
        col_to_remain.append(col)

final_df = final_df[col_to_remain + ['Approved_Flag']]
```

4. **Chi-Square Test**
   - Apply on categorical columns.
   - Drop columns with p-value greater than 0.05.

```python
from scipy.stats import chi2_contingency

drop_col = []
cat_col = final_df.select_dtypes(include='object').columns

for col in cat_col:
    contingency_table = pd.crosstab(final_df[col], final_df['Approved_Flag'])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    if p_value > 0.05:
        drop_col.append(col)

final_df.drop(columns=drop_col, inplace=True)
```

### Model Building
1. **Separate Input and Output Columns**
2. **Encode Target Column**
3. **Separate Numerical and Categorical Columns**
4. **Train-Test Split**
5. **Build Pipelines for Numerical and Categorical Columns**
6. **Column Transformer**
7. **Build Final Pipeline for Model**

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV

# Separate Input and Output Columns
features = final_df.drop(columns=['Approved_Flag'])
label = final_df['Approved_Flag']

# Encode Target Column
lb = LabelEncoder()
label = lb.fit_transform(label)

# Separate Numerical and Categorical Columns
num_col = features.select_dtypes(include='number').columns
cat_col = features.select_dtypes(include='object').columns

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=43)

# Numerical Pipeline
num_pipe = Pipeline(steps=[
    ("impute", SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

# Categorical Pipeline
cat_pipe = Pipeline(steps=[
    ("impute", SimpleImputer(strategy='most_frequent')),
    ('Encode', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
])

# Column Transformer
transformer = ColumnTransformer(transformers=[
    ('num_transformer', num_pipe, num_col),
    ("Encode", cat_pipe, cat_col)
], remainder='passthrough')

# Build Final Pipeline for Model
model_dic = {
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "xgboost": XGBClassifier()
}

results = {
    "model_name": [],
    "score": [],
    'train score': [],
    "test score": [],
    'matrix': []
}

for model_name, model in model_dic.items():
    final_pipeline = Pipeline(steps=[
        ("process", transformer),
        ("model", model)
    ])
    
    # Fit the model pipeline
    final_pipeline.fit(x_train, y_train)
    
    # Prediction
    predictions = final_pipeline.predict(x_test)
    
    # Calculate score
    score = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    
    # Cross-validation score
    train_score = cross_val_score(final_pipeline, x_train, y_train, cv=5, scoring='accuracy')
    test_score = cross_val_score(final_pipeline, x_test, y_test, cv=5, scoring='accuracy')
    
    results['model_name'].append(model_name)
    results['score'].append(score)
    results['train score'].append(train_score.mean())
    results['test score'].append(test_score.mean())
    results['matrix'].append(matrix)

# Hyperparameter Tuning for XGBoost
param_grid = {
    'model__model__colsample_bytree': [0.1, 0.3, 0.5, 0.7, 0.9],
    'model__model__learning_rate': [0.001, 0.01, 0.1, 1],
    'model__model__max_depth': [3, 5, 8, 10],
    'model__model__n_estimators': [10, 50, 100]
}

pipeline = Pipeline(steps=[
    ("process", transformer),
    ('model', XGBClassifier())
])

grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=5)
grid_search.fit(x_train, y_train)

# Best parameters
best_params = grid_search.best_params_
print("Best Parameters: ", best_params)

# Predictions with best parameters
predictions = grid_search.predict(x_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: ", accuracy)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix: ", conf_matrix)
```

### Conclusion
- **XGBoost Model**: Achieved an accuracy score of 78% after hyperparameter tuning.
- **Cross Validation**: 
  - Train score: 0.775
  - Test score: 0.767
