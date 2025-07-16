import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_excel("dataset2_v2.xlsx")

# Drop irrelevant columns
df.drop(['customerid', 'count', 'country', 'state', 'city', 'zipcode', 'latlong',
         'latitude', 'longitude', 'churnlabel', 'churnscore', 'cltv', 'churnreason'],
        axis=1, inplace=True, errors='ignore')

# Drop missing values
df.dropna(inplace=True)

# Encode categorical features using LabelEncoder
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split into features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)

# Combine back into a DataFrame
df_balanced = pd.concat([pd.DataFrame(X_bal, columns=X.columns), pd.Series(y_bal, name='Churn')], axis=1)

# Split into train and test sets
train_df, test_df = train_test_split(df_balanced, test_size=0.2, random_state=42, stratify=df_balanced['Churn'])

# Save to CSV
train_df.to_csv("balanced_train.csv", index=False)
test_df.to_csv("balanced_test.csv", index=False)

print("✅ SMOTE applied, and dataset split successfully!")
