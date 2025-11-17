# =========================================================
#        TITANIC DATA PREPROCESSING (DIRECT FETCH)
# =========================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------
# 1. FETCH DATASET DIRECTLY FROM URL
# ---------------------------------------------------------
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA INFO =====")
print(df.info())

print("\n===== NULL VALUES =====")
print(df.isnull().sum())


# ---------------------------------------------------------
# 2. HANDLE MISSING VALUES
# ---------------------------------------------------------

# Numerical columns → fill with mean
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Categorical columns → fill with mode
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].apply(lambda x: x.fillna(x.mode()[0]))

print("\n===== AFTER FILLING MISSING VALUES =====")
print(df.isnull().sum())


# ---------------------------------------------------------
# 3. ENCODING CATEGORICAL COLUMNS
# ---------------------------------------------------------
df_encoded = df.copy()
le = LabelEncoder()

for col in cat_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

print("\n===== ENCODED DATASET HEAD =====")
print(df_encoded.head())


# ---------------------------------------------------------
# 4. NORMALIZATION / STANDARDIZATION
# ---------------------------------------------------------
scaler = StandardScaler()
df_scaled = df_encoded.copy()

df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])

print("\n===== STANDARDIZED DATASET HEAD =====")
print(df_scaled.head())


# ---------------------------------------------------------
# 5. OUTLIER VISUALIZATION (BOXPLOTS)
# ---------------------------------------------------------
plt.figure(figsize=(15, 8))
df_scaled[num_cols].boxplot()
plt.title("Outlier Detection Using Boxplots")
plt.show()


# ---------------------------------------------------------
# 6. REMOVE OUTLIERS USING IQR METHOD
# ---------------------------------------------------------
df_outliers_removed = df_scaled.copy()

for col in num_cols:
    Q1 = df_outliers_removed[col].quantile(0.25)
    Q3 = df_outliers_removed[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_outliers_removed = df_outliers_removed[
        (df_outliers_removed[col] >= lower) &
        (df_outliers_removed[col] <= upper)
    ]

print("\n===== SHAPE AFTER OUTLIER REMOVAL =====")
print(df_outliers_removed.shape)

print("\n===== FINAL CLEANED DATA HEAD =====")
print(df_outliers_removed.head())

# =========================================================
#                        END
# =========================================================

