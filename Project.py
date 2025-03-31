import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from factor_analyzer import FactorAnalyzer

# Step 1: Definition of the problem
# Population Data
population = {
    "Total": 226,
    "1st Year GE": 52,
    "1st Year GI": 58,
    "1st Year GM": 52,
    "1st Year RST": 64
}

# Step 2: Data Collection
file_path = "C:/Users/DeLL LaTiTuDe/Downloads/Adoption de la technologie Google Classroom dans l'enseignement supérieur durant COVID-19 (réponses).xlsx"
df = pd.read_excel(file_path)

# Step 3: Pretreatment
# 3.1 Conversion
for col in ['Sexe', 'Filiere']:
    df[col] = df[col].astype('category')

# 3.2 Encoding categorical responses into numerical values
response_mapping = {
    "pas du tout d'accord": 1,
    "pas d'accord": 2,
    "neutre": 3,
    "d'accord": 4,
    "tout-à-fait d'accord": 5
}

for col in df.columns[4:24]:
    df[col] = df[col].map(response_mapping).astype(float)

# 3.2 Data Cleaning
# 3.2.1 Outlier Detection
outlier_cols = ['Age']
for col in outlier_cols:
    plt.figure()
    sns.boxplot(df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# 3.2.2 Handling Missing Values
df.fillna(df.mean(), inplace=True)

# 3.3 Normality Test
shapiro_test = stats.shapiro(df['Age'])
print("Shapiro test for Age:", shapiro_test)

# 3.4 Representativity Test
sex_chi2 = stats.chisquare(pd.value_counts(df['Sexe']))
age_chi2 = stats.chisquare(pd.value_counts(df['Age']))
filiere_chi2 = stats.chisquare(pd.value_counts(df['Filiere']))
print("Chi-square test results:", sex_chi2, age_chi2, filiere_chi2)

# 3.5 Reliability Test
from scipy.stats import pearsonr

def cronbach_alpha(items):
    items = np.array(items)
    item_vars = items.var(axis=1, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    n_items = items.shape[0]
    return n_items / (n_items - 1) * (1 - item_vars.sum() / total_var)

alpha_usefulness = cronbach_alpha(df.iloc[:, 4:10].values)
alpha_ease = cronbach_alpha(df.iloc[:, 10:15].values)
alpha_intention = cronbach_alpha(df.iloc[:, 15:19].values)
alpha_external = cronbach_alpha(df.iloc[:, 19:24].values)

print("Cronbach Alpha Scores:")
print(f"Usefulness: {alpha_usefulness}")
print(f"Ease of Use: {alpha_ease}")
print(f"Intention: {alpha_intention}")
print(f"External Variables: {alpha_external}")

# Step 4: Data Analysis
# 4.1 Univariate Descriptive Statistics
print(df.describe())

# 4.2 Visualization
sns.histplot(df['Age'], bins=10, kde=True)
plt.show()

sns.countplot(x='Sexe', data=df)
plt.show()

sns.countplot(x='Filiere', data=df)
plt.show()

# 4.3 Hypothesis Testing
# Hypothesis 1: Perceived ease of use influences perceived usefulness
p_values = []
for i in range(4, 10):
    for j in range(10, 15):
        _, p = stats.chisquare(df.iloc[:, i], df.iloc[:, j])
        p_values.append(p)

print("Mean p-value for ease of use influencing usefulness:", np.mean(p_values))

# Hypothesis 2: Usefulness influences intention to use
p_values = []
for i in range(4, 10):
    for j in range(15, 19):
        _, p = stats.chisquare(df.iloc[:, i], df.iloc[:, j])
        p_values.append(p)

print("Mean p-value for usefulness influencing intention:", np.mean(p_values))

# Factor Analysis
fa = FactorAnalyzer(n_factors=4, rotation='varimax')
fa.fit(df.iloc[:, 4:24])
print("Factor Loadings:")
print(pd.DataFrame(fa.loadings_, index=df.columns[4:24]))
