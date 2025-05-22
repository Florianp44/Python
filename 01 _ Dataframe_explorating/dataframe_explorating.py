# This project is based on the "heart_disease.csv" dataset.
# It contains medical information and test results for just over 300 patients.
# The target variable indicates the presence (1) or absence (0) of heart disease in a patient.

# The main objective is to use data exploration and preparation methods, along with a classification algorithm,
# to predict whether or not a patient has heart disease.
# In the second part, we optimize the code using advanced Python techniques.

## Dataset Exploration

# Importing Pandas
import pandas as pd

# Import the file "heart_disease.csv" (separator ",")
df = pd.read_csv("heart_disease.csv", sep = ",", header = 0)

# Preview the first 5 rows of the DataFrame
print(df.head())

# Description of the DataFrame
print("\ndf comprend", df.shape[1], "colonnes et", df.shape[0], "lignes")
print("\nLe nom des colonnes :", df.columns)

# Display of data related to patients aged 37
print("\nPatients de 37 ans :\n", df.loc[df["age"] == 37])

# Display of the oldest patient's data
print("\nPatients le plus âgé :\n", df.loc[df["age"] == df["age"].max()])

# Mean of DataFrame columns grouped by gender
df_groupby_sex = df.groupby("sex").mean()
print(df_groupby_sex)

# Display the proportion of patients with heart disease among men and women
print("\nLa proportion de femmes malades parmis le panel de femme est de :", df_groupby_sex["target"][0]*100,"%")
print("\nLa proportion d'hommes malades parmis le panel des hommes est de :", df_groupby_sex["target"][1]*100,"%")

# Mean of DataFrame columns based on the presence or absence of heart disease (target column)
df_groupby_target = df.groupby("target").mean()
print(df_groupby_target)

# Display the average age between healthy and sick individuals
print("\nL'âge moyen des individus sains est de :", df_groupby_target["age"][0].round(1),"ans")
print("\nL'âge moyen des individus malades est de :", df_groupby_target["age"][1].round(1),"ans")

### Data Preparation

# Replacement of the 'Male' and 'Female' categories in the 'sex' column with 0 and 1

# Creation of new values
new_values = {"Male" : 0,
            "Female" : 1}

# Replacement of the Male/Female categories with 0/1
df["sex"] = df["sex"].replace(new_values)

print(df)


# The variable thalach corresponds to the maximum heart rate reached by the patient.
# The maximum heart rate is generally between 50 and 250 beats per minute.
# Sometimes, some values fall outside this expected range due to human error (e.g., a misplaced comma); 
# these are called outliers.

# Identification of outliers (i.e., values outside the range [50, 250]) in the thalach column, and if any, replacement with a plausible value.

# Creation of df_thalach_false, filtered on "thalach" outliers (not in the range 50 to 250)
df_thalach_true = df.loc[(df["thalach"] >= 50) & (df["thalach"] <= 250)]

# Calculation of the mean "thalach" value excluding outliers
mean_thalach = df_thalach_true["thalach"].mean().round(0)

# Replacement of "thalach" outliers with the mean of the correct values
df["thalach"] = df["thalach"].apply(lambda x:mean_thalach if x<50 or x>250 else x)

print(df)

# Display of the number of missing values per column
print(df.isna().sum(axis = 0))

# Deletion of rows with missing values in the "target" column
df = df.dropna(axis = 0, how = "all", subset = ["target"])

print("Vérification nombre de manquant en colonne target :",df["target"].isna().sum(axis = 0))

# Verification that the "target" column has 0 missing values

# Some variables like 'ca' or 'exang' are numeric, but qualitative. Indeed, they contain a finite number of categories (four for the first, two for the second).

# - Replacement of missing values in the ca and exang columns with their respective modes

# Finding the most frequent values (modes) in the ca and exang columns
print("modalité la plus fréquente de la colonne ca :", df["ca"].mode()[0])
print("modalité la plus fréquente de la colonne exang :", df["exang"].mode()[0])

# Replacement of missing values with the most frequent ones in the ca and exang columns
df["ca"] = df["ca"].fillna(df["ca"].mode()[0])
df["exang"] = df["exang"].fillna(df["exang"].mode()[0])

print("\nVérification nombre de manquant en colonne ca :",df["ca"].isna().sum(axis = 0))
print("Vérification nombre de manquant en colonne exang :",df["exang"].isna().sum(axis = 0))

# Replacement of missing values in other columns with their respective medians
df["trestbps"] = df["trestbps"].fillna(df["trestbps"].median())
df["chol"] = df["chol"].fillna(df["chol"].median())
df["thalach"] = df["thalach"].fillna(df["thalach"].median())

# Verifying that there are no more missing values in the DataFrame
print(df.isna().sum(axis = 0))

# - Separation of the explanatory variables from df into a new DataFrame X, and the target variable into a Series y

# Creation of df_x (explanatory variables) and serie_y (target column)
df_x = df[df.columns[:-1]]
serie_y = df[df.columns[-1:]]

print("df_x :\n",df_x)
print("\nserie_y :\n",serie_y)

# The 13 explanatory variables we have vary in different ranges, which can cause issues for certain classification algorithms. For example, the variable oldpeak contains values between 0 and 6.2, while the variable trestbps is in the range [94,200].
# To transform these variables so that they are all within the fixed range [-1,1], we can apply a slightly modified Min-Max normalization.
# The normalization we want to apply uses the following formula:
# > $$X_{new} = 2\frac{X - X_{min}}{X_{max} - X_{min}} - 1$$
# Where:
#   -  $X_{min}$: the minimum observed value for variable $X$
#   -  $X_{max}$: the maximum observed value for variable $X$
#   -  $X$: The value of the variable we want to normalize


# - Apply this transformation to each column of X, and store the result in a new DataFrame called X_norm

# Create the normalization function to apply to all columns of df_x
def normalisation(df_x):
    return 2 * ((df_x - df_x.min()) / (df_x.max() - df_x.min())) - 1

X_norm = normalisation(df_x)

print(X_norm)

# ### Code Optimization with Advanced Python

# > Up to this point, we have implemented a series of data exploration and preprocessing techniques, as well as a classification algorithm to predict heart disease. 
# The code has been executed sequentially, line by line. 
# Now, let’s explore how we can optimize this process by encapsulating our code into functions and using advanced Python techniques for greater efficiency, code readability, and reusability

# Importing the "heart_disease.csv" file (separator: ",")
df = pd.read_csv("heart_disease.csv", sep = ",", header = 0)

# Preview of the first 5 rows of the DataFrame
print(df.head())


# - Implement a decorator that ensures the input arguments of a function and the returned object are all DataFrames.
#   To do this, we can use Python's isinstance() method, which checks if an object is an instance or subclass of a specific class. It returns True if so, and False otherwise

# - Test the decorator on a function called preprocess_data(df), which summarizes all preprocessing steps from the sequential code. The method should:

# > - Replace the 'Male' and 'Female' categories in the 'sex' column with 0 and 1
# > - Replace outlier values in the 'thalach' column with a plausible value
# > - Remove rows from the DataFrame that are not labeled (target is missing)
# > - Replace missing values in the ca and exang columns with their modes
# > - Replace missing values in the other columns with their respective medians

# Create the decorator that checks input arguments and return values are DataFrames
def check_entree_sorties(function):
      def check(*args, **kwargs):
            # Check that each argument is a DataFrame
            for arg in args:
                  if not isinstance(arg, pd.DataFrame):
                        raise TypeError(f"L'argument en entrée {arg} n'est pas un dataframe")     
            for key, value in kwargs.items():
                  if not isinstance(value, pd.DataFrame):
                        raise TypeError("L'argument en entrée '{key}'' n'est pas un dataframe")
            result = function(*args, **kwargs)
      
            # Check that the result is a DataFrame
            if not isinstance(result, pd.DataFrame):
                  raise TypeError("Le résultat retourné n'est pas un dataframe")
            
            return result
      return check

# Display functions used
def affiche_doc(function):
      def print_doc(*args, **kwargs):
            print(function.__doc__)
            return function(*args, **kwargs)
      return print_doc

@check_entree_sorties
@affiche_doc
def preprocess_data(df):
      '''
      Operations performed on df:
      1. Replacement of 'Male'/'Female' with 0/1  
      2. Replacement of outlier values in 'thalach'  
      3. Deletion of rows with missing values in "target"  
      4. Replacement of null values with the most frequent values for 'ca'/'exang'  
      5. Replacement of missing values with respective medians
      '''
      # Replacement of 'Male'/'Female' with 0/1
      df["sex"] = df["sex"].replace({"Male" : 0, "Female" : 1})
      
      # Replacement of outlier values in 'thalach'
      df_thalach_true = df.loc[(df["thalach"] >= 50 ) & (df["thalach"] <= 250)]
      mean_thalach = df_thalach_true["thalach"].mean().round(0)
      df["thalach"] = df["thalach"].apply(lambda x:mean_thalach if x < 50 or x > 250 else x)
      
      # Deletion of rows with missing values in "target"
      df = df.dropna(axis = 0, how = "all", subset = ["target"])
      
      # Replacement of null values with the most frequent values for 'ca'/'exang'
      df["ca"] = df["ca"].fillna(df["ca"].mode()[0])
      df["exang"] = df["exang"].fillna(df["exang"].mode()[0])
      
      # Replacement of missing values with respective medians
      df["trestbps"] = df["trestbps"].fillna(df["trestbps"].median())
      df["chol"] = df["chol"].fillna(df["chol"].median())
      df["thalach"] = df["thalach"].fillna(df["thalach"].median())
      
      # display df
      print(df)
      return df
preprocess_data(df)


# - Create a function normalize_df() using Python type annotations that takes a DataFrame as input and returns a normalized DataFrame. 
#   The normalization should be performed using the min-max normalization as before, adjusting values between -1 and 1 with the following formula:

# > $$X_{new} = 2\frac{X - X_{min}}{X_{max} - X_{min}} - 1$$
# > 
# > Where:
# > 
# > - $X_{min}$ : the minimum value observed for variable $X$.
# > - $X_{max}$ : the maximum value observed for variable $X$.
# > - $X$ : the value to be normalized.

# - Display the function’s type annotations


# Create the normalization function to apply to all columns
def normalize_df(df : pd.DataFrame) -> pd.DataFrame:
      return 2*((df - df.min()) / (df.max() - df.min()))-1

print(normalize_df.__annotations__)
print(normalize_df(df))
