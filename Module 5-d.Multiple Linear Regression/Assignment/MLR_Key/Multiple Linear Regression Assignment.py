'''
# Profit Prediction of Startups

**CRISP-ML(Q) process model describes six phases:**

- Business and Data Understanding
- Data Preparation (Data Engineering)
- Model Building (Machine Learning)
- Model Evaluation and Tunning
- Deployment
- Monitoring and Maintenance


**Objective(s):** Maximize the profits

**Constraints:** Maximize the customer satisfaction

**Success Criteria**

- **Business Success Criteria**: Improve the profits from anywhere between 10% to 20%

- **ML Success Criteria**: RMSE should be less than 0.15

- **Economic Success Criteria**: Second/Used cars sales delars would see an increase in revenues by atleast 20%
'''

# Load the Data and perform EDA and Data Preprocessing

# Importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sidetable
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from feature_engine.outliers import Winsorizer
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from sklearn.model_selection import train_test_split
# import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import joblib
import pickle

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

import sweetviz as sv
# Recursive feature elimination
from sklearn.feature_selection import RFE
from sqlalchemy import create_engine
from urllib.parse import quote 

user_name = 'root'
database = 'mul_reg_db'
your_password = 'Seemscrazy1994#'

engine = create_engine(f'mysql+pymysql://{user_name}:%s@localhost:3306/{database}' % quote(f'{your_password}'))


# Load the offline data into Database to simulate client conditions
data = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 5-d.Multiple Linear Regression\Assignment\MLR_Key\50_Startups.csv")
data.to_sql('data', con = engine, if_exists = 'replace', chunksize = 1000, index= False)


#### Read the Table (data) from MySQL database
from sqlalchemy import text

sql = 'SELECT * FROM data'
sql2 ="show tables"
with engine.begin() as conn:
    query = text("""SELECT * FROM data""")
    df = pd.read_sql_query(query, conn)


#### Descriptive Statistics and Data Distribution
df.describe()

# Missing values check
df.isna().sum()
df.info()
#no missing values found


#EDA using Autoviz
sweet_report = sv.analyze(df)

#Saving results to HTML file
sweet_report.show_html('sweet_report.html')


# Seperating input and output variables 
x = pd.DataFrame(df.iloc[:,:-1])
y = pd.DataFrame(df.iloc[:,-1:])

# Checking for unique values
x["State"].unique()

x["State"].value_counts()

# Build a frequency table using sidetable library
x.stb.freq(["State"])

# Segregating Non-Numeric features
categorical_features = x.select_dtypes(include = ['object']).columns
print(categorical_features)


# Segregating Numeric features
numeric_features = x.select_dtypes(exclude = ['object']).columns
print(numeric_features)

### Imputation to handle missing values 
### MinMaxScaler to convert the magnitude of the columns to a range of 0 to 1
num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'mean')),('scale',MinMaxScaler())])

### Encoding - One Hot Encoder to convert Categorical data to Numeric values
## Encoding Categorical features
encoding_pipeline = Pipeline(steps = [('onehot', OneHotEncoder(sparse_output = False))]) #(sparse_output = Flase) dosent give output as sparse_matrix 

# Creating a transformation of variable with ColumnTransformer()
preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, numeric_features), ('categorical', encoding_pipeline, categorical_features)])

imp_enc_scale = preprocessor.fit(x)

#### Save the imputation model using joblib
joblib.dump(imp_enc_scale, 'imp_enc_scale')
cleandata = pd.DataFrame(imp_enc_scale.transform(x), columns = imp_enc_scale.get_feature_names_out())

cleandata
### Outlier Analysis
# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.

# pandas plot() function with parameters kind = 'box' and subplots = True

cleandata.iloc[:,0:3].plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 
'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''
# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


cleandata.iloc[:, 0:3].columns


#### Outlier analysis: Columns 'months_loan_duration', 'amount', and 'age' are continuous, hence outliers are treated
winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = ['num__R&D Spend', 'num__Administration', 'num__Marketing Spend'])


outlier = winsor.fit(cleandata[['num__R&D Spend', 'num__Administration', 'num__Marketing Spend']])

# Save the winsorizer model 
joblib.dump(outlier, 'winsor')
cleandata[['num__R&D Spend', 'num__Administration', 'num__Marketing Spend']] = outlier.transform(cleandata[['num__R&D Spend', 'num__Administration', 'num__Marketing Spend']])

cleandata
cleandata.iloc[:,0:3].plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 


# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


####################
# Multivariate Analysis
sns.pairplot(df)   # original data

# Correlation Analysis on Original Data
orig_df_cor = df.corr()
orig_df_cor

# Heatmap
dataplot = sns.heatmap(orig_df_cor, annot = True, cmap = "YlGnBu")


# Library to call OLS model
# import statsmodels.api as sm

# Build a vanilla model on full dataset

# By default, statsmodels fits a line passing through the origin, i.e. it 
# doesn't fit an intercept. Hence, you need to use the command 'add_constant' 
# so that it also fits an intercept

P = add_constant(cleandata)
basemodel = sm.OLS(y, P).fit()
basemodel.summary()

# p-values of coefficients found to be insignificant due to colinearity

# Identify the variale with highest colinearity using Variance Inflation factor (VIF)
# Variance Inflation Factor (VIF)
# Assumption: VIF > 10 = colinearity
# VIF on clean Data
vif = pd.Series([variance_inflation_factor(P.values, i) for i in range(P.shape[1])], index = P.columns)
vif
# inf = infinity
# VIF equal to 1 = variables are not correlated. 
# VIF between 1 and 5 = variables are moderately correlated. 
# VIF greater than 5 = variables are highly correlated2.

sm.graphics.influence_plot(basemodel)

cleandata_new = P.drop(cleandata.index[[45, 46, 48, 49]])
y_new = y.drop(y.index[[45, 46, 48, 49]])

# Build model on dataset
basemode3 = sm.OLS(y_new, cleandata_new).fit()
basemode3.summary()


# Splitting data into training and testing data set
X_train, X_test, Y_train, Y_test = train_test_split(cleandata_new, y_new, 
                                                    test_size = 0.2, random_state = 0) 

## Build the best model Model building with out cv
model = sm.OLS(Y_train, X_train).fit()
model.summary()

# Predicting upon X_train
ytrain_pred = model.predict(X_train)
r_squared_train = r2_score(Y_train, ytrain_pred)
r_squared_train

# Train residual values
train_resid  = Y_train.Profit - ytrain_pred
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse


# Predicting upon X_test
y_pred = model.predict(X_test)

# checking the Accurarcy by using r2_score
r_squared = r2_score(Y_test, y_pred)
r_squared

# Test residual values
test_resid  = Y_test.Profit - y_pred
# RMSE value for train data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


## Scores with Cross Validation (cv)
# k-fold CV (using all variables)
lm = LinearRegression()

## Scores with KFold
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)

scores = cross_val_score(lm, X_train, Y_train, scoring = 'r2', cv = folds)
scores   

## Model building with CV and RFE

# step-1: create a cross-validation scheme
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)

# step-2: specify range of hyperparameters to tune
hyper_params = [{'n_features_to_select': list(range(1, 9))}]


# step-3: perform grid search
# 3.1 specify model
# lm = LinearRegression()
lm.fit(X_train, Y_train)

# Recursive feature elimination
rfe = RFE(lm)

# 3.2 call GridSearchCV()
model_cv = GridSearchCV(estimator = rfe, 
                        param_grid = hyper_params, 
                        scoring = 'r2', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score = True)      

# fit the model
model_cv = model_cv.fit(X_train, Y_train)     

X_train.info()

cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results

# plotting cv results
plt.figure(figsize = (16, 6))

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('r-squared')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc = 'upper left')

model_cv.best_params_

cv_lm_grid = model_cv.best_estimator_
cv_lm_grid

## Saving the model into pickle file
pickle.dump(cv_lm_grid, open('profit.pkl', 'wb'))

## Testing
test = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 5-d.Multiple Linear Regression\Assignment\MLR_Key\test_data.csv")
model1 = pickle.load(open('profit.pkl','rb'))
imp_enc_scale = joblib.load('imp_enc_scale')
winsor = joblib.load('winsor')


clean = pd.DataFrame(imp_enc_scale.transform(data), columns = imp_enc_scale.get_feature_names_out())

clean[['num__R&D Spend', 'num__Administration', 'num__Marketing Spend']] = winsor.transform(clean[['num__R&D Spend', 'num__Administration', 'num__Marketing Spend']])

clean = add_constant(clean)

clean.info()

prediction = pd.DataFrame(model1.predict(clean), columns = ['MPG_pred'])
prediction

