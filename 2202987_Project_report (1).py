#!/usr/bin/env python
# coding: utf-8

# # Importing dataset and necessary libraries

# In[319]:


import pandas as pd   
import numpy as np       
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import sklearn.linear_model as skl_lm
import math
import matplotlib as mpl
import graphviz

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# Ensure inline plotting in Jupyter Notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Set Seaborn style directly
sns.set_style('whitegrid')


# In[17]:


# importing the Absenteeism dataset
df = pd.read_csv('Absenteeism_at_work.csv', sep = ';')


# # Understanding the data

# In[20]:


# inspecting variable type
df.info()


# In[24]:


# looking at the descriptive statistics
# minimum value of 'Month of absence' and 'Reason of absence' is 0. There entries
df.describe()


# ## Preparing the dataset

# In[27]:


# rename variables to avoid future problems in running the code
df = df.rename(columns={'Reason for absence':'reason_absence',
                        'Month of absence':'month',
                        'Day of the week':'day_of_week',
                        'Seasons':'season',
                        'Transportation expense':'transpo_exp',
                        'Distance from Residence to Work':'home_work_km',
                        'Service time':'work_hours',
                        'Work load Average/day ':'work_avg',
                        'Disciplinary failure':'discipline',
                        'Education':'education',
                        'Son':'no_children',
                        'Social drinker':'social_drinker',
                        'Social smoker':'social_smoker',
                        'Body mass index':'bmi',
                        'Absenteeism time in hours':'hours_absent'})


# In[33]:


## visualising values in categorical variables


# In[49]:


sns.countplot(x='day_of_week', data=df)
plt.title('on which days workers are absent')
plt.savefig('count_plot.png')
plt.show()

sns.countplot(x='season', data=df)
plt.title('on which season workers are absent')
plt.savefig('count_1.png')
plt.show()

sns.countplot(x='discipline', data=df)
plt.title('if they ever received disciplinary action')
plt.savefig('count_2.png')
plt.show()

sns.countplot(x='education', data=df)
plt.title('education level')
plt.savefig('count_3.png')
plt.show()

sns.countplot(x='no_children', data=df)
plt.title('number of children they have')
plt.savefig('count_4.png')
plt.show()

sns.countplot(x='social_drinker', data=df)
plt.title('if they drink')
plt.savefig('count_5.png')
plt.show()

sns.countplot(x='social_smoker', data=df)
plt.title('if they smoke')
plt.savefig('count_6.png')
plt.show()


# In[104]:


# inspecting the values
print(df['reason_absence'].value_counts())
print()
print(df['day_of_week'].value_counts())
print()
print(df['season'].value_counts())
print()
print(df['discipline'].value_counts())
print()
print(df['education'].value_counts())
print()
print(df['no_children'].value_counts())
print()
print(df['social_drinker'].value_counts())
print()
print(df['social_smoker'].value_counts())


# In[51]:


# showing entries zero values for 'reason_absence', 'month', and 'hours_absent' 
# Filter rows where 'Reason for absence' is 0
reason_absence_zero = df[df['reason_absence'] == 0]
print(reason_absence_zero)

# Filter rows where 'Month of absence' is 0
month_absence_zero = df[df['month'] == 0]
print(month_absence_zero)

# Filter rows where 'Absenteeism time in hours' is 0
absenteeism_time_zero = df[df['hours_absent'] == 0]
print(absenteeism_time_zero)


# ## Bivariate Analysis

# In[61]:


col_for_regression = ['transpo_exp', 'home_work_km', 'work_hours', 'work_avg', 'no_children', 'Pet', 'hours_absent']
df_4 = df[col_for_regression]
df_4.describe()


# ## regression

# In[85]:


sns.regplot(x=df_4.Pet, y=df_4.hours_absent, ci=None)



# In[94]:


sns.regplot(x=df_4.no_children, y=df_4.hours_absent, ci=None)



# In[96]:


sns.regplot(x=df_4.work_avg, y=df_4.hours_absent, ci=None)



# In[98]:


sns.regplot(x=df_4.work_hours, y=df_4.hours_absent, ci=None)


# In[100]:


sns.regplot(x=df_4.home_work_km, y=df_4.hours_absent, ci=None)



# In[102]:


sns.regplot(x=df_4.transpo_exp, y=df_4.hours_absent, ci=None)


# In[108]:


plt.scatter(x=df_4.no_children, y=df_4.hours_absent, facecolors='None', edgecolors='k', alpha=.8) 
sns.regplot(x=df_4.no_children, y=df_4.hours_absent, ci=None, label='Linear', scatter=False, color='orange') # ci captures the relevant area of data
sns.regplot(x=df_4.no_children, y=df_4.hours_absent, ci=None, label='Degree x', order=5, scatter=False, color='g') #order picks up noises
plt.legend()
plt.savefig('regression2.png')


# In[112]:


# fitting a model
import statsmodels.formula.api as smf

hours_absent = smf.ols('hours_absent ~ no_children', df_4).fit()
hours_absent.summary()


# In[114]:


hours_absent1 = smf.ols('hours_absent ~ no_children + work_avg + work_hours', df_4).fit()
hours_absent1.summary()


# In[116]:


# correlation matrix
df_4.corr()


# In[122]:


df_4.describe()


# In[118]:


X = df_4[['transpo_exp', 'work_avg', 'no_children']]
y = df_4['hours_absent']
std_reg = LinearRegression().fit(X, y)
print(std_reg.coef_)


# ## Logistic regression as inference

# In[129]:


formula = 'hours_absent ~ transpo_exp+home_work_km+work_hours+work_avg+no_children+Pet'
model = smf.glm(formula = formula, data=df, family=sm.families.Binomial())
result = model.fit()
print(result.summary())


# # Resampling

# In[134]:


df_7 = df[['transpo_exp', 'home_work_km', 'work_hours', 'work_avg', 'no_children', 'Pet', 'hours_absent']]


# In[141]:


corr = df_7.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap='Greens', annot=True)
plt.show()
plt.savefig('heatmap.png')


# ## validation set approach
# or 2-fold cross validation, splitting the data into training and testing

# In[147]:


train_df = df.sample(370, random_state = 1)
test_df = df[~df.isin(train_df)].dropna(how ='all')
 
X_train = train_df['no_children'].values.reshape(-1,1)
y_train = train_df['hours_absent']
X_test = test_df['no_children'].values.reshape(-1,1)
y_test = test_df['hours_absent']
 
# fit a linear regression
lm = LinearRegression()
model = lm.fit(X_train, y_train)


# In[149]:


# predicting on the test data set
pred = model.predict(X_test)


# In[178]:


MSE = mean_squared_error(y_test, pred) # comparing prediction to what is actually oobserved in y_test
MSE


# In[180]:


math.sqrt(MSE)


# ## Leave one out cross validation (for 740 observations)

# In[158]:


loo = LeaveOneOut()

# Reshape the input features and target variable
X = df_7['no_children'].values.reshape(-1, 1)
y = df_7['hours_absent'].values.reshape(-1, 1)

# Now use X and y in your scikit-learn model or cross-validation


# In[160]:


loo.get_n_splits(X)


# In[167]:


crossvalidation = KFold (n_splits = 740, random_state = None, shuffle = False) # each value are once used for prediction
scores = cross_val_score(model, X, y, scoring = "neg_mean_squared_error", cv = crossvalidation, n_jobs = 1)


# In[169]:


print("Folds: " + str(len(scores)) + ", MSE: " + str(np.mean(np.abs(scores))) + ", STD: " + str(np.std(scores)))


# ## K - Fold cross validation

# In[174]:


crossvalidation = KFold(n_splits = 10, random_state = None, shuffle = False)

for i in range (1, 11):
    poly = PolynomialFeatures (degree = i)
    X_current = poly.fit_transform(X)
    model = lm.fit(X_current, y)
    scores = cross_val_score(
        model,
        X_current,
        y,
        scoring = 'neg_mean_squared_error',
        cv = crossvalidation,
        n_jobs = 1)
    
    print("Degree-"+str(i)+" polynomial MSE: " + str(np.mean(np.abs(scores))) + ", STD: " + str(np.std(scores)))


# ## Bootstrapping

# In[201]:


df_82 = df[['transpo_exp', 'home_work_km', 'work_hours', 'work_avg', 'no_children', 'Pet', 'hours_absent']]


# In[224]:


# running a conventional linear regression
lm = skl_lm.LinearRegression()
X = df_82['no_children'].values.reshape(-1,1)
y = df_82['hours_absent']
clf = lm.fit(X,y)
print(clf.intercept_, clf.coef_)


# In[232]:


# using bootstrap to look at the estimates
Xsamp, ysamp = resample(X, y, n_samples = 1000)
clf = lm.fit(Xsamp, ysamp)
print(clf.intercept_, clf.coef_)


# # Simple Linear Models for prediction

# In[237]:


df9 = pd.read_csv('Absenteeism_at_work.csv', sep = ';')

df9 = df9.rename(columns={'Reason for absence':'reason_absence',
                        'Month of absence':'month',
                        'Day of the week':'day_of_week',
                        'Seasons':'season',
                        'Transportation expense':'transpo_exp',
                        'Distance from Residence to Work':'home_work_km',
                        'Service time':'work_hours',
                        'Work load Average/day ':'work_avg',
                        'Disciplinary failure':'discipline',
                        'Education':'education',
                        'Son':'no_children',
                        'Social drinker':'social_drinker',
                        'Social smoker':'social_smoker',
                        'Body mass index':'bmi',
                        'Absenteeism time in hours':'hours_absent'})


# In[239]:


# Define X and y. 
# found variables that has 1 and 0 are discipline, social_drinker, social smoker
y = df9.hours_absent
X = df9.drop(['ID', 'hours_absent'], axis = 1)


# ## Lasso

# In[263]:


alphas = 10**np.linspace(10,-2,100)*0.5 
# linspace(10,-2,100)        generates a sequence of 100 numbers evenly spaced between 10 and -2
# 10**np.linspace(10,-2,100) using the exponentiation operator **: raises 10 to the power of each value in the sequence
#10**np.linspace(10,-2,100)*0.5 multiplies each element of the array obtained in step 2 by 0.5. 
# This scales down the values by half.

alphas


# In[273]:


# Assuming you have your input features stored in X and target variable in y
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[275]:


# Split data into training and test sets
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)


# In[277]:


# Convert X into a DataFrame for the coefficients to fill
X_df = pd.DataFrame(X, columns=['reason_absence', 'month', 'day_of_week', 'season', 'transpo_exp', 'home_work_km', 
                                'work_hours', 'Age', 'work_avg', 'Hit target', 'discipline', 'education', 'no_children', 
                                'social_drinker', 'social_smoker', 'Pet', 'Weight', 'Height', 'bmi'])  


# In[286]:


lasso = Lasso(max_iter = 10000)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(scale(X_train), y_train)
    coefs.append(lasso.coef_)
    
ax = plt.gca()
ax.plot(alphas*2, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.savefig('lasso.png')


# In[282]:


lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000)
lassocv.fit(X_train, y_train)

lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(X_train, y_train)
mean_squared_error(y_test, lasso.predict(X_test))


# In[284]:


print(pd.Series(lasso.coef_, index=X_df.columns))  # Print coefficients


# # Non - linear models

# ## Random forests

# In[294]:


df11 = pd.read_csv('Absenteeism_at_work.csv', sep = ';')
df11 = df11.drop(columns=['ID', 'Weight', 'Height'])
df11 = df11.rename(columns={'Reason for absence':'reason_absence',
                        'Month of absence':'month',
                        'Day of the week':'day_of_week',
                        'Seasons':'season',
                        'Transportation expense':'transpo_exp',
                        'Distance from Residence to Work':'home_work_km',
                        'Service time':'work_hours',
                        'Work load Average/day ':'work_avg',
                        'Disciplinary failure':'discipline',
                        'Education':'education',
                        'Son':'no_children',
                        'Social drinker':'social_drinker',
                        'Social smoker':'social_smoker',
                        'Body mass index':'bmi',
                        'Absenteeism time in hours':'hours_absent'})


# In[298]:


# define a dependent variable and independent variables
y = df11.hours_absent
X = df11.drop('hours_absent', axis = 1)

# split our data set into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)


# In[312]:


# Random forests: using 6 features
random_forest_absent = RandomForestRegressor(max_features = 1, random_state = 1)

random_forest_absent.fit(X_train, y_train)

pred = random_forest_absent.predict(X_test)
mean_squared_error(y_test, pred)


# In[316]:


Importance = pd.DataFrame({'Importance':random_forest_absent.feature_importances_*100}, 
                          index = X.columns)

Importance.sort_values(by = 'Importance', 
                       axis = 0, 
                       ascending = True).plot(kind = 'barh', 
                                              color = 'r', )

plt.xlabel('Variable Importance')
plt.gca().legend_ = None
plt.savefig('random_forest.png')


# ## Boosting

# In[321]:


boosted_absent = GradientBoostingRegressor(n_estimators = 500, 
                                           learning_rate = 0.01, 
                                           max_depth = 4, 
                                           random_state = 1)

boosted_absent.fit(X_train, y_train)


# In[347]:


feature_importance = boosted_absent.feature_importances_*100

rel_imp = pd.Series(feature_importance, 
                    index = X.columns).sort_values(inplace = False)

rel_imp.T.plot(kind = 'barh', 
               color = 'r', )

plt.xlabel('Variable Importance')

plt.gca().legend_ = None
plt.savefig('boosting.png')


# In[341]:


mean_squared_error(y_test, boosted_absent.predict(X_test))


# In[345]:


boosted_absent2 = GradientBoostingRegressor(n_estimators = 500, 
                                            learning_rate = 0.001, 
                                            max_depth = 4, 
                                            random_state = 1)
boosted_absent2.fit(X_train, y_train)

mean_squared_error(y_test, boosted_absent2.predict(X_test))


# In[ ]:




