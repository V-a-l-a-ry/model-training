import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import preprocessing
df = pd.read_csv('SolarPrediction.csv')
df
df.isnull().any()
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = df[['Temperature','Pressure','Humidity','Speed','WindDirection(Degrees)']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns
df.drop('Pressure', axis=1)
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables_1 = df[['Temperature','Humidity','Speed','WindDirection(Degrees)']]
vif_1 = pd.DataFrame()
vif_1['VIF'] = [variance_inflation_factor(variables_1.values, i) for i in range(variables_1.shape[1])]
vif_1['Features'] = variables_1.columns
df.columns.values
cols = ['Radiation', 'Temperature', 
       'Humidity', 'WindDirection(Degrees)', 'Speed', 'Pressure','TimeSunRise',
       'TimeSunSet', 'UNIXTime', 'Data', 'Time']
data_ready = df[cols]
data_ready
data_ready = data_ready.drop(['Pressure','TimeSunRise','TimeSunSet','UNIXTime','Data','Time'], axis=1)
data_ready
data = data_ready.values
X, y = data[:,1:], data[:,0]  # splitting and getting the independent attributes, X and dependent attribute y
X = preprocessing.StandardScaler().fit_transform(X)
X[0:5]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)  # splitting in the ratio 70:30
model = LinearRegression()
model.fit(X_train, y_train)
y_hat = model.predict(X_train)
sns.distplot(y_train-y_hat)
plt.title('Residuals PDF')
plt.show()
model.intercept_
model.coef_
y_pred = model.predict(X_test)
type(X)
X_new = pd.DataFrame(X)
type(X_new)
X_new_renamed = X_new.rename(columns={0:'Temparature',1:'Humidity',2:'WindDirection(Degrees)',3:'Speed'})
X_new_renamed
reg_summary = pd.DataFrame(X_new_renamed.columns.values, columns=['Features'])
reg_summary['Weights/Coeffs'] = model.coef_
reg_summary
new_row = {'Features':'Slope Intercept', 'Weights/Coeffs':model.intercept_}
reg_summary.append(new_row, ignore_index=True)
reg_summary
r2_score(y_test, y_pred)
y_hat_test = model.predict(X_test)
plt.scatter(y_test,y_hat_test, alpha=0.05)
plt.xlabel('y_test',fontsize=15)
plt.ylabel('Y_hat_test', fontsize=15)   #This is X_test predicted
plt.title('Testing the Accuracy', fontsize=20)
plt.show()
