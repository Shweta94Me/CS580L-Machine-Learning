import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
from sklearn.metrics import accuracy_score

df=pd.read_csv('dataset/train.csv')
#sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
#print(df.shape)

#Dropping the the columns from the dataset were there are more than 60% of missing data
rowcountindf = len(df)
for fields in df:
    cntfieldnullval = df[fields].isnull().sum()
    if(cntfieldnullval > 0):
        nullfieldperc = (cntfieldnullval/rowcountindf)*100
        if(nullfieldperc > 60):
            df.drop([fields],axis=1,inplace=True)

#Aprint(df.shape)
            
df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
#df['GarageYrBlt']=df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean())

#Taking Mode for categorical data
df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])
df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])

#Dropping GarageYrBlt and ID coulmns are they have less number of data and the unique Id of the dataset will be of no use.
df.drop(['GarageYrBlt'],axis=1,inplace=True)
df.drop(['Id'],axis=1,inplace=True)

#print(df.shape)

#When very less number of null values are left i have dropped few records
df.dropna(inplace=True)

#print(df.shape)

#Handling Categorical Feature by identifing categorical feauture form the data set 
columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']

#Method to handle categorical feature to convert them to dummy numeric values so that the prediction algorithm could understand the same
def category_onehot_multcols(multcolumns):
    df_final = final_df
    i=0
    for fields in multcolumns:
        
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final

#create a copy of train data 
main_df=df.copy()

#read test data
test_df=pd.read_csv('dataset/formulatedtest.csv')
#print(test_df.head())

final_df=pd.concat([df,test_df],axis=0,sort=False)


final_df=category_onehot_multcols(columns)
print(final_df.shape)

#print(final_df)


final_df =final_df.loc[:,~final_df.columns.duplicated()]
print(final_df.shape)

df_Train= final_df.iloc[:1422,:]
df_Test= final_df.iloc[1422:,:]

df_Test.drop(['SalePrice'],axis=1,inplace=True)
X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']

print(X_train.shape)

#XGB Boost Classifier
classifer = xgboost.XGBRegressor()
classifer.fit(X_train,y_train)

y_pred = classifer.predict(df_Test)

pred=pd.DataFrame(y_pred)
sub_df=pd.read_csv('dataset/sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('sample_submission.csv',index=False)
#print(y_pred)

#predictions = [round(value) for value in y_pred]
## evaluate predictions
#accuracy = accuracy_score(y_test, predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

