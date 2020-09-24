#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression



def model_est_data(model,X_test,Y_test):
    '''
    获得模型评估的结果
    输入：模型和X，Y
    输出：计算得到的R2和RMSE
    '''
    Y_pred = pd.Series(model.predict(X_test),index=X_test.index)
    res = np.polyfit(Y_test, Y_pred, 1)
    slop, intercept = res[0], res[1]
    r2 = r2_score(Y_test,Y_pred)
    rmse = mean_squared_error(Y_test,Y_pred)**0.5
    return r2, rmse, slop, intercept

def model_est_data_day(model,X_test,Y_test):
    '''
    获得按照天聚合的模型评估的结果
    输入：模型和X，Y
    输出：计算得到的R2和RMSE（按照天聚合）
    '''
    Y_pred = pd.Series(model.predict(X_test),index=X_test.index).resample('d').mean().dropna()
    Y_test = Y_test.resample('d').mean().dropna()
    return r2_score(Y_test,Y_pred),mean_squared_error(Y_test,Y_pred)**0.5




def plot_compare(Y_test,Y_pred, show_density=False):
    res = dict()
    res['Mean_measured']=round(np.array(Y_test).mean(),2)
    res['Mean_predicted']=round(np.array(Y_pred).mean(),2)
    res['r2']=round(r2_score(Y_test,Y_pred),3)
    res['RMSE']=round(mean_squared_error(Y_test,Y_pred)**0.5,2)
    res['RMSE/Mean']=round((res['RMSE']/res['Mean_measured']),2)
    res['MAE']=round(mean_absolute_error(Y_test,Y_pred),2)
    res['Explain_Var']=round(explained_variance_score(Y_test,Y_pred),3)
    print(res)
    
    if show_density:
        from scipy.stats import gaussian_kde
        xy = np.vstack([Y_test,Y_pred])
        z = gaussian_kde(xy)(xy)
        rate = 0.01*len(z)/np.quantile(z, 0.75)
        z = z*rate
    
    if not show_density:
        plt.figure(figsize=(5,5),dpi=150)
        plt.scatter(Y_test,Y_pred,linewidth=1, s=5,ls='-',cmap='cividis')
    else:
        plt.figure(figsize=(5.5,5),dpi=150)
        plt.scatter(Y_test,Y_pred,c=z, linewidth=1, s=5,ls='-',cmap='cividis')
        plt.colorbar()
    plt.xlabel('Measured PM2.5(μg/m3)')
    plt.ylabel('Predicted PM2.5(μg/m3)')
    lim = max(max(Y_pred),max(Y_test))
    plt.xlim(-1,lim)
    plt.ylim(-1,lim)
    plt.plot(np.arange(0,lim),np.arange(0,lim),color='black',linestyle=':')
    a,b=np.polyfit(Y_test,Y_pred,1)[0],np.polyfit(Y_test,Y_pred,1)[1]
        
    x=np.arange(0,lim,1)
    y=a*x+b

    plt.plot(x,y,color='red',linestyle='-',alpha=0.7)
    plt.tick_params(labelsize=10)
    step=lim/15
    plt.text(15,lim-step,'R2: {}'.format(res['r2']),fontsize=13)
    plt.text(15,lim-2*step,'RMSE: {}'.format(res['RMSE']),fontsize=13)
    plt.text(15,lim-3*step,'a:{} b:{}'.format(round(a,2),round(b,2)),fontsize=13)
    
    
class Est():
    def __init__(self,model):
        self.model = model
    def fit(self,X_train,Y_train):
        self.model.fit(X_train,Y_train)
    def self_est(self,X_test,Y_test):
        '''
        模型评估
        打印：各种评估的参数
        返回：各种参数的列表'''
        Y_pred=pd.Series(self.model.predict(X_test),index=Y_test.index)
        res = dict()
        res['Mean_measured']=round(Y_test.mean(),2)
        res['Mean_predicted']=round(Y_pred.mean(),2)
        res['r2']=round(r2_score(Y_test,Y_pred),3)
        res['RMSE']=round(mean_squared_error(Y_test,Y_pred)**0.5,2)
        res['RMSE/Mean']=round((res['RMSE']/res['Mean_measured']),2)
        res['MAE']=round(mean_absolute_error(Y_test,Y_pred),2)
        res['Explain_Var']=round(explained_variance_score(Y_test,Y_pred),3)
        return res
    def plot_residual_time(self,X_test,Y_test):
        '''随时间变化的残差图'''
        plt.figure(dpi=80)
        Y_pred=pd.Series(self.model.predict(X_test),index=Y_test.index)
        plt.xlabel('Time')
        plt.ylabel('Measured subtract predicted')
        plt.plot(Y_test.index,[0 for i in range(len(Y_test))],color='red')
        plt.scatter(Y_test.index,(Y_pred-Y_test),s=1,ls='-',color = 'black',cmap='cividis',alpha = 0.8)
    def plot_residual(self,X_test,Y_test):
        '''残差图'''
        Y_pred=pd.Series(self.model.predict(X_test),index=Y_test.index)
#         from scipy.stats import gaussian_kde
#         xy = np.vstack([Y_pred-Y_test,Y_test])
#         z = gaussian_kde(xy)(xy)
        x=np.arange(0,max(Y_test),1)
        y=x*0
        plt.plot(x,y,color='black',linestyle=':')
        plt.xlabel('Measured')
        plt.ylabel('Predicted sub Measured')
        plt.scatter(Y_test,Y_pred-Y_test,s=5,ls='-',cmap='cividis',alpha = 0.8)
    
    def self_est_plot(self,X_test,Y_test):
        Y_pred=pd.Series(self.model.predict(X_test),index=Y_test.index)
        res = self.self_est(X_test,Y_test)
        print(res)
        
#         from scipy.stats import gaussian_kde
#         xy = np.vstack([Y_pred,Y_test])
#         z = gaussian_kde(xy)(xy)*1000
        plt.figure(figsize=(5,5),dpi=80)
        plt.scatter(Y_test,Y_pred,linewidth=1, s=5,ls='-',cmap='cividis')
        plt.xlabel('Measured PM2.5(μg/m3)')
        plt.ylabel('Predicted PM2.5(μg/m3)')
        lim = max(max(Y_pred),max(Y_test))
        plt.xlim(-1,lim)
        plt.ylim(-1,lim)
        plt.plot(np.arange(0,lim),np.arange(0,lim),color='black',linestyle=':')
        a,b=np.polyfit(Y_test,Y_pred,1)[0],np.polyfit(Y_test,Y_pred,1)[1]
        
        x=np.arange(0,lim,1)
        y=a*x+b
        cor=Y_test.corr(Y_pred)

        plt.plot(x,y,color='red',linestyle='-',alpha=0.7)
        plt.tick_params(labelsize=10)
        step=lim/15
        plt.text(15,lim-step,'R2: {}'.format(res['r2']),fontsize=13)
        plt.text(15,lim-2*step,'RMSE: {}'.format(res['RMSE']),fontsize=13)
        plt.text(15,lim-3*step,'a:{} b:{}'.format(round(a,2),round(b,2)),fontsize=13)
    def residual_time_autoCor(self,X_test,Y_test,lag=50):
        '''检查随时间变化的残差图的自相关性'''
        Y_pred=pd.Series(self.model.predict(X_test),index=Y_test.index)
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(Y_pred-Y_test,lags=[i for i in range(lag)])
    def tree_plot_importance(self,col):
        '''绘制重要性图
        输入：X.columns比较推荐'''
        array = self.model.feature_importances_
#         print(array)
        a = pd.DataFrame(array,columns=['imp'])
        a.insert(0,value = col,column='feature')
#         print(a.head())
        a.sort_values('imp',ascending=False,inplace=True)
#         print(a)
        plt.figure(figsize=(len(col),5),dpi=80)
        plt.bar(a['feature'],a['imp'],color='crimson')
        plt.tick_params(labelsize=13)
        plt.title('Feature Importance',fontsize=15)
        plt.xlabel('Features',fontsize=15)
        plt.ylabel('Importance',fontsize=15)
        return a

# In[ ]:




