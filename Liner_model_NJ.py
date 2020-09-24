
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np


class NJModel:
    def __init__(self):
        self.param_list = []
        
    def creat_month(self, df):
        '''
        给dataframe创建month的对象
        '''
        if "month" in df.columns:
            return df
        else:
            df['dt'] = df.index
            df["month"] = df["dt"].dt.month
            return df
            
    def fit(self, df: pd.DataFrame):
        data = df.copy()
        
        data = self.creat_month(data)
            
        
#         df["month"] = df["dt"].dt.month
        
        for i in range(1, 13):
            data = df[df['month']==i]
            if len(data)==0 :
                print("month", i, "zero data!")
                continue
            LRModel = LinearRegression()
            array = np.polyfit(np.log(data['vsb']), data['pm2_5'], 1)
            k, b = array[0], array[1] 
            self.param_list.append([k, b])
    def predict(self, data):
        df = data.copy()
        
        df = self.creat_month(df)
        
        df['vsb'] = np.log(df['vsb'])
        
        res = []
        for i, row in df.iterrows():
            lis = self.param_list[int(row['month'])-1]
            data = lis[0]*row['vsb']+lis[1]
            res.append(data)
        return res


if __name__ == "__main__":
    import get_his_data as ghd

    df = ghd.get_train_data()

    # info = 
    print(df.head())