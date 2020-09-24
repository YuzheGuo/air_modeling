import sys
sys.path.append(r'D:\大学\大三下\大创-机器学习\#WorkSpace_after_May')
from mypkg import db_connect as db,model_est_v2 as es,preprocessing as pr
from mypkg import useful_tools as ut
import os
import pymysql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl


def get_his_data_city(cityName):
    '''
    根据城市名称获得his的dataframe
    '''
    info = db.find_location_city(cityName)
    return get_his_data(info[-1])

def get_his_data(stationCode: int)-> pd.DataFrame:
    '''
    获得存储在本地的his数据，但是仍然需要进一步加工
    '''
    path = r"D:\大学\大三下\大创-机器学习\#WorkSpace_after_May\data\hisData_132"
    fileList = os.listdir(path)
    find_str = "{}.csv".format(stationCode)
    
    for name in fileList:
        if name==find_str:
            fileName = name
            break
    df = pd.read_csv(path+"\\"+fileName, index_col=0)
    df.index = pd.DatetimeIndex(df.index)
    return df

def get_predicted_data(stationCode, startYear=1956, endYear=2019):
    '''
    从database中获得data，输入stationCode，获得相应的his data，还可以根据startYear以及endYear来选取
    '''
    sql = 'select * from weather_data_add_locationinfo where code = {}'.format(stationCode)
   
    config1={
            "host":"rm-uf622639ux3c1wwjiio.mysql.rds.aliyuncs.com",
            "user":"airdb_admin",
            "password":"2019@sjtu",
            "database":"互联网比赛数据"}
    
    db_weather = pymysql.connect(**config1)
    cur1 = db_weather.cursor()
    cur1.execute(sql)
    col = [i[0] for i in cur1.description]
    cur1.close()
    
    data = pd.DataFrame(cur1.fetchall(), columns=col)
   
    data.index = data["datetime"]
    data = data.drop(columns=["datetime"])
    data = data.astype("float64")
    return data

def check_vsb(df: pd.DataFrame)-> int:
    '''
    检查2000-2019每年的vsb的最大值
    返回超过30000的个数
    '''
    vsbMaxLis = df["2000":"2019"]["vsb"].resample("y").max()
    return len(list(vsbMaxLis[vsbMaxLis>30000]))

def get_train_data():
    dataPath = r"D:\大学\大三下\大创-机器学习\#WorkSpace_after_May\data\train_data_132_stations_clean2.csv"
    df_train = pd.read_csv(dataPath, index_col=0)
    df_train.index = pd.DatetimeIndex(df_train.index)
    return df_train

def get_code_lis():
    '''
    获得132个站点的code列表
    '''
    df_train = get_train_data()
    codeArray = list(df_train["code"].unique())
    return codeArray


def get_his_data_for_predict(stationCode: int)-> pd.DataFrame:
    '''
    直接从本地获得可以预测的his data
    '''
    path = r"D:\大学\大三下\大创-机器学习\#WorkSpace_after_May\data\hisData_clean_132"
    for name in os.listdir(path):
        if int(name[:name.find(".")])==stationCode:
            df = pd.read_csv(path+"\\"+name, index_col=0)
            df.index = pd.DatetimeIndex(df.index)
            break
    return df

if __name__ == "__main__":
    df = get_his_data_for_predict(590820)
    print(df.head())





