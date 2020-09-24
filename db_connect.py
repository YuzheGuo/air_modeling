#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import pymysql
import numpy as np
from math import radians, cos, sin, asin, sqrt


# ## 定义计算球面距离的公式

# In[4]:


#return kilometer
def cal_distance(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    输出：距离km
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine公式
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # 地球平均半径，单位为公里
    return c * r 
# print(haversine(116.46,39.92,116.45,39.90))


# ## 根据code和city获得站点的位置信息

# In[5]:


def find_location(code:int):
    '''
    输入visi站点的城市，输出data的列表
    [lon,lat,province,city,stationCode_6]
    '''
    sql = 'select lon,lat,province,city,stationCode_6 from visi_station_selected_within_30km a where a.stationCode_6={}'.format(code)
    config1={
    "host":"rm-uf622639ux3c1wwjiio.mysql.rds.aliyuncs.com",
    "user":"airdb_admin",
    "password":"2019@sjtu",
    "database":"visibility_data"}
    db_air = pymysql.connect(**config1)
    cur1 = db_air.cursor()
    cur1.execute(sql)
    res = cur1.fetchall()
    
    if len(res)==1: return res[0]
    else: 
        print('error: find no matched station_info')
        return 


# In[6]:


def find_location_city(city:str):
    '''
    输入visi站点的城市，输出data的列表
    [lon,lat,province,city,stationCode_6]
    '''
    sql = 'select lon,lat,province,city,stationCode_6 from visi_station_selected_within_30km a where a.city="{}"'.format(city)
    config1={
    "host":"rm-uf622639ux3c1wwjiio.mysql.rds.aliyuncs.com",
    "user":"airdb_admin",
    "password":"2019@sjtu",
    "database":"visibility_data"}
    db_air = pymysql.connect(**config1)
    cur1 = db_air.cursor()
    cur1.execute(sql)
    
    res = cur1.fetchall()
    if len(res)==1: return res[0]
    elif len(res)>1:
        print('error: find more than one station_info')
        return
    else:
        print('error: find no matched station_info')
        return


# # 连接数据库

# ## 获得某个站点的气象数据

# In[7]:


def get_weather_data(code: int, startYear=1956):
    '''
    从数据库中获得weather data，会含有空值
    输入：站点的编号（6位），开始的年份：默认为1956
    输出：dataframe的数据：所有年份的天气数据
    '''
    
    sql = 'select * from weather_data_view where code = {} and datetime>"{}-1-1"'.format(code,startYear)
   
    
    config1={
    "host":"rm-uf622639ux3c1wwjiio.mysql.rds.aliyuncs.com",
    "user":"airdb_admin",
    "password":"2019@sjtu",
    "database":"互联网比赛数据"}
    
    db_weather = pymysql.connect(**config1)
    cur1 = db_weather.cursor()
    cur1.execute(sql)
    data = pd.DataFrame(cur1.fetchall())
    cur1.close()
    
    #如果没有查询出来结果，返回空的dataframe
    if data.shape[0]==0:
        print('get_weather_data: find no weather data!')
        return pd.DataFrame([])
    #添加列标题和index
    data.columns = ['code','datetime','wd','ws','vsb','tem','psea','palt','psta','rh']
    data.index = data['datetime']
    
    #改变数据的类型:object->float64
    col = list(data.columns)
    col.remove('datetime')
    col.remove('code')
    data[col] = data[col].astype('float64')
    
    #数据检查：
    c = data.groupby(by=data.index).count()
    if len(c[c['code']>1]) != 0:
        print('error: one datetime -> two or more data !')
    
    return data

def get_weather_data_city(city, startYear=1956):
    '''
    获得weather data
    输入是城市的名称和起始年份
    输出是气象的数据
    '''
    info = find_location_city(city)
    return get_weather_data(info[4], startYear=startYear)
    


# In[8]:


# df = get_weather_data(504680,2010)
# # df.head()


# ## 获得空气污染的数据：PM2.5 or PM10

# In[16]:


def get_air_data(province:str, city:str,station:str,isPM25 = True, isPM10 = False, 
                 isSelectTime=False, isStationInfo=False):
    '''
    获得指定站点的空气污染数据，会含有空值
    输入：站点的省份，城市，站点名称，PM25和PM10的选项
    输出：对应的dataframe，含有省份，城市，站点名称，站点编号，对应的数值
    '''
    
    if isPM25:
        if isPM10:
            sql = "select a.province,a.city,a.station,a.station_code stationCode,a.pubtime datetime,a.pm2_5,a.pm10 from national_air_hourly.na_station_realtime a where a.province = '{}' and a.city='{}' and a.station='{}' and a.pm2_5 > 0 and a.pm2_5 is not Null and a.pm10 > 0 and a.pm10 is not Null".format(province,city,station)
        else:
            sql = "select a.province,a.city,a.station,a.station_code stationCode,a.pubtime datetime,a.pm2_5 from national_air_hourly.na_station_realtime a where a.province = '{}' and a.city='{}' and a.station='{}' and a.pm2_5 > 0 and a.pm2_5 is not Null".format(province,city,station)
    elif isPM10:
        sql = "select a.province,a.city,a.station,a.station_code stationCode,a.pubtime datetime,a.pm10 from national_air_hourly.na_station_realtime a where a.province = '{}' and a.city='{}' and a.station='{}' and a.pm10 > 0 and a.pm10 is not Null".format(province,city,station)
    else:
        print('error: isPM25 and isPM10 are False !')
    
    #减少IO的负担
    if isSelectTime:
        sql = sql + ' and hour(a.pubtime)%3=2'
    
    config1={
    "host":"rm-uf622639ux3c1wwjiio.mysql.rds.aliyuncs.com",
    "user":"airdb_admin",
    "password":"2019@sjtu",
    "database":"national_air_hourly"}
    db_air = pymysql.connect(**config1)
    cur1 = db_air.cursor()
    cur1.execute(sql)
    data = pd.DataFrame(cur1.fetchall())
    cur1.close()
    
    if data.shape[0]==0: print('find no air data')
    
    col = ['province','city','station','stationCode','datetime']
    if isPM25: col.append('pm2_5')
    if isPM10: col.append('pm10')
    data.columns = col
    data.index = data['datetime']
    
    # 检查是否有重复的情况，如果有的话，直接去除
    data.drop_duplicates(inplace=True)
    
    # 根据是否包含stationInfo的选项来确定最终的输出
    drop_col = ['province','city','station','stationCode','datetime']
    if not isStationInfo:
        return data.drop(columns=drop_col)
    else:
        return data
    


# In[14]:


# df = get_air_data('青海省','西宁市','第五水厂',isPM25=True,isPM10=False,isStationInfo=True)


# In[26]:


# df = get_air_data('青海省','西宁市','第五水厂',isPM25=True,isPM10=True)


# In[14]:


# get_air_data('青海省','西宁市','第五水厂',isPM25=False,isPM10=False)


# ## 获得站点周围m千米的空气污染站点列表

# In[11]:


def match_air_station(visi_lon:float,visi_lat:float,distance=5,min_dis=0):
    '''
    将指定的经纬度所在的visis_station和air_station进行匹配，给定特定的距离
    输入：经度和纬度，距离
    输出：dataframe类型的表格，包含站点的信息和相应的距离; 如果没有找到，输出整数0
    '''
    
    sql = 'select a.province,a.city,a.station,a.station_code,    round(`长三角能见度污染物反演测试数据`.dis_between_station({},{}, a.lng, a.lat),1) dis     from na_stations a '.format(visi_lon,visi_lat) + 'where `长三角能见度污染物反演测试数据`.    dis_between_station({},{}, a.lng, a.lat)<={} '.format(visi_lon,visi_lat,distance) + 'and     `长三角能见度污染物反演测试数据`.dis_between_station({},{}, a.lng, a.lat)>{} order by dis'.format(visi_lon,visi_lat,min_dis) 
    
    config1={
    "host":"rm-uf622639ux3c1wwjiio.mysql.rds.aliyuncs.com",
    "user":"airdb_admin",
    "password":"2019@sjtu",
    "database":"national_air_hourly"}
    db_air = pymysql.connect(**config1)
    cur1 = db_air.cursor()
    cur1.execute(sql)
    data = pd.DataFrame(cur1.fetchall())
    
    #如果没有找到站点，输出0
    if data.shape[0] == 0:
        print('find no matched air station within {}km !'.format(distance))
        return pd.DataFrame([])
    data.columns = ['province','city','station','stationCode','dis']
    
    #更改数据的类型
#     data.iloc[:,:-1] = data.iloc[:,:-1].astype('string')
    
    return data

def match_air_station_city(city, dis=5, min_dis=0):
    '''
    找出指定城市在指定的距离内的match air station list
    输入：城市名称，最近和最大距离
    输出：dataframe类型的表格，包含站点的信息和相应的距离; 如果没有找到，输出整数0'''
    info = find_location_city(city)
    return match_air_station(info[0], info[1], dis, min_dis)


# In[16]:


# df = match_air_station(102.744,24.992,20,min_dis=15)
# match_air_station_city('济南市', dis=100, min_dis=50)
# df.shape


# ## 获得候选的站点信息列表

# In[17]:


def get_weather_station_candidate():
    '''
    获得165个候选的站点信息
    输出：dataframe
    '''
    sql = 'select * from visi_station_selected_within_30km'
   
    
    config1={
    "host":"rm-uf622639ux3c1wwjiio.mysql.rds.aliyuncs.com",
    "user":"airdb_admin",
    "password":"2019@sjtu",
    "database":"visibility_data"}
    
    db = pymysql.connect(**config1)
    cur1 = db.cursor()
    cur1.execute(sql)
    data = pd.DataFrame(cur1.fetchall())
    cur1.close()
    col = ['code_6','code_5','name','lat','lon','ele','province','city']
    data.columns = col
    return data


# # 数据处理匹配模块

# ## get_train_data 获得匹配好的训练数据
# - 目前不是非常适用了，需要用最新的函数

# In[50]:


def get_train_data(code=0, city=0, distance=5, isPM25=True, isPM10=False, isSelectTime=True):
    '''
    获得某个站点的训练数据（PM2.5）：在某个特定的距离之内。
    
    数据不进行清洗，直接就展示出来
    
    80%的才能进行air站点的聚合
    
    输入：站点编号，经纬度，以及周围的距离，选择PM10和PM25的情况，是否选择仅仅间隔3h的air的数据
    输出：对应的训练数据，每隔三个小时取数据，并且能够将周围的PM数据汇总好
    '''
    visi_info = find_location_city(city) if bool(city) else find_location(code)
    for i in visi_info:
        print(i, end=' ')
    
    #获得能见度数据
    visiData = get_weather_data(visi_info[-1], 2012)
    print('\nget weather_data')
    
    #获得air数据
    airStations = match_air_station(visi_info[0], visi_info[1], distance = distance)
    
    if airStations.shape[0]==0:
        print('no matched air data')
        return pd.DataFrame([])
    else:
        print('number of air station matched: ', len(airStations.index))
    
    #获得airData，还没有聚合，但是已经放到一块了
    airData = pd.DataFrame([])
    for i, row in airStations.iterrows():
        df = get_air_data(row['province'], row['city'], row['station'], isPM25,isPM10,isSelectTime=isSelectTime)
        airData = pd.concat([airData,df])
        print('get air data at {} {} with number'.format(row['city'],row['station']), airData.shape[0])
    
    #将数据进行聚合，至少有50%的站点有数据
    num_min = int(len(airStations.index)*0.5)
    pmData = airData.resample('h').mean()
    count = airData.resample('h').count()
    pmData = pmData.loc[count[count['city']>=num_min].index]
    
    print('get averaged pmData with 50% avaliable')
    
    
    #将air和能见度进行聚合
    merge_df = pd.merge(visiData,pmData,left_on=visiData.index,right_on=pmData.index,
                        left_index=True).drop(columns='key_0')
    print('matched successfully with number {}'.format(merge_df.shape[0]))
    
    #转换数据的类型，变成float64,除了datetime之外
#     col = list(merge_df.columns)
#     col.remove('datetime')
#     merge_df[col] = merge_df[col].astype('float64')
    
    return merge_df


# In[53]:


# %%time
# res1 = get_train_data(city='北京市',distance=30,isPM25=False,isPM10=True, isSelectTime=True)


# In[54]:


# %%time
# res2 = get_train_data(city='北京市',distance=30,isPM25=False,isPM10=True, isSelectTime=False)


# In[51]:


# %%time
# for city in ['北京市','济南市','青岛市','广州市']:
#     get_train_data(city=city, distance=30,isPM25=False,isPM10=True, isSelectTime=False)


# In[55]:


# %%time
# for city in ['北京市','济南市','青岛市','广州市']:
#     get_train_data(city=city, distance=30,isPM25=False,isPM10=True, isSelectTime=True)


# In[1]:


# res = get_train_data(city='怀化市',distance=30,isPM25=False,isPM10=True)


# In[ ]:


# def find_matched_air_station

