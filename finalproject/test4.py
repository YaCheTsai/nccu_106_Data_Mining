# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd 
import matplotlib.pyplot as plt
from pandas import Series, DataFrame 
from datetime import datetime

SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
data = pd.read_csv(SCRIPT_PATH + '/sample_submission.csv')
data2 = pd.read_csv(SCRIPT_PATH + '/data2.csv')
air_store_info = pd.read_csv(SCRIPT_PATH + "/air_store_info.csv") 
hpg_store_info = pd.read_csv(SCRIPT_PATH + "/hpg_store_info.csv") 
store_id_relation = pd.read_csv(SCRIPT_PATH + "/store_id_relation.csv") 
date_info = pd.read_csv(SCRIPT_PATH + "/date_info.csv")  
hpg_reserve = pd.read_csv(SCRIPT_PATH + "/hpg_reserve.csv") 
air_reserve = pd.read_csv(SCRIPT_PATH + "/air_reserve.csv") 

def set_date(row):
    x= str(row['id'])
    id1,id2,date = x.split("_")
    return date
    
def set_id(row):
    x= str(row['id'])
    id1,id2,date = x.split("_")
    return id1+"_"+id2

    
data['air_store_id'] = data.apply(set_id, axis=1)
data['date'] = data.apply(set_date, axis=1)

del data2['visitors']
del data2['genre_name']
del data2['area_name']
del data2['latitude']
del data2['longitude']
del data2['calendar_date']
del data2['day_of_week']
del data2['holiday_flg']

data = pd.merge(data,data2,how='left',left_on=['air_store_id','date'], right_on=['air_store_id','visit_date'] )

result = pd.merge(store_id_relation,hpg_store_info, on=['hpg_store_id','hpg_store_id'])
del result['hpg_store_id']

result = pd.merge(result,air_store_info,how='outer', on=['air_store_id','air_store_id'])

del result['hpg_area_name']
del result['hpg_genre_name']
del result['latitude_x']
del result['longitude_x']

result.rename(columns={'air_genre_name': 'genre_name', 'air_area_name': 'area_name', 'latitude_y': 'latitude', 'longitude_y': 'longitude'},inplace=True)

result = pd.merge(data,result, on=['air_store_id','air_store_id'])

result = pd.merge(result,date_info,how='left',left_on='date', right_on='calendar_date' )

del result['calendar_date']
del result['visit_date']
result.fillna(0, inplace=True)
print result
result.to_csv(SCRIPT_PATH + '/test.csv', index=False)