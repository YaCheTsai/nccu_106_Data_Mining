# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd 
import matplotlib.pyplot as plt
from pandas import Series, DataFrame 
from datetime import datetime

def replace_date(x):
 
    x = str(x)
    return x[:10]  

def replace_date1(x):
 
    x = str(x)
    if x[:2] == '0:':
        return int('0')
    else :
        return int(x[:2].replace(" ", ""))
    
def set_SUM_visitors(row): 
    if np.isnan(row['reserve_visitors_y']) :
        x = float(row['reserve_visitors_x']) + 0 
    elif np.isnan(row['reserve_visitors_x']) :
        x = float(row['reserve_visitors_y']) + 0 
    else:
        x = float(row['reserve_visitors_x']) + float(row['reserve_visitors_y']) 
    
    return x 
    
def set_avg_date(row):
    x = datetime.strptime(row['reserve_datetime'] , '%Y-%m-%d')
    y = datetime.strptime(row['visit_datetime'] , '%Y-%m-%d')
    
    return  str(y-x)

def set_avg_reserve_date(row):
    if np.isnan(row['mean_y']) :
        return float(row['mean_x']) 
    elif np.isnan(row['mean_x']) :
        return float(row['mean_y'])
    else:
        y = float(row['count_x']) + float(row['count_y'])
        x = float(row['mean_x'])*float(row['count_x']) + float(row['mean_y'])*float(row['count_y'])
        return x/y 
    

 
SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
air_visit_data = pd.read_csv(SCRIPT_PATH + "/air_visit_data.csv") 
air_reserve = pd.read_csv(SCRIPT_PATH + "/air_reserve.csv") 
air_store_info = pd.read_csv(SCRIPT_PATH + "/air_store_info.csv") 
date_info = pd.read_csv(SCRIPT_PATH + "/date_info.csv") 
hpg_reserve = pd.read_csv(SCRIPT_PATH + "/hpg_reserve.csv") 
hpg_store_info = pd.read_csv(SCRIPT_PATH + "/hpg_store_info.csv") 
store_id_relation = pd.read_csv(SCRIPT_PATH + "/store_id_relation.csv") 

#remove time
air_reserve['visit_datetime'] = air_reserve['visit_datetime'].map(lambda x:replace_date(x))
air_reserve['reserve_datetime'] = air_reserve['reserve_datetime'].map(lambda x:replace_date(x))
hpg_reserve['visit_datetime'] = hpg_reserve['visit_datetime'].map(lambda x:replace_date(x))
hpg_reserve['reserve_datetime'] = hpg_reserve['reserve_datetime'].map(lambda x:replace_date(x))

#count reserve num
air_grouped = air_reserve.groupby(['air_store_id','visit_datetime']).sum().reset_index()

hpg_grouped = hpg_reserve.groupby(['hpg_store_id','visit_datetime']).sum().reset_index()

#avg reserve_datetime to visit_datetime
air_reserve['avg_reserve_date'] = air_reserve.apply(set_avg_date, axis=1)
air_reserve['avg_reserve_date'] = air_reserve['avg_reserve_date'].map(lambda x:replace_date1(x))
air_grouped1 = air_reserve.groupby(['air_store_id','visit_datetime'])['avg_reserve_date'].agg(['mean', 'count']).astype(int).reset_index()

hpg_reserve['avg_reserve_date'] = hpg_reserve.apply(set_avg_date, axis=1)
hpg_reserve['avg_reserve_date'] = hpg_reserve['avg_reserve_date'].map(lambda x:replace_date1(x))
hpg_grouped1 = hpg_reserve.groupby(['hpg_store_id','visit_datetime'])['avg_reserve_date'].agg(['mean', 'count']).astype(int).reset_index()


#merge avg_reserve_date & reserve num
result = pd.merge(air_grouped,air_grouped1, on=['air_store_id','visit_datetime'])
h_result = pd.merge(hpg_grouped,hpg_grouped1, on=['hpg_store_id','visit_datetime'])

#merge above to store info & date info 
result = pd.merge(result, air_store_info, left_on='air_store_id', right_on='air_store_id') 
result = pd.merge(result, date_info, left_on='visit_datetime', right_on='calendar_date')
result = result.sort_values(['air_store_id','visit_datetime'],ascending=True)

h_result = pd.merge(h_result, hpg_store_info, left_on='hpg_store_id', right_on='hpg_store_id') 
h_result = pd.merge(h_result, date_info, left_on='visit_datetime', right_on='calendar_date')
h_result = pd.merge(store_id_relation,h_result, on=['hpg_store_id','hpg_store_id'])
h_result = h_result.sort_values(['hpg_store_id','visit_datetime'],ascending=True)

#merge hpg & air
result = pd.merge(result, h_result, how='outer', on=['air_store_id', 'visit_datetime'])
#count reserve num sum
result['SUM_reserve_visitors'] = result.apply(set_SUM_visitors, axis=1)
result['avg_reserve_date'] = result.apply(set_avg_reserve_date, axis=1).astype(int)

result.to_csv(SCRIPT_PATH + "/merge.csv", index=False)



