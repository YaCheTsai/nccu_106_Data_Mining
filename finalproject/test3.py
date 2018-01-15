# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd 
import matplotlib.pyplot as plt
from pandas import Series, DataFrame 
from datetime import datetime

SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
data = pd.read_csv(SCRIPT_PATH + '/data.csv') 

air_store_info = pd.read_csv(SCRIPT_PATH + "/air_store_info.csv") 
hpg_store_info = pd.read_csv(SCRIPT_PATH + "/hpg_store_info.csv") 
store_id_relation = pd.read_csv(SCRIPT_PATH + "/store_id_relation.csv") 
date_info = pd.read_csv(SCRIPT_PATH + "/date_info.csv") 

del data['genre_name']
del data['area_name']
del data['latitude']
del data['longitude']
del data['calendar_date']
del data['day_of_week']
del data['holiday_flg']

result = pd.merge(store_id_relation,hpg_store_info, on=['hpg_store_id','hpg_store_id'])
del result['hpg_store_id']

result = pd.merge(result,air_store_info,how='outer', on=['air_store_id','air_store_id'])

del result['hpg_area_name']
del result['hpg_genre_name']
del result['latitude_x']
del result['longitude_x']

result.rename(columns={'air_genre_name': 'genre_name', 'air_area_name': 'area_name', 'latitude_y': 'latitude', 'longitude_y': 'longitude'},inplace=True)

result = pd.merge(data,result, on=['air_store_id','air_store_id'])

result = pd.merge(result,date_info,how='left',left_on='visit_date', right_on='calendar_date' )
result=result.sort_values(['air_store_id', 'visit_date'], ascending=True)

result.to_csv(SCRIPT_PATH + '/data2.csv', index=False)
#data2 detail version for testset 

result = result[~ np.isnan(result['visitors'])]  


result.fillna(0, inplace=True)
print result 
del result['calendar_date']

result.to_csv(SCRIPT_PATH + '/train.csv', index=False)
#train detail version for trainset 