# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd 
import matplotlib.pyplot as plt
from pandas import Series, DataFrame 
from datetime import datetime

SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
fields = ['air_store_id', 'station_id']

or_data = pd.read_csv(SCRIPT_PATH + '/F_train.csv')
#or_data = pd.read_csv(SCRIPT_PATH + '/F_test.csv')
air_relation = pd.read_csv('./weather/air_store_info_with_nearest_active_station.csv', skipinitialspace=True, usecols=fields)


id = or_data['air_store_id']
id = pd.DataFrame(id).drop_duplicates()['air_store_id']


result = pd.DataFrame()
for i in id :
    tmp = air_relation.loc[air_relation['air_store_id'] == i].values[0][1]
    
    dtail_weather = pd.read_csv('./weather/Weather/'+ tmp +'.csv')
    
    or_tmp = or_data.loc[or_data['air_store_id'] == i]
    
  
    tmp_result = pd.merge(or_tmp,dtail_weather,how='left',left_on=' visit_date', right_on='calendar_date' )
    #tmp_result = pd.merge(or_tmp,dtail_weather,how='left',left_on='date', right_on='calendar_date' )
    result =  pd.concat([result, tmp_result])
    
   

result.fillna(0, inplace=True)    
result.to_csv(SCRIPT_PATH + '/train.csv', index=False)
#result.to_csv(SCRIPT_PATH + '/test.csv', index=False)

    




