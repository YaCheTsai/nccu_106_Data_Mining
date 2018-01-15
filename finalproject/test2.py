# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd 
import matplotlib.pyplot as plt
from pandas import Series, DataFrame 
from datetime import datetime

SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
merge = pd.read_csv(SCRIPT_PATH + '/merge.csv') 
air_visit_data = pd.read_csv(SCRIPT_PATH + '/air_visit_data.csv')

def set_reserve_visitors_x(row):
    if np.isnan(row['reserve_visitors_x']):
        return row['reserve_visitors_y']
    else:
        return  row['reserve_visitors_x']
        
        
def set_air_genre_name(row):
    
    if  pd.isnull(row['air_genre_name']):
        return row['hpg_genre_name']
    else:
        return  row['air_genre_name']
        
def set_air_area_name(row):
    if pd.isnull( row['air_area_name']):
        return row['hpg_area_name']
    else:
        return  row['air_area_name']
        
def set_latitude_x(row):
    if pd.isnull( row['latitude_x']):
        return row['latitude_y']
    else:
        return  row['latitude_x']
        
def set_longitude_x(row):
    if pd.isnull( row['longitude_x']):
        return row['longitude_y']
    else:
        return  row['longitude_x']

def set_calendar_date_x(row):
    if pd.isnull( row['calendar_date_x']):
        return row['calendar_date_y']
    else:
        return  row['calendar_date_x']
        
def set_day_of_week_x(row):
    if pd.isnull( row['day_of_week_x']):
        return row['day_of_week_y']
    else:
        return  row['day_of_week_x']
        
def set_holiday_flg_x(row):
    if pd.isnull( row['holiday_flg_x']):
        return row['holiday_flg_y']
    else:
        return  row['holiday_flg_x']
merge['reserve_visitors_x'] = merge.apply(set_reserve_visitors_x, axis=1)

merge['air_genre_name'] = merge.apply(set_air_genre_name, axis=1)
merge['air_area_name'] = merge.apply(set_air_area_name, axis=1)
merge['latitude_x'] = merge.apply(set_latitude_x, axis=1)
merge['longitude_x'] = merge.apply(set_longitude_x, axis=1)
merge['calendar_date_x'] = merge.apply(set_calendar_date_x, axis=1)
merge['day_of_week_x'] = merge.apply(set_day_of_week_x, axis=1)
merge['holiday_flg_x'] = merge.apply(set_holiday_flg_x, axis=1)

del merge['hpg_store_id']
del merge['hpg_genre_name']
del merge['hpg_area_name']
del merge['latitude_y']
del merge['longitude_y']
del merge['calendar_date_y']
del merge['day_of_week_y']
del merge['holiday_flg_y']


merge.rename(columns={'air_genre_name': 'genre_name', 'air_area_name': 'area_name', 'latitude_x': 'latitude', 'longitude_x': 'longitude', 'calendar_date_x': 'calendar_date', 'day_of_week_x': 'day_of_week', 'holiday_flg_x': 'holiday_flg', 'visit_datetime': 'visit_date'},inplace=True)

merge = pd.merge(merge,air_visit_data,how='outer', on=['air_store_id','visit_date'])
merge=merge.sort_values(['air_store_id', 'visit_date'], ascending=True)

merge.to_csv(SCRIPT_PATH + '/data.csv', index=False)