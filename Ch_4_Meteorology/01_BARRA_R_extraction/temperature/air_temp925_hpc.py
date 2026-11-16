# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 16:40:12 2021

@author: sarojkumars
"""


# netCDF data extraction for single defined location

import glob
import os
from netCDF4 import Dataset
import pandas as pd
from os import walk
from datetime import datetime
import xarray as xr

# set up current working directory
dpath = os.environ.get("BARRA_DATA_DIR", "data/BARRA_R")
os.chdir(dpath)


#Create a list of time_coverage_start (as a netCDF global attribute) for each files in the directory

all_day = [] # create empty list

#since all_days list in BARRA data gives same date (19700101) for all list items and the date and time is stored in filename itself, create a filename list 
_, _, filenames = next(walk(dpath))
#rename filenames with date and time name

filenames = sorted(name for name in filenames if name.endswith(".nc"))

#print(filenames[0])
#print(filenames[1])


for file in glob.glob('*.nc'):
    print(file) # confirm files
    data = Dataset(file, 'r')
    day = data.time_coverage_start[0:19]
    all_day.append(day)
    
    print(all_day) # confirm that the first file (when sorted) is last day of previous month (one day before) and end day is the last day
    
# Creating an empty Pandas DataFrame for whole month
# Please note that the netCDF file has previous day date as a start from where we extract the min date, for ex. recording starts on 11:30 pm of 31 January for 1st of Jan 12am.
# This leads data frame (all_month) index date is of a day before, eg 2009-01-31 00:00:00 actually is of 2009-02-01 00:00:00 
# to get the day next to minimum, create a set of the list, e.g. s = set(a) and print sorted(s)[1]
all_day_set = set(all_day)
month_start= sorted(all_day_set)[1] # VVI -  confirm that month start is the first day; BE CAREFUL!!

    
day_start = month_start # data record starts on 31 of next month but is of 00 hrs of begining of the month 
day_end = max(all_day)
date_range = pd.date_range(start = str(day_start) + ' 00:00:00',
                           end = str(day_end) + ' 18:00:00',
                           #closed = "left",
                           freq = '6H')
df = pd.DataFrame(0.0, columns = ["air_temp"], index = date_range)
all_day.sort() # to restructure list to get value from lowest to highest date and time 


# Defining fire points based on CSV data

firepts = pd.read_csv(os.environ.get("FIRE_POINTS_CSV", "data/fire_points.csv"))

# Read all fire points with FID, latitude and longitude
for index, row in firepts.iterrows():
    firept_id = row['FID']
    fire_lat = row['LATITUDE']
    fire_lon = row['LONGITUDE']
    fire_date = row['ACQ_DateTime']
    
             
        
    #testlist1 = []
    value_list1= []
    
    for i in filenames:
                      
                
        # Reading the data
        #data2 = Dataset(str(i), 'r')
        data2 = xr.open_dataset(str(i))
        
        # Sorting the lat lon data of the netCDF file into variables
        lat = data2['latitude'].values
        lon = data2['longitude'].values
           
        #squared difference between the specified lat, lon and the lat,lon of the netCDF
        sq_diff_lat = (lat - fire_lat)**2
        sq_diff_lon = (lon - fire_lon)**2
        
        #Identify the index of the min value for lat and lon
        min_index_lat = sq_diff_lat.argmin()
        min_index_lon = sq_diff_lon.argmin()
        
        # reading data access
        level = data2.isel(pressure = 33)['air_temp']
        #level = data2.sel(pressure = 1000.0)['relhum']#select level in vertical dataset
        #time_mean = level.mean('time')
        #temp = time_mean.variables['soil_temp']
        #temp = time_mean.values #['soil_temp']
        temp = level.values# ['relhum']
        #temp = level.variables['relhum']
        
        print('recording data of: ' + str([firept_id])+ str(i) + ' ' + str(temp[min_index_lat, min_index_lon]))
        
      
        #testlist = str(temp[min_index_lat, min_index_lon])
        value_list = str(temp[min_index_lat, min_index_lon])
        value_list1.append(value_list)
        
    
    df["air_temp"]  = value_list1
    df["FID"] = firept_id
    df.to_csv((str(firept_id) + '.csv')) # index = True)
   

path = dpath
all_csv = glob.glob(os.path.join(path, "[0-9]*.csv"))
df_from_each_file = (pd.read_csv(f) for f in all_csv)
concatenated_df = pd.concat(df_from_each_file) #, ignore_index = True)
concatenated_df.to_csv('air_temp925mmhg_all_pts.csv', header=1)

# change column heading
min_temp_scrn_pts = pd.read_csv(r'air_temp925mmhg_all_pts.csv')
min_temp_scrn_pts.rename(columns={'Unnamed: 0': 'Index', 'Unnamed: 0.1': 'ACQ_DateTime'}, inplace = True)
#min_temp_scrn_pts.drop(columns=['Index']) # Drop first index column
cols = list(min_temp_scrn_pts.columns.values)
min_temp2 = min_temp_scrn_pts[[cols[-1]] + cols[1:2] + cols[2:3]]
min_temp2.sort_values(['FID', 'ACQ_DateTime'], ascending = [1, 1])
min_temp2.to_csv("air_temp925mmhg_pts2.csv", index = False)

custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
temp_date = pd.read_csv('air_temp925mmhg_pts2.csv', parse_dates=['ACQ_DateTime'], date_parser=custom_date_parser)

custom_date_parser2 = lambda x: datetime.strptime(x, "%d-%m-%y %H:%M")
firepts2 = pd.read_csv(os.environ.get("FIRE_POINTS_CSV", "data/fire_points.csv"), parse_dates=['ACQ_DateTime'], date_parser=custom_date_parser2)

#firepts_temp= firepts2.merge(temp_date, left_on='ACQ_DateTime', right_on='ACQ_DateTime')
firepts_temp= firepts2.merge(temp_date, left_on=['FID','ACQ_DateTime'], right_on=['FID','ACQ_DateTime'])

firepts_temp.to_csv('firepts_air_temp925mmhg.csv', index = False) 