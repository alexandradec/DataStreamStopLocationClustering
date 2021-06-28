import pandas as pd
import numpy as np
import pyspark as spark
import os
from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as F
import pyspark.sql as spark
from pyspark.sql.types import DoubleType
from pyspark.sql.types import StringType
from pyspark.sql.types import StructType, StructField
from pyspark.sql.functions import udf
from pyspark.sql.functions import pandas_udf, PandasUDFType

import infostop
from infostop import Infostop
from infostop import plot_map
from infostop import postprocess as post
from infostop import utils

import time
import uuid

from IPython.display import clear_output

# Set spark environments
os.environ['PYSPARK_PYTHON'] = '/opt/miniconda3/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/opt/miniconda3/bin/python'
conf = SparkConf().setAppName('app').setMaster('local[10]')
sc = SparkContext(conf=conf)
ss = spark.SparkSession.builder.getOrCreate()

schema = StructType([StructField("user_id", StringType(),True), StructField("stop_id", StringType(),True), StructField("stop_label", StringType(),True), StructField("Start_time", StringType(),True), StructField("Finish_time", StringType(),True), StructField("Med_Lat", StringType(),True), StructField("Med_Long", StringType(),True), StructField("Grid_index", StringType(),True)])

tic0 = time.time()
cd = os.getcwd()
path = cd + '/data/2020021900/'
groups = os.listdir(path)
groups = [group for group in groups if (group != '_SUCCESS' and group != '._SUCCESS.crc')]
noOfGroups = len(groups)

for index, user_group in enumerate(groups):
    
    tic1 = time.time()
    
    input_path = path + user_group
    df = ss.read.parquet(input_path)
    df = df.sort(F.col('user_id'),F.col('Timestamp'))
    df = df.withColumn('Timestamp', df['Timestamp'].cast(DoubleType()))
    df = df.withColumn('Long', df['Long'].cast(DoubleType()))
    df = df.withColumn('Lat', df['Lat'].cast(DoubleType()))
    
    df_ = df.select('user_id', 'Lat', 'Long', 'Timestamp')
    df_ = df_.sort(F.col('Timestamp').asc())
    
    #Run 1st step of Infostop to get stop events
    events = df_.groupBy('user_id').apply(aggregate_step1)
    
    output_path = cd + '/temp_f_day19'
    events.write.mode('append').partitionBy('Grid_index').save(output_path)
    
    toc1 = time.time()
    
    print('Done for group %s out of %s, in %s mins' %((index+1), noOfGroups, (toc1-tic1)/60))
    
    
toc0 = time.time()
print('Overall time: %s' %((toc0-tic0)/60))
    
@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def aggregate_step1(pdf):
    final_output = []
    model = Infostop()
    
    #Remove entries with (0,0) coordinates
    #zero_indices = pdf[(pdf[pdf.columns[1]] == 0) & (pdf[pdf.columns[2]] == 0)].index
    #pdf.drop(zero_indices, inplace = True) 
    
    #Create the numpy array for Infostop
    coords_ = pdf[[pdf.columns[1], pdf.columns[2],pdf.columns[3]]].values
    coords = np.asarray(coords_)
    coords_sort = coords[coords[:,2].argsort()]
    username = pdf[pdf.columns[0]].values[0]
    
    #Infostop part 1 starts
    stop_events, event_map = model.find_stat_coords(coords_sort)
    
    #add stop labels to raw data
    stop_labels = np.column_stack((coords_sort, event_map[0]))
       
    if len(stop_events[0]) != 0:
        #Take the timestamps and stop labels from the raw data and run postprocess
        #to compute the time intervals per stop event (start time & finish time)
        labels = np.asarray([row[3] for row in stop_labels])
        times = np.asarray([row[2] for row in stop_labels])
        
        final_trajectory = post.compute_intervals(labels, times)
        
        #remove events labeled -1 (trips)
        final_trajectory = [rec for rec in final_trajectory if rec[0] != -1]
        
        #hash the user_id + stop_label
        stop_ids = utils.hash_stop_ids(username, final_trajectory)

        
        entries = np.column_stack((stop_ids, final_trajectory, stop_events[0]))
        
        #add indices
        grid_offset = np.array([24.62, -125.12])
        grid_spacing = np.array([0.5, 0.5])
        final_output_ = utils.add_grid_indices(entries, final_output, grid_offset, grid_spacing)
        
        #add the user_id
        user_id = np.full((len(final_output_), 1), username)
        final_output = np.column_stack((user_id, final_output_))
                
    out = pd.DataFrame(final_output, columns=["user_id","stop_id","stop_label", "Start_time", "Finish_time", "Med_Lat", "Med_Long", "Grid_index"])
    
    return out

