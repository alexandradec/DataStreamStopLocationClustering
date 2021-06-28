import pandas as pd
import glob
import numpy as np
import pickle
import dask.dataframe as dd
import pyspark.sql as spark
import os
from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as F

conf = SparkConf().setAppName('app').setMaster('local[5]')
sc = SparkContext(conf=conf)
ss = spark.SparkSession.builder.getOrCreate()

input_path = r'/data/raw/2020021900'

cd = os.getcwd()

output_path = cd + '/data/2020021900'
os.mkdir(output_path)

df = ss.read.option('sep', "\t").csv(input_path)

df = df.toDF('Timestamp', 'user_id', 'Device_type', 'Lat', 'Long', 'Accuracy', 'Time_zone_offset', 'Classification_type', 'Transformation_type')

df = df.withColumn('user_group', F.substring('user_id',0,2))

df.write.partitionBy('user_group').mode('append').format("parquet").save(output_path)

