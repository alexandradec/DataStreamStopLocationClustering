import pyspark as spark
import os
from pyspark import SparkContext, SparkConf
import pyspark.sql as spark
from pyspark.sql.types import DoubleType
from pyspark.sql.types import StringType
from pyspark.sql.types import StructType, StructField
from pyspark.sql.functions import udf,col,when,count
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.functions import lit
from pyspark.sql import DataFrame

# Set spark environments
os.environ['PYSPARK_PYTHON'] = '/opt/miniconda3/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/opt/miniconda3/bin/python'
conf = SparkConf().setAppName('app').setMaster('local[10]')
sc = SparkContext(conf=conf)
ss = spark.SparkSession.builder.getOrCreate()

cd = os.getcwd()

path = cd + '/temp_f_day17'
df = ss.read.parquet(path)

import pandas as pd
from DenStream import DenStream
schema = StructType([StructField("user_id", StringType(),True), StructField("stop_id", StringType(),True), StructField("Start_time", StringType(),True), StructField("Finish_time", StringType(),True), StructField("Med_Lat", StringType(),True), StructField("Med_Long", StringType(),True), StructField("user_group", StringType(),True), StructField("Grid_index", StringType(),True), StructField("destination_id", StringType(),True), StructField("DenStream_RunningTime", StringType(), True)])
@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def runDenStream(pdf):
    grid_ = pdf['Grid_index'].values[0]
    
    pdf = pdf.astype({'Med_Lat': 'float'})
    pdf = pdf.astype({'Med_Long': 'float'})
    pdf = pdf.astype({'stop_id': 'str'})
    
    coords_ = pdf[['Med_Lat','Med_Long']]
    ids = pdf['stop_id']
        
    coords = np.asarray(coords_)
    
    clusterer = DenStream(30, grid=grid_, lambd=0.001, _min_spatial_resolution=0.0001, verbose=False)
    tic = time.time()
    labels = clusterer.partial_fit(coords)
    #clusterer.save_clusterer()
    toc = time.time()
    print('Grid: %s, Size: %s, Time %s' %(grid_, len(pdf.index), (toc-tic)/60))

    l = np.column_stack((ids, labels))
    dest_ = pd.DataFrame(l, columns=['stop_id', 'destination_id'])
    
    pdf = pdf.astype({'Med_Lat': 'str'})
    pdf = pdf.astype({'Med_Long': 'str'})
    pdf = pdf.astype({'stop_id': 'str'})
    
    result = pd.merge(pdf, dest_, on=['stop_id'])
    result = result.drop(columns=['stop_label'])
    result['DenStream_RunningTime'] = (toc-tic)/60
    result = result.astype({'DenStream_RunningTime': 'str'})
    
    return result   

trajectory = df.groupBy('Grid_index').apply(runDenStream)
output_path = cd + '/Trajectories_100/2020021700'
trajectory.write.partitionBy('user_group').mode('append').format("parquet").save(output_path)