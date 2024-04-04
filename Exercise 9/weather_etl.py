import sys
from pyspark.sql import SparkSession, functions, types

'''
input:
spark-submit weather_etl.py weather-1 output        # locally
spark-submit weather_etl.py /courses/353/weather-1 output       # cluster

output:
cat output/* | zless            # local
hdfs dfs -cat output/* | zless          # cluster (not applicable)
'''

spark = SparkSession.builder.appName('weather ETL').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.4' # make sure we have Spark 2.4+

observation_schema = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('date', types.StringType()),
    types.StructField('observation', types.StringType()),
    types.StructField('value', types.IntegerType()),
    types.StructField('mflag', types.StringType()),
    types.StructField('qflag', types.StringType()),
    types.StructField('sflag', types.StringType()),
    types.StructField('obstime', types.StringType()),
])


def main(in_directory, out_directory):

    # 1. Read the input directory of .csv.gz files
    data = spark.read.csv(in_directory, schema=observation_schema)

    # TODO: finish here.

    # 2. Keep records of only:
    #    a. qflag(quality flag) is null
            # https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.sql.Column.isNull.html
    #    b. the station that starts with 'CA'; options:
            # https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.sql.Column.startswith.html
            # https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.sql.functions.substring.html
    #    c. the observation is 'TMAX'
   # data = spark.createDataFrame(data)

    weather = data.select(
        data['station'],
        data['date'],
        data['observation'] == 'TMAX',
        data['value'],
        data['mflag'],
        data['qflag'].isNull(),
        data['sflag'],
        data['station'].startswith('CA'),
        data['obstime'],
    )

    # 3. Divide the temperature by 10 (to C) -> create new col 'tmax'
    weather2 = weather.withColumn('tmax', weather['value']/10)

    # 4. Keep only the columns: station, date, & tmax
    weather_cleaned = weather2.select(
        weather2['station'],
        weather2['date'],
        weather2['tmax'],
    )

    # 5. Write result as a directory of JSON files GZIP compressed
    #   in the Spark one-JSON object-per-line way

    weather_cleaned.write.json(out_directory, compression='gzip', mode='overwrite')


if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
