import sys
from pyspark.sql import SparkSession, functions as f, types
from datetime import datetime

spark = SparkSession.builder.appName('wikipedia popular').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

wiki_schema = types.StructType([
    types.StructField("lang", types.StructType()),
    types.StructField("page", types.StringType()),
    types.StructField("views", types.IntegerType()),
    types.StructField("bytes", types.IntegerType()),
])

'''
Find the most-viewed page for each hour (file) & number of times viewed
fromat: 'lang Page Count Byes'
(1) English Wiki pages (language = 'en')
(2) Exclude "Main_Page" from results.
(3) Exclude "Special:" from results.
Note: Smaller DS with subset pages might not have the main page or special pages

filenames: pagecounts-YYYYMMDD-HHMMSS*
want: YYYYMMDD-HH substring as a label for day/hour
'''

# Retrieve file name
# spark.read.csv(...).withColumn('filename', functions.input_file_name())

# Write a python function that takes pathnames and returns string like YYYYMMDD-HH then turn into a UDF.
# path_to_hour = functions.udf(..., returnType=types.StringType())

# Most FQ page -> find the largest # page_views/hour
# Join back to collection of all page counts
# Keep only those with count == max(count) for that hour

# Sort results by date/hour and output a CSV as:
# YYYYMMDD-HH,_page_,_count_

def main(input, output):

    def get_date(x):
        x1 = x.rsplit('/', 1)[-1]
        x2 = x1.rsplit('.', 1)[0]
        x3 = x2.split('-', 1)[1]
        return x3

    # Specify filename format
    filename = f.udf(lambda x: get_date(x))

    # Read the input
    file = spark.read.option("delimiter", " ").csv(input).withColumn('filename', filename(f.input_file_name()))

    # Create and organize df
    data = file.selectExpr("_c0 as lang", "_c1 as page", "_c2 as views", "_c3 as bytes", "filename as timestamp")
    data = data.withColumn("views", f.col("views").astype(types.IntegerType()))

    # Filter data according to specified conditions
    filter_data = data.filter(~f.lower(f.col('page')).contains('main_page') &
                             ~f.lower(f.col('page')).contains('special:') &
                              f.lower(f.col('lang')).contains('en')).sort('views', ascending=False).cache()
    #filter_data.show()

    # Find the max page viewed entry
    max_views = filter_data.groupBy('timestamp').agg(
        f.max_by('page','views').alias('page'),
        f.max('views').alias("views"),).sort(f.col('timestamp'), ascending=False)

    #max_views.show()

    max_views.write.csv(output + '-wiki', mode='overwrite')


    # ---- ETC -----

    # Filtering data and converting string to timestamp
    # data_timestamp = filter_data.withColumn("timestamp", f.to_date(f.col('timestamp'), 'yyyyMMdd-HHmmss'))
    # data_timestamp.show()

    # Selecting and sorting values only
    # sorted_views = filter_data.select(['timestamp','page','views']).sort('timestamp', ascending=False)
    # sorted_views.show()

    # Getting the max view count
    # max_viewed_count = sorted_views.filter(f.max(f.col('views'))).collect()[0][0]
    # max_viewed_count.show()


if __name__=='__main__':
    input = sys.argv[1]
    output = sys.argv[2]
    main(input, output)