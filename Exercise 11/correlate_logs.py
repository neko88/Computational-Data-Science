import math
import sys

import pyspark
from pyspark import RDD
from pyspark.sql import SparkSession, functions as f, types, Row
import re

spark = SparkSession.builder.appName('correlate logs').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

line_re = re.compile(r"^(\S+) - - \[\S+ [+-]\d+\] \"[A-Z]+ \S+ HTTP/\d\.\d\" \d+ (\d+)$")


def line_to_row(line):
    """
    Take a logfile line and return a Row object with hostname and bytes transferred. Return None if regex doesn't match.
    """
    m = line_re.findall(line)
    if m:
        host = m[0][0]
        bytes = m[0][1]
        return Row(host=str(host), bytes=float(bytes))
    else:
        return None


def not_none(row):
    """
    Is this None? Hint: .filter() with it.
    """
    return row is not None


def create_row_rdd(in_directory):
    log_lines = spark.sparkContext.textFile(in_directory)
    rows = log_lines.map(line_to_row) \
        .filter(not_none)
    return rows
    # TODO: return an RDD of Row() objects



def main(in_directory):
    logs = spark.createDataFrame(create_row_rdd(in_directory),
                                 schema='host:string, bytes:float')

    values = logs.groupBy('host').agg(
        f.count('host').alias('count_requests'),
        f.sum('bytes').alias('sum_request_bytes'),
        (f.count('host')**2).alias('count_requests_squared'),
        (f.sum('bytes')**2).alias('sum_request_bytes_squared'),
        f.try_multiply(f.count('host'), f.sum('bytes')).alias('product'),
    )
    values = values.withColumn('n', f.lit(1))
   # values.show()

    sums = values.groupBy().sum().collect()

    n = sums[0]['sum(n)']
    x = sums[0]['sum(count_requests)']
    y = sums[0]['sum(sum_request_bytes)']
    x2 = sums[0]['sum(count_requests_squared)']
    y2 = sums[0]['sum(sum_request_bytes_squared)']
    xy = sums[0]['sum(product)']

    numerator = (n * xy) - x * y
    denominator_x = math.sqrt(n * x2 - x**2)
    denominator_y = math.sqrt(n * y2 - y**2)
    r = numerator / (denominator_x * denominator_y)

    print("r = %g\nr^2 = %g" % (r, r**2))


if __name__=='__main__':
    in_directory = sys.argv[1]
    main(in_directory)
