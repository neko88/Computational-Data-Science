import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('reddit averages').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

#time spark-submit --master=local[1] reddit_averages.py reddit-2 output
#time spark-submit --master=local[1] reddit_averages.py reddit-0 output


comments_schema = types.StructType([
    types.StructField('archived', types.BooleanType()),
    types.StructField('author', types.StringType()),
    types.StructField('author_flair_css_class', types.StringType()),
    types.StructField('author_flair_text', types.StringType()),
    types.StructField('body', types.StringType()),
    types.StructField('controversiality', types.LongType()),
    types.StructField('created_utc', types.StringType()),
    types.StructField('distinguished', types.StringType()),
    types.StructField('downs', types.LongType()),
    types.StructField('edited', types.StringType()),
    types.StructField('gilded', types.LongType()),
    types.StructField('id', types.StringType()),
    types.StructField('link_id', types.StringType()),
    types.StructField('name', types.StringType()),
    types.StructField('parent_id', types.StringType()),
    types.StructField('retrieved_on', types.LongType()),
    types.StructField('score', types.LongType()),
    types.StructField('score_hidden', types.BooleanType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('subreddit_id', types.StringType()),
    types.StructField('ups', types.LongType()),
    #types.StructField('year', types.IntegerType()),
    #types.StructField('month', types.IntegerType()),
])

    # TODO: calculate averages, sort by subreddit. Sort by average score and output that too.
    # What is the average score in each subreddit?
        # (1) Subreddit Name
        # (2) Score (h -> l)
        # Two separate output dictionaries


def main(in_directory, out_directory):

    # Read data from json
    #comments = spark.read.json(in_directory, schema=comments_schema)
    comments = spark.read.json(in_directory)

    data = comments.select(
        comments['subreddit'],
        comments['score'],
    )

    #data.show()
    # Show unique values
    #data.select('subreddit').distinct().show()

    # -- Not Cached --
    #averages_by_subreddit = spark.createDataFrame(data.groupBy('subreddit').avg().collect()).sort('subreddit')
    #averages_by_score = spark.createDataFrame(data.groupBy('subreddit').avg().collect()).sort('avg(score)', ascending=False)
    #averages_by_subreddit.show()
    #averages_by_score.show()


    # -- Cached --
    averages = spark.createDataFrame(data.groupBy('subreddit').avg().collect()).cache()

    averages.sort('subreddit').write.csv(out_directory + '-subreddit', mode='overwrite')
    averages.sort('avg(score)', ascending=False).write.csv(out_directory + '-score', mode='overwrite')


if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
