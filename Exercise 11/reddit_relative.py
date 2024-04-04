import sys
from pyspark.sql import SparkSession, functions as f, types

spark = SparkSession.builder.appName('reddit relative scores').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

'''
1. Calculate the average score for each subreddit, as before.
2. Exclude any subreddits with average score ≤ 0.
3. Join the average score to the collection of all comments. Divide to get the relative score.
4. Determine the max relative score for each subreddit.
5. Join again to get the best comment on each subreddit: we need this step to get the author.

output: uncompressed JSON fields: subreddit, author, rel_score
.cache() on DFs used more than once.
'''

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


def main(in_directory, out_directory):
    data = spark.read.json(in_directory, schema=comments_schema)

    # Select relevant features
    comments = data.select(
        data['author'],
        data['subreddit'],
        data['score'],
    )
    # Calculate averages
    averages = spark.createDataFrame(comments.groupBy('subreddit').agg(f.avg('score').alias('average')).collect())
    # Exclude subreddits with avg score <= 0
    averages = averages.filter(f.col('average') > 0).broadcast()
    # Add average score for each comment as per subreddit
    comments = comments.join(averages, "subreddit").cache()
    # Add relative score for each comment as per subreddit
    comments = comments.withColumn('rel_score', f.try_divide(f.col('score'),f.col('average')))
    # Find the max relative score for each subreddit
    best_author = comments.groupBy('subreddit').agg(
        f.max_by('author', 'rel_score').alias('author'),
       # f.max_by('score', 'rel_score').alias('score'),
       # f.max_by('average', 'rel_score').alias('average'),
        f.max_by('rel_score', 'rel_score').alias('rel_score'),
    )

    best_author.write.json(out_directory, mode='overwrite')


if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
