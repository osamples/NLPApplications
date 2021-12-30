def pyspark_cleaned(dataset):
    import pyspark.sql.functions as F
    from pyspark.sql.functions import split, initcap, col, lower, upper, split, regexp_replace, trim

    def removePunctuation(column):
        # Define this to cleanse yes/no answers
        return lower(regexp_replace(column, '\\p{Punct}', '')).alias('sentence')

    pyspark_cleaned = dataset.withColumn('msg_lower', removePunctuation(dataset.text))
    return pyspark_cleaned
