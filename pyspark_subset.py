def pyspark_subset(pyspark_cleaned):
    pyspark_cleaned = pyspark_cleaned.na.drop(subset='msg_lower')
    return pyspark_cleaned.limit(10000)
