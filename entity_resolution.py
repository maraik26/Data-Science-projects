import sys
import operator
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, explode, concat_ws
import re

from pyspark.sql.types import ArrayType,StringType,FloatType
spark = SparkSession.builder.master("local").appName("Entity Resolution").config(conf = SparkConf()).getOrCreate()

class EntityResolution:
    def __init__(self, dataFile1, dataFile2, stopWordsFile):
        self.f = open(stopWordsFile, "r")
        self.stopWords = set(self.f.read().split("\n"))
        self.stopWordsBC = spark.sparkContext.broadcast(self.stopWords).value
        self.df1 = spark.read.parquet(dataFile1).cache()
        self.df2 = spark.read.parquet(dataFile2).cache()
        
#to preprocess DataFrames by transforming each record into a set #of tokens 
#Return a new DataFrame that adds the "joinKey" column into the #input $df
    
    def preprocessDF(self, df, cols): 
        stopwords = self.stopWordsBC
        
        @udf(returnType = ArrayType(StringType()))
        def tokens(r):
            token = re.split('\W+', r) 
#to split a string into a set of tokens
            for t in token:        
                t = t.lower() 
#Convert each token to its lower-case            
            return list(set([x for x in token if len(x)>0 and x not in stopwords]))
        
#apply the tokenizer to the concatenated string
#transform_udf = udf(tokens,ArrayType(StringType()))
        
#Return a new DataFrame that adds the "joinKey" column into the #input $df
#$df represents a DataFrame
#$cols represents the list of columns (in $df) that will be #concatenated and be tokenized

        df = df.withColumn('joinKey',concat_ws(' ',*df[cols]))
        df = df.withColumn('joinKey',tokens(df.joinKey))
        return df
    
#This function will filter all the record pairs whose joinKeys #do not share any token. 
#This is because that based on the definition of Jaccard, we can #deduce that if two sets do not share any element 
#, their Jaccard similarity values must be zero. Thus, we can #safely remove them.
#Return a new DataFrame $candDF with four columns: 'id1', #'joinKey1', 'id2', 'joinKey2'
#$candDF is the joined result between $df1 and $df2 on the #condition that their joinKeys share at least one token. 
    
    def filtering(self, df1, df2):
        df1_expl = df1.withColumn('col1', explode(df1.joinKey))
        df2_expl = df2.withColumn('col1', explode(df2.joinKey))
        
        df1_expl.createOrReplaceTempView("table1")
        df2_expl.createOrReplaceTempView("table2")
        
        Joined_dataframes_expl = spark.sql('''
            SELECT distinct table1.id as id1, table1.joinKey as joinKey1, table2.id as id2, table2.joinKey as joinKey2 from table1 inner join table2 on table1.col1 = table2.col1 ''') 

# "on" means based on some condition
# distinct just choose one entry without duplicate

        return Joined_dataframes_expl

    def verification(self, candDF, threshold):
        
        @udf(returnType = FloatType())
        def jaccard_similarity(col1, col2):
            a = set(col1).intersection(set(col2)) 
            b = set(col1).union(set(col2))
            c = len(a)/len(b)
            return c
                
        jaccard_df = candDF.withColumn("jaccard", jaccard_similarity(candDF.joinKey1, candDF.joinKey2))
        jaccard_df = jaccard_df.filter(jaccard_df.jaccard>=threshold)
        
        return jaccard_df

#Compute precision, recall, and fmeasure of $result based on Grt 
#and return the evaluation result as a triple: (precision, #recall, fmeasure)
    def evaluate(self, result, groundTruth):
        k = 0
        m = 0
        for i in result:
            if i in groundTruth:
                k = k + 1
        precision = k/len(result)
        
        for l in groundTruth:
            if l in result:
                m = m + 1
        recall = m/len(groundTruth)      
        
        fmeasure = 2*precision*recall/(precision+recall)
        
#Thus, the ER result should be a set of identified matching #pairs, denoted by R. 
#One thing that we want to know is that what percentage of the #pairs in  R  that are truly matching? 
#This is what Precision can tell us. Let  T  denote the truly #matching pairs in  R
        
# what percentage of truly matching pairs that are identified. 
# Let  AA denote the truly matching pairs in the entire dataset. #Recall is defined as:
        
        return (precision, recall, fmeasure)

    def jaccardJoin(self, cols1, cols2, threshold):
        newDF1 = self.preprocessDF(self.df1, cols1)
        
        newDF2 = self.preprocessDF(self.df2, cols2)
        print ("Before filtering: %d pairs in total" %(self.df1.count()*self.df2.count())) 
        
#$candDF is the joined result between $df1 and $df2 on the #condition that their joinKeys share at least one token.
        candDF = self.filtering(newDF1, newDF2)
        print ("After Filtering: %d pairs left" %(candDF.count()))
        
#$resultDF adds a new column, called jaccard, which stores the #jaccard similarity between $joinKey1 and $joinKey2
#$resultDF removes the rows whose jaccard similarity is smaller #than $threshold
        resultDF = self.verification(candDF, threshold) 
        print ("After Verification: %d similar pairs" %(resultDF.count()))

        return resultDF


    def __del__(self):
        self.f.close()

if __name__ == "__main__":
    er = EntityResolution("Amazon_sample", "Google_sample", "stopwords.txt")
    amazonCols = ["title", "manufacturer"]
    googleCols = ["name", "manufacturer"]
    resultDF = er.jaccardJoin(amazonCols, googleCols, 0.5)
    
    result = resultDF.rdd.map(lambda row: (row.id1, row.id2)).collect()
    groundTruth = spark.read.parquet("Amazon_Google_perfectMapping_sample").rdd.map(lambda row: (row.idAmazon, row.idGoogle)).collect()
    print ("(precision, recall, fmeasure) = ", er.evaluate(result, groundTruth))

