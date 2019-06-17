
# anomaly_detection.py
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.sql import Row
import operator

spark = SparkSession.builder         .master("local")         .appName("Anomalies Detection")         .config("spark.some.config.option", "some-value")         .getOrCreate()

#convert special values which are not numbers to a vector in a form [0,1]
def onehot(strng, indices, values, c):
    conv = [0.0]*c # one value will be 0, c is a number of those special values
    others = [float(strng[k]) for k in range(len(strng)) if k not in indices] #other values ina row which are not special simbols
    for i in indices:
        indx = values.index(Row(strng[i]))
        conv[indx] = 1.0 # another value will be 1
    conv.extend(others) # extend the one-hot vector with original numerical list,
    return conv

class AnomalyDetection():
    
    #def readToyData(self):
    #    data = [(0, ["http", "udt", 0.4]), \
    #            (1, ["http", "udf", 0.5]), \
    #            (2, ["http", "tcp", 0.5]), \
    #            (3, ["ftp", "icmp", 0.1]), \
    #            (4, ["http", "tcp", 0.4])]
    #    schema = ["id", "rawFeatures"]
    #    self.rawDF = spark.createDataFrame(data, schema)

    def readData(self, filename):
        self.rawDF = spark.read.parquet(filename).cache()

# in rawFeatures, the first 2 categorical data convert to one hot vector such as [0,0,1,0,1,0]
# extend the one-hot vector with original numerical list, and all convert to Double type
# put the numerical list to a new column called "features"    
    def cat2Num(self, df, indices):
        simbols = [] # list of special values which are not numbers
        for i in indices:
            d = udf(lambda r: r[i], StringType())
            other = df.select(d(df.rawFeatures)).distinct().collect() #other numbers
            simbols.extend(other) # extend list of simbols with others numbers

        number_of_simbols = len(simbols)
        convertUDF = udf(lambda r: onehot(r, indices, simbols, number_of_simbols), ArrayType(DoubleType()))# converted simbols plus others
        new_dataframe = df.withColumn("features", convertUDF(df.rawFeatures)) # add a new column with converted simbols plus other numbers
        
        print("number of special simbols", len(simbols)) # number of special simbols which are not numbers (12)
        return new_dataframe

#Input: $df represents a DataFrame with four columns: "id", "rawFeatures", "features", and "prediction"
#Output: Return a new DataFrame that adds the "score" column into the input $df
#To compute the score of a data point x, we use:
#score(x) = (N_max - N_x)/(N_max - N_min), Nmax  and  Nmin  reflect the size of the largest and smallest clusters, respectively.  
#Nx  represents the size of the cluster assigned to  x
#score(x)=1 when x is assigned to the smallest cluster and score(x) = 0 when x is assigned to a large cluster.

    def addScore(self, df):
        cluster_dict = {}
        all_data = df.select("prediction").collect() #take all data and add prediction column
        
        print(len(all_data))
        
        for c in all_data:
            cluster_dict[c] = cluster_dict.setdefault(c,0.0)+1.0
        sorted_clusters = sorted(cluster_dict.items(), key=operator.itemgetter(1))  # sort by value
        n_max = sorted_clusters[-1][1] #maximum size of cluster
        n_min = sorted_clusters[0][1]  #minimum size of cluster
        score = udf(lambda p: float(n_max - cluster_dict.get(Row(p)))/(n_max - n_min), DoubleType())
        score_dataframe = df.withColumn("score", score(df.prediction)) #calculating score based on predicted clusters
        return score_dataframe

    def detect(self, k, t):
        # Encoding categorical features using one-hot.
        df1 = self.cat2Num(self.rawDF, [0, 1]).cache()
        df1.show(n=5, truncate=True)

        # Clustering points using KMeans
        features = df1.select("features").rdd.map(lambda row: row[0]).cache()
        model = KMeans.train(features, k, maxIterations=40, initializationMode="random", seed=60)

        # Adding the prediction column to df1
        modelBC = spark.sparkContext.broadcast(model)
        predictUDF = udf(lambda x: modelBC.value.predict(x), StringType())
        df2 = df1.withColumn("prediction", predictUDF(df1.features)).cache()
        df2.show(n=5, truncate=True)

        # Adding the score column to df2; The higher the score, the more likely it is an anomaly
        df3 = self.addScore(df2).cache()
        df3.show(n=10, truncate=True)

        return df3.where(df3.score > t)

if __name__ == "__main__":
    ad = AnomalyDetection()
    
    #ad.readToyData()
    #anomalies = ad.detect(2, 0.9)
    
    ad.readData('data/logs-features-sample')
    anomalies = ad.detect(8, 0.97)
    
    print(anomalies.count(), "Anomalies")
    anomalies.show(n=20, truncate=True)

