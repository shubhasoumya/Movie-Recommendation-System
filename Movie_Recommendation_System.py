# Databricks notebook source
pip install pyspark

# COMMAND ----------

from pyspark.sql import SparkSession
#initialize spark session

newspark = SparkSession.builder.appName('Movie Recommendation').getOrCreate()
 
 #load movies and ratings

movies_df=newspark.read.csv('dbfs:/FileStore/movie.csv',header=True,inferSchema=True)
movies_df.show(5)
ratings_df=newspark.read.csv('dbfs:/FileStore/rating.csv',header=True,inferSchema=True)
ratings_df.show(5)
#I don't want to remove null values so I am agaiin extracting ratings dataset with name o_ratings_df

o_rating_df=newspark.read.csv("dbfs:/FileStore/rating.csv", header=True, inferSchema=True)
o_rating_df.show()


# COMMAND ----------

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

#split o_ratings_df into training set and testing set

(training,test)=o_rating_df.randomSplit([0.8,0.2],seed=1234)

#initiaize ALS model

als=ALS(maxIter= 10, regParam=0.1, userCol="userId",itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")

#fit the model to training set
model=als.fit(training)

#make prediction

predictions = model.transform(test)

#Show some prediction

predictions.show(5)



# COMMAND ----------

# Evaluate the model using RMSE
evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="rating", metricName="rmse")
rmse = evaluator.evaluate(predictions)

print(f"Root-mean-square error = {rmse}")


# COMMAND ----------

#let's predict for userID 2

userId=2
user_ratings= o_rating_df.filter(o_rating_df.userId==userId)


#get Recommendation for the user

user_recommendation= model.recommendForUserSubset(user_ratings,5)
user_recommendation.show(5)

# COMMAND ----------

#Get top 10 recommendation for all user

all_recommemdation = model.recommendForAllUsers(10)
all_recommemdation.show()


# COMMAND ----------

#save the model

model.save("Movie_Recommendation_System")

# COMMAND ----------

# Load the saved model
from pyspark.ml.recommendation import ALSModel
loaded_model = ALSModel.load("dbfs:/Movie_Recommendation_System")
