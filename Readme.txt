Kaggle: Santander Value Prediction Challenge

Download train.csv and test.csv dataset from https://www.kaggle.com/c/santander-value-prediction-challenge 

Contents
 src - source code of project
 spearman.csv - Spearman Correlation values between target and each feature
		
Import project in IntellIJ IDEA

Run sbt task - "assembly" to build the fat jar.

Upload the following to amazon s3.
	- train.csv
	- test.csv
	- Built fat jar
	- spearman.csv from the extracted folder. 

In aws, start emr spark cluster. Create the following step
	Class takes four arguments - Input train file path, spearman file path,output directory path, input test file path
	Add step
		- Spark Application
		- Spark-submit options 
			provide --class Santander_Value_Predictor
		- Select the jar in s3 as application location
		- In arguments, provide input train file path, spearman file path,output directory path, input test file path.
			eg: s3://bigDataFolder/train.csv
			    s3://bigDataFolder/spearman.csv
			    s3://bigDataFolder/finalOutputFolder
			    s3://bigDataFolder/test.csv
		- Click Add
	After execution output will be saved in the following format.
		finalOutputFolder
			/metrics - will contain the text file with metrics of the best model
			/submission - will contain csv file with predictions on test data in kaggle contest appropriate format.



