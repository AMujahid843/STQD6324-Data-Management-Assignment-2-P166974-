# STQD6324-Data-Management-Assignment-2-P166974-

## Software Requirements
The project was developed and tested using the following software versions:
- Python: 3.12.13
- Apache Spark: 4.0.2
- PySpark: 4.0.2
- Cassandra: 5.x
- MongoDB: 8.0
- PyMongo: 4.17.0

Using the same or compatible versions is recommended to ensure reproducible results.

## Dataset

- Access the MovieLens 100k Dataset from https://grouplens.org/datasets/movielens/
- Download the "ml-100k.zip" folder
- Extract the zip folder
- Only three data files will be used (u.user, u.data and u.item)

## Technologies

- Apache Spark
- Cassandra
- PySpark
- MongoDB
- PyMongo

## How to Run
1. Start Cassandra
2. Start Spark
3. Upload dataset to HDFS
4. Open notebook
5. Run cells sequentially

## Results
The pipeline performs five analytical tasks:
1. Calculate the average rating for each movie.
2. Identify top ten movies with the highest average ratings.
3. Identify users who have rated at least 50 movies and determine their favourite movie
genre based on the genre they rated most frequently.
4. Find all users who are less than 20 years old.
5. Find all users whose occupation is “scientist” and whose age is between 30 and 40 years
old.
