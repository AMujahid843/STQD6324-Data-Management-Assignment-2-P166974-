cd /home/maria_dev/stqd6324_assignment2
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
ls ml-100k
jps
# Create folder and subfolders
hdfs dfs -mkdir /movielens100k
hdfs dfs -mkdir /movielens100k/users
hdfs dfs -mkdir /movielens100k/movies
hdfs dfs -mkdir /movielens100k/ratings
hdfs dfs -ls /movielens100k
# Copy from local to HDFS
hdfs dfs -put ~/ml-100k/u.user /movielens100k/users/
hdfs dfs -put ~/ml-100k/u.item /movielens100k/movies/
hdfs dfs -put ~/ml-100k/u.data /movielens100k/ratings/
hdfs dfs -ls -R /movielens100k
# Display top 10 rows to verify
hdfs dfs -cat /movielens100k/users/u.user | head
hdfs dfs -cat /movielens100k/ratings/u.data | head
hdfs dfs -cat /movielens100k/movies/u.item | head
# Parse with PySpark
pyspark
users = spark.read.csv(
    "hdfs:///movielens100k/users/u.user",
    sep="|",
    inferSchema=True
)
users.show()
ratings = spark.read.csv(
    "hdfs:///movielens100k/ratings/u.data",
    sep="\t",
    inferSchema=True
)
ratings.show()
movies = spark.read.csv(
    "hdfs:///movielens100k/movies/u.item",
    sep="|",
    inferSchema=True
)
movies.show()
# Create RDDs
users_rdd = sc.textFile(
    "hdfs:///movielens100k/users/u.user"
)
users_rdd.take(5)
ratings_rdd = sc.textFile(
    "hdfs:///movielens100k/ratings/u.data"
)
ratings_rdd.take(5)
movies_rdd = sc.textFile(
    "hdfs:///movielens100k/movies/u.item"
)
movies_rdd.take(5)
users_parsed_rdd = users_rdd.map(
    lambda x: x.split("|")
)
users_clean_rdd = users_parsed_rdd.map(
    lambda x: (
        int(x[0]),
        int(x[1]),
        x[2],
        x[3],
        x[4]
    )
)
# Transform RDDs into DataFrame
users_df = spark.createDataFrame(
    users_clean_rdd,
    [
        "user_id",
        "age",
        "gender",
        "occupation",
        "zip_code"
    ]
)
users_df.show(10)
ratings_parsed_rdd = ratings_rdd.map(
    lambda x: x.split("\t")
)
ratings_clean_rdd = ratings_parsed_rdd.map(
    lambda x: (
        int(x[0]),
        int(x[1]),
        int(x[2]),
        int(x[3])
    )
)
ratings_df = spark.createDataFrame(
    ratings_clean_rdd,
    [
        "user_id",
        "movie_id",
        "rating",
        "timestamp"
    ]
)
ratings_df.show(10)
movies_parsed_rdd = movies_rdd.map(
    lambda x: x.split("|")
)
movies_clean_rdd = movies_parsed_rdd.map(
    lambda x: (
        int(x[0]),     # movie_id
        x[1],          # title
        x[2],          # release_date
        x[3],          # video_release_date
        x[4],          # IMDb_URL
        int(x[5]),     # unknown
        int(x[6]),     # Action
        int(x[7]),     # Adventure
        int(x[8]),     # Animation
        int(x[9]),     # Children
        int(x[10]),    # Comedy
        int(x[11]),    # Crime
        int(x[12]),    # Documentary
        int(x[13]),    # Drama
        int(x[14]),    # Fantasy
        int(x[15]),    # Film-Noir
        int(x[16]),    # Horror
        int(x[17]),    # Musical
        int(x[18]),    # Mystery
        int(x[19]),    # Romance
        int(x[20]),    # Sci-Fi
        int(x[21]),    # Thriller
        int(x[22]),    # War
        int(x[23])     # Western
    )
)
movies_df = spark.createDataFrame(
    movies_clean_rdd,
    [
        "movie_id",
        "title",
        "release_date",
        "video_release_date",
        "IMDb_URL",
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "FilmNoir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "SciFi",
        "Thriller",
        "War",
        "Western"
    ]
)
movies_df.show(10)
# Data Cleaning and Preprocessing
users_df = users_df.dropna()
ratings_df = ratings_df.dropna()
movies_df = movies_df.dropna()
users_df = users_df.dropDuplicates()
ratings_df = ratings_df.dropDuplicates()
movies_df = movies_df.dropDuplicates()
from pyspark.sql.functions import col
ratings_df = ratings_df \
    .withColumn("user_id", col("user_id").cast("int")) \
    .withColumn("movie_id", col("movie_id").cast("int")) \
    .withColumn("rating", col("rating").cast("float")) \
    .withColumn("timestamp", col("timestamp").cast("int"))
users_df = users_df \
    .withColumn("user_id", col("user_id").cast("int")) \
    .withColumn("age", col("age").cast("int"))
movies_df = movies_df \
    .withColumn("movie_id", col("movie_id").cast("int"))
print("Users:", users_df.count())
print("Ratings:", ratings_df.count())
print("Movies:", movies_df.count())
# Analytical tasks
ratings_df.createOrReplaceTempView("ratings")
movies_df.createOrReplaceTempView("movies")
users_df.createOrReplaceTempView("users")
# 1. Movie Average Rating
movie_avg_rating_df = spark.sql("""
SELECT
    movie_id,
    AVG(rating) AS avg_rating
FROM ratings
GROUP BY movie_id
""")
movie_avg_rating_df.show(10)
# 2. Movie Average Rating with Title
movie_avg_rating_with_title_df = spark.sql("""
SELECT
    r.movie_id,
    m.title,
    AVG(r.rating) AS avg_rating
FROM ratings r
JOIN movies m
ON r.movie_id = m.movie_id
GROUP BY r.movie_id, m.title
""")
movie_avg_rating_with_title_df.show(10)
# 3. Top 10 Movies
top_10_movies_df = spark.sql("""
SELECT
    r.movie_id,
    m.title,
    AVG(r.rating) AS avg_rating
FROM ratings r
JOIN movies m
ON r.movie_id = m.movie_id
GROUP BY r.movie_id, m.title
ORDER BY avg_rating DESC
LIMIT 10
""")
top_10_movies_df.show()
# 4. Top 10 Movies with Rating Counts
top_10_movies_ratingcounts_df = spark.sql("""
SELECT
    r.movie_id,
    m.title,
    AVG(r.rating) AS avg_rating,
    COUNT(r.rating) AS rating_count
FROM ratings r
JOIN movies m
ON r.movie_id = m.movie_id
GROUP BY r.movie_id, m.title
ORDER BY avg_rating DESC, rating_count DESC
LIMIT 10
""")
top_10_movies_ratingcounts_df.show()
# 5. Active Users
active_users_df = spark.sql("""
SELECT
    user_id,
    COUNT(*) AS total_ratings
FROM ratings
GROUP BY user_id
HAVING COUNT(*) >= 50
""")
active_users_df.show(10)
active_users_count = active_users_df.count()
print("Total users who have rated at least 50 movies:", active_users_count, "users")
spark.sql("""
SELECT movie_id, title, genre, flag
FROM (
    SELECT movie_id, title,
    stack(19,
        'Action', Action,
        'Adventure', Adventure,
        'Animation', Animation,
        'Children', Children,
        'Comedy', Comedy,
        'Crime', Crime,
        'Documentary', Documentary,
        'Drama', Drama,
        'Fantasy', Fantasy,
        'FilmNoir', FilmNoir,
        'Horror', Horror,
        'Musical', Musical,
        'Mystery', Mystery,
        'Romance', Romance,
        'SciFi', SciFi,
        'Thriller', Thriller,
        'War', War,
        'Western', Western
    ) AS (genre, flag)
    FROM movies
) t
WHERE flag = 1
""").createOrReplaceTempView("movie_genres")
spark.sql("""
SELECT
    r.user_id,
    mg.genre
FROM ratings r
JOIN movie_genres mg
ON r.movie_id = mg.movie_id
""").createOrReplaceTempView("user_genres")
active_users_df.createOrReplaceTempView("active_users")
spark.sql("""
SELECT
    ug.user_id,
    ug.genre
FROM user_genres ug
JOIN active_users au
ON ug.user_id = au.user_id
""").createOrReplaceTempView("active_user_genres")
spark.sql("""
SELECT
    user_id,
    genre,
    COUNT(*) AS freq
FROM active_user_genres
GROUP BY user_id, genre
""").createOrReplaceTempView("genre_count")
# 6. Favourite Genre
favourite_genre_df = spark.sql("""
SELECT user_id, genre, freq
FROM (
    SELECT
        user_id,
        genre,
        freq,
        ROW_NUMBER() OVER (
            PARTITION BY user_id
            ORDER BY freq DESC
        ) AS rn
    FROM genre_count
) t
WHERE rn = 1
ORDER BY freq DESC
""")
favourite_genre_df.show()
# 7. Young Users
young_users_df = spark.sql("""
SELECT *
FROM users
WHERE age < 20
""")
young_users_df.show()
young_users_count = young_users_df.count()
print("Total users less than 20 years old:", young_users_count, "users")
# 8. Scientists aged 30-40
scientist_30_40_df = spark.sql("""
SELECT *
FROM users
WHERE occupation = 'scientist'
AND age BETWEEN 30 AND 40
""")
scientist_30_40_df.show()
scientist_30_40_count = scientist_30_40_df.count()
print("Scientists aged 30 to 40 years old:", scientist_30_40_count, "users")
# Save in CSV file to local filesystem in Ambari
users_df.coalesce(1).write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("/home/maria_dev/stqd6324_assignment2/users")
ratings_df.coalesce(1).write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("/home/maria_dev/stqd6324_assignment2/ratings")
movies_df.coalesce(1).write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("/home/maria_dev/stqd6324_assignment2/movies")
movie_avg_rating_df.coalesce(1).write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("/home/maria_dev/stqd6324_assignment2/movie_avg_rating")
movie_avg_rating_with_title_df.coalesce(1).write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("/home/maria_dev/stqd6324_assignment2/movie_avg_rating_with_title")
top_10_movies_df.coalesce(1).write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("/home/maria_dev/stqd6324_assignment2/top_10_movies")
top_10_movies_ratingcounts_df.coalesce(1).write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("/home/maria_dev/stqd6324_assignment2/top_10_movies_ratingcounts")
active_users_df.coalesce(1).write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("/home/maria_dev/stqd6324_assignment2/active_users")
favourite_genre_df.coalesce(1).write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("/home/maria_dev/stqd6324_assignment2/favourite_genre")
young_users_df.coalesce(1).write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("/home/maria_dev/stqd6324_assignment2/young_users")
scientist_30_40_df.coalesce(1).write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("/home/maria_dev/stqd6324_assignment2/scientist_30_40")
# Create empty Keyspace and Tables in Cassandra
sudo service cassandra start
cqlsh
-- Create Keyspace called stqd6324_assignment2
CREATE KEYSPACE IF NOT EXISTS stqd6324_assignment2
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};
USE stqd6324_assignment2;
-- 1. Users Table
CREATE TABLE IF NOT EXISTS users (
    user_id INT PRIMARY KEY,
    age INT,
    gender TEXT,
    occupation TEXT,
    zip_code TEXT
 );
-- 2. Movies Table
CREATE TABLE IF NOT EXISTS movies (
    movie_id INT PRIMARY KEY,
    title TEXT,
    release_date TEXT,
    video_release_date TEXT,
    imdb_url TEXT,
    unknown INT,
    Action INT,
    Adventure INT,
    Animation INT,
    Children INT,
    Comedy INT,
    Crime INT,
    Documentary INT,
    Drama INT,
    Fantasy INT,
    FilmNoir INT,
    Horror INT,	
    Musical INT,
    Mystery INT,
    Romance INT,
    SciFi INT,
    Thriller INT,
    War INT,
    Western INT
);
-- 3. Ratings Table
CREATE TABLE IF NOT EXISTS ratings (
    user_id INT,
    movie_id INT,
    rating FLOAT,
    timestamp INT,
    PRIMARY KEY (user_id, movie_id)
);
-- 4. Movie Average Rating Table
CREATE TABLE IF NOT EXISTS movie_avg_rating (
    movie_id INT PRIMARY KEY,
    avg_rating DOUBLE
);
-- 5. Movie Average Rating with Title Table
CREATE TABLE IF NOT EXISTS movie_avg_rating_with_title (
    movie_id INT PRIMARY KEY,
    title TEXT,
    avg_rating DOUBLE
);
-- 6. Top 10 Movies Table
CREATE TABLE IF NOT EXISTS top_10_movies (
    movie_id INT PRIMARY KEY,
    title TEXT,
    avg_rating DOUBLE
);
-- 7. Top 10 Movies with Rating Counts Table
CREATE TABLE IF NOT EXISTS top_10_movies_ratingcounts (
    movie_id INT PRIMARY KEY,
    title TEXT,
    avg_rating DOUBLE,
    rating_count BIGINT
);
-- 8. Active Users Table
CREATE TABLE IF NOT EXISTS active_users (
    user_id INT PRIMARY KEY,
    total_ratings BIGINT
);
-- 9. Favourite Genre Table
CREATE TABLE IF NOT EXISTS favourite_genre (
    user_id INT PRIMARY KEY,
    genre TEXT,
    freq BIGINT
);
-- 10. Young Users Table
CREATE TABLE IF NOT EXISTS young_users (
    user_id INT PRIMARY KEY,
    age INT,
    gender TEXT,
    occupation TEXT,
    zip_code TEXT
);
-- 11. Scientists Aged 30 to 40 Table
CREATE TABLE IF NOT EXISTS scientist_30_40 (
    user_id INT PRIMARY KEY,
    age INT,
    gender TEXT,
    occupation TEXT,
    zip_code TEXT
);
exit
# Move the Spark DataFrame into Cassandra Keyspace and Tables
vi savetocassandra.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("STQD6324_Assignment2") \
    .config("spark.cassandra.connection.host", "127.0.0.1") \
    .config("spark.cassandra.connection.port", "9042") \
    .getOrCreate()

# 1. Users
users_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="users", keyspace="stqd6324_assignment2") \
    .save()
# 2. Ratings
ratings_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="ratings", keyspace="stqd6324_assignment2") \
    .save()
# Force every column name in the DataFrame to be lowercase
for col_name in movies_df.columns:
    movies_df = movies_df.withColumnRenamed(col_name, col_name.lower())
# 3. Movies
movies_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="movies", keyspace="stqd6324_assignment2") \
    .save()
# 4. Movie Average Rating
movie_avg_rating_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="movie_avg_rating", keyspace="stqd6324_assignment2") \
    .save()
# 5. Movie Average Rating with Title
movie_avg_rating_with_title_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="movie_avg_rating_with_title", keyspace="stqd6324_assignment2") \
    .save()
# 6. Top 10 Movies
top_10_movies_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="top_10_movies", keyspace="stqd6324_assignment2") \
    .save()
# 7. Top 10 Movies with Rating Counts
top_10_movies_ratingcounts_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="top_10_movies_ratingcounts", keyspace="stqd6324_assignment2") \
    .save()
# 8. Active Users
active_users_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="active_users", keyspace="stqd6324_assignment2") \
    .save()
# 9. Favourite Genre
favourite_genre_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="favourite_genre", keyspace="stqd6324_assignment2") \
    .save()
# 10. Young Users
young_users_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="young_users", keyspace="stqd6324_assignment2") \
    .save()
# 11. Scientists aged 30-40
scientist_30_40_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="scientist_30_40", keyspace="stqd6324_assignment2") \
    .save()

spark-submit --packages com.datastax.spark:spark-cassandra-connector_2.11:2.4.3 savetocassandra.py

vi stqd6324_assignment2.py
from pyspark.sql import SparkSession

# 1. Initialize SparkSession
spark = SparkSession.builder \
    .appName("STQD6324_Assignment2") \
    .config("spark.cassandra.connection.host", "127.0.0.1") \
    .config("spark.cassandra.connection.port", "9042") \
    .getOrCreate()

# 2. Define 'sc'
sc = spark.sparkContext
# Data Loading, Parsing and Cleaning
users = spark.read.csv(
    "hdfs:///movielens100k/users/u.user",
    sep="|",
    inferSchema=True
)
users.show()
ratings = spark.read.csv(
    "hdfs:///movielens100k/ratings/u.data",
    sep="\t",
    inferSchema=True
)
ratings.show()
movies = spark.read.csv(
    "hdfs:///movielens100k/movies/u.item",
    sep="|",
    inferSchema=True
)
movies.show()
users_rdd = sc.textFile(
    "hdfs:///movielens100k/users/u.user"
)
users_rdd.take(5)
ratings_rdd = sc.textFile(
    "hdfs:///movielens100k/ratings/u.data"
)
ratings_rdd.take(5)
movies_rdd = sc.textFile(
    "hdfs:///movielens100k/movies/u.item"
)
movies_rdd.take(5)
users_parsed_rdd = users_rdd.map(
    lambda x: x.split("|")
)
users_clean_rdd = users_parsed_rdd.map(
    lambda x: (
        int(x[0]),
        int(x[1]),
        x[2],
        x[3],
        x[4]
    )
)
users_df = spark.createDataFrame(
    users_clean_rdd,
    [
        "user_id",
        "age",
        "gender",
        "occupation",
        "zip_code"
    ]
)
users_df.show(10)
ratings_parsed_rdd = ratings_rdd.map(
    lambda x: x.split("\t")
)
ratings_clean_rdd = ratings_parsed_rdd.map(
    lambda x: (
        int(x[0]),
        int(x[1]),
        int(x[2]),
        int(x[3])
    )
)
ratings_df = spark.createDataFrame(
    ratings_clean_rdd,
    [
        "user_id",
        "movie_id",
        "rating",
        "timestamp"
    ]
)
ratings_df.show(10)
movies_parsed_rdd = movies_rdd.map(
    lambda x: x.split("|")
)
movies_clean_rdd = movies_parsed_rdd.map(
    lambda x: (
        int(x[0]),     # movie_id
        x[1],          # title
        x[2],          # release_date
        x[3],          # video_release_date
        x[4],          # IMDb_URL
        int(x[5]),     # unknown
        int(x[6]),     # Action
        int(x[7]),     # Adventure
        int(x[8]),     # Animation
        int(x[9]),     # Children
        int(x[10]),    # Comedy
        int(x[11]),    # Crime
        int(x[12]),    # Documentary
        int(x[13]),    # Drama
        int(x[14]),    # Fantasy
        int(x[15]),    # Film-Noir
        int(x[16]),    # Horror
        int(x[17]),    # Musical
        int(x[18]),    # Mystery
        int(x[19]),    # Romance
        int(x[20]),    # Sci-Fi
        int(x[21]),    # Thriller
        int(x[22]),    # War
        int(x[23])     # Western
    )
)
movies_df = spark.createDataFrame(
    movies_clean_rdd,
    [
        "movie_id",
        "title",
        "release_date",
        "video_release_date",
        "IMDb_URL",
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "FilmNoir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "SciFi",
        "Thriller",
        "War",
        "Western"
    ]
)
movies_df.show(10)
users_df = users_df.dropna()
ratings_df = ratings_df.dropna()
movies_df = movies_df.dropna()
users_df = users_df.dropDuplicates()
ratings_df = ratings_df.dropDuplicates()
movies_df = movies_df.dropDuplicates()
from pyspark.sql.functions import col
ratings_df = ratings_df \
    .withColumn("user_id", col("user_id").cast("int")) \
    .withColumn("movie_id", col("movie_id").cast("int")) \
    .withColumn("rating", col("rating").cast("float")) \
    .withColumn("timestamp", col("timestamp").cast("int"))
users_df = users_df \
    .withColumn("user_id", col("user_id").cast("int")) \
    .withColumn("age", col("age").cast("int"))
movies_df = movies_df \
    .withColumn("movie_id", col("movie_id").cast("int"))
print("Users:", users_df.count())
print("Ratings:", ratings_df.count())
print("Movies:", movies_df.count())
# Analytical tasks
ratings_df.createOrReplaceTempView("ratings")
movies_df.createOrReplaceTempView("movies")
users_df.createOrReplaceTempView("users")
# 1. Movie Average Rating
movie_avg_rating_df = spark.sql("""
SELECT
    movie_id,
    AVG(rating) AS avg_rating
FROM ratings
GROUP BY movie_id
""")
movie_avg_rating_df.show(10)
# 2. Movie Average Rating with Title
movie_avg_rating_with_title_df = spark.sql("""
SELECT
    r.movie_id,
    m.title,
    AVG(r.rating) AS avg_rating
FROM ratings r
JOIN movies m
ON r.movie_id = m.movie_id
GROUP BY r.movie_id, m.title
""")
movie_avg_rating_with_title_df.show(10)
# 3. Top 10 Movies
top_10_movies_df = spark.sql("""
SELECT
    r.movie_id,
    m.title,
    AVG(r.rating) AS avg_rating
FROM ratings r
JOIN movies m
ON r.movie_id = m.movie_id
GROUP BY r.movie_id, m.title
ORDER BY avg_rating DESC
LIMIT 10
""")
top_10_movies_df.show()
# 4. Top 10 Movies with Rating Counts
top_10_movies_ratingcounts_df = spark.sql("""
SELECT
    r.movie_id,
    m.title,
    AVG(r.rating) AS avg_rating,
    COUNT(r.rating) AS rating_count
FROM ratings r
JOIN movies m
ON r.movie_id = m.movie_id
GROUP BY r.movie_id, m.title
ORDER BY avg_rating DESC, rating_count DESC
LIMIT 10
""")
top_10_movies_ratingcounts_df.show()
# 5. Active Users
active_users_df = spark.sql("""
SELECT
    user_id,
    COUNT(*) AS total_ratings
FROM ratings
GROUP BY user_id
HAVING COUNT(*) >= 50
""")
active_users_df.show(10)
active_users_count = active_users_df.count()
print("Total users who have rated at least 50 movies:", active_users_count, "users")
spark.sql("""
SELECT movie_id, title, genre, flag
FROM (
    SELECT movie_id, title,
    stack(19,
        'Action', Action,
        'Adventure', Adventure,
        'Animation', Animation,
        'Children', Children,
        'Comedy', Comedy,
        'Crime', Crime,
        'Documentary', Documentary,
        'Drama', Drama,
        'Fantasy', Fantasy,
        'FilmNoir', FilmNoir,
        'Horror', Horror,
        'Musical', Musical,
        'Mystery', Mystery,
        'Romance', Romance,
        'SciFi', SciFi,
        'Thriller', Thriller,
        'War', War,
        'Western', Western
    ) AS (genre, flag)
    FROM movies
) t
WHERE flag = 1
""").createOrReplaceTempView("movie_genres")
spark.sql("""
SELECT
    r.user_id,
    mg.genre
FROM ratings r
JOIN movie_genres mg
ON r.movie_id = mg.movie_id
""").createOrReplaceTempView("user_genres")
active_users_df.createOrReplaceTempView("active_users")
spark.sql("""
SELECT
    ug.user_id,
    ug.genre
FROM user_genres ug
JOIN active_users au
ON ug.user_id = au.user_id
""").createOrReplaceTempView("active_user_genres")
spark.sql("""
SELECT
    user_id,
    genre,
    COUNT(*) AS freq
FROM active_user_genres
GROUP BY user_id, genre
""").createOrReplaceTempView("genre_count")
# 6. Favourite Genre
favourite_genre_df = spark.sql("""
SELECT user_id, genre, freq
FROM (
    SELECT
        user_id,
        genre,
        freq,
        ROW_NUMBER() OVER (
            PARTITION BY user_id
            ORDER BY freq DESC
        ) AS rn
    FROM genre_count
) t
WHERE rn = 1
ORDER BY freq DESC
""")
favourite_genre_df.show()
# 7. Young Users
young_users_df = spark.sql("""
SELECT *
FROM users
WHERE age < 20
""")
young_users_df.show()
young_users_count = young_users_df.count()
print("Total users less than 20 years old:", young_users_count, "users")
# 8. Scientists aged 30-40
scientist_30_40_df = spark.sql("""
SELECT *
FROM users
WHERE occupation = 'scientist'
AND age BETWEEN 30 AND 40
""")
scientist_30_40_df.show()
scientist_30_40_count = scientist_30_40_df.count()
print("Scientists aged 30 to 40 years old:", scientist_30_40_count, "users")
# Save into Cassandra
# 1. Users
users_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="users", keyspace="stqd6324_assignment2") \
    .save()
# 2. Ratings
ratings_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="ratings", keyspace="stqd6324_assignment2") \
    .save()
# 3. Movies
movies_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="movies", keyspace="stqd6324_assignment2") \
    .save()
# 4. Movie Average Rating
movie_avg_rating_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="movie_avg_rating", keyspace="stqd6324_assignment2") \
    .save()
# 5. Movie Average Rating with Title
movie_avg_rating_with_title_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="movie_avg_rating_with_title", keyspace="stqd6324_assignment2") \
    .save()
# 6. Top 10 Movies
top_10_movies_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="top_10_movies", keyspace="stqd6324_assignment2") \
    .save()
# 7. Top 10 Movies with Rating Counts
top_10_movies_ratingcounts_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="top_10_movies_ratingcounts", keyspace="stqd6324_assignment2") \
    .save()
# 8. Active Users
active_users_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="active_users", keyspace="stqd6324_assignment2") \
    .save()
# 9. Favourite Genre
favourite_genre_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="favourite_genre", keyspace="stqd6324_assignment2") \
    .save()
# 10. Young Users
young_users_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="young_users", keyspace="stqd6324_assignment2") \
    .save()
# 11. Scientists aged 30-40
scientist_30_40_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="scientist_30_40", keyspace="stqd6324_assignment2") \
    .save()

spark-submit --packages com.datastax.spark:spark-cassandra-connector_2.11:2.4.3 stqd6324_assignment2.py
# Verify in Cassandra
SELECT * FROM movies LIMIT 5;
SELECT * FROM ratings LIMIT 10;
SELECT * FROM users LIMIT 10;
SELECT * FROM movie_avg_rating LIMIT 10;
SELECT * FROM movie_avg_rating_with_title LIMIT 10;
SELECT * FROM top_10_movies;
SELECT * FROM top_10_movies_ratingcounts;
SELECT * FROM active_users LIMIT 10;
SELECT * FROM favourite_genre LIMIT 10;
SELECT * FROM young_users LIMIT 10;
SELECT * FROM scientist_30_40 LIMIT 10;
exit()

# Read the Cassandra tables back into Spark DataFrames for validation purposes
vi validate_cassandra.py
from pyspark.sql import SparkSession

# 1. Initialize the Spark Session with Cassandra configurations
spark = SparkSession.builder \
    .appName("Cassandra_Validation_Script") \
    .config("spark.cassandra.connection.host", "127.0.0.1") \
    .config("spark.cassandra.connection.port", "9042") \
    .getOrCreate()

print("\n=============================================")
print("   STARTING CASSANDRA DATA VALIDATION")
print("=============================================\n")

# 2. List of all 11 tables to verify
tables_to_validate = [
    "users",
    "ratings",
    "movies",
    "movie_avg_rating",
    "movie_avg_rating_with_title",
    "top_10_movies",
    "top_10_movies_ratingcounts",
    "active_users",
    "favourite_genre",
    "young_users",
    "scientist_30_40"
]

# 3. Loop through each table, read it back, and print validation stats
for table_name in tables_to_validate:
    print("---------------------------------------------")
    print("Checking Table: " + str(table_name))
    print("---------------------------------------------")
    
    try:
        # Read table from Cassandra back into a Spark DataFrame
        df = spark.read \
            .format("org.apache.spark.sql.cassandra") \
            .options(table=table_name, keyspace="stqd6324_assignment2") \
            .load()
        
        # Get row count
        total_rows = df.count()
        print(" STATUS: Success")
        print(" TOTAL ROWS: " + str(total_rows))
        
        # Print Schema
        print(" SCHEMA:")
        df.printSchema()
        
        # Display a 10-row data preview
        print(" DATA PREVIEW (Top 10 Rows):")
        df.show(10, truncate=False)
        print("\n")
        
    except Exception as e:
        print(" STATUS: Failed to read table")
        print(" ERROR MESSAGE: " + str(e) + "\n")

print("=============================================")
print("   VALIDATION COMPLETED")
print("=============================================")

# 4. Stop the Spark session cleanly
spark.stop()
spark-submit --packages com.datastax.spark:spark-cassandra-connector_2.11:2.4.3 validate_cassandra.py
