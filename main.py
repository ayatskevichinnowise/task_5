import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, dense_rank, \
                                  coalesce, unix_timestamp, round, row_number
from pyspark.sql.window import Window
from dotenv import load_dotenv
import os

load_dotenv()

spark = SparkSession \
            .builder \
            .master('local[*]') \
            .appName('task5') \
            .getOrCreate()


def get_table(name: str) -> pyspark.sql.DataFrame:
    '''Function that gets tables from database
    :param name: name of table
    :return: table from database in DataFrame format
    '''
    return spark.read \
                .format('jdbc') \
                .option('url', os.getenv('URL')) \
                .option('dbtable', name) \
                .option('user', os.getenv('USER')) \
                .option('password', os.getenv('PASSWORD')) \
                .option('driver', 'org.postgresql.Driver') \
                .load()


# getting tables from db
category = get_table('category')
film = get_table('film')
film_actor = get_table('film_actor')
actor = get_table('actor')
inventory = get_table('inventory')
rental = get_table('rental')
film_category = get_table('film_category')
payment = get_table('payment')
customer = get_table('customer')
address = get_table('address')
city = get_table('city')


query_1 = category.join(film_category, on='category_id', how='left') \
                  .groupBy('name') \
                  .agg(count('film_id').alias('amount_of_films')) \
                  .select('name', 'amount_of_films') \
                  .orderBy(col('amount_of_films').desc())
query_1.show(truncate=False)


query_2 = actor \
            .join(film_actor, 'actor_id', 'inner') \
            .join(film, 'film_id', 'inner') \
            .join(inventory, 'film_id', 'inner') \
            .join(rental, 'inventory_id', 'inner') \
            .groupBy('actor_id', 'first_name', 'last_name') \
            .agg(count('rental_id').alias('amount_of_rent')) \
            .select('actor_id', 'first_name', 'last_name', 'amount_of_rent') \
            .orderBy(col('amount_of_rent').desc()) \
            .limit(10)
query_2.show(truncate=False)


query_3 = category \
            .join(film_category, 'category_id', 'inner') \
            .join(film, 'film_id', 'inner') \
            .join(inventory, 'film_id', 'inner') \
            .join(rental, 'inventory_id', 'inner') \
            .join(payment, on=['rental_id', 'customer_id'], how='left') \
            .groupBy('category_id', 'name') \
            .agg(sum('amount').alias('revenue')) \
            .select('category_id', 'name', 'revenue') \
            .orderBy(col('revenue').desc()) \
            .limit(1)
query_3.show(truncate=False)


query_4 = film \
            .join(inventory, 'film_id', 'left') \
            .filter(col('inventory_id').isNull()) \
            .select('title')
query_4.show(truncate=False)


query_5 = actor \
            .join(film_actor, 'actor_id', 'inner') \
            .join(film, 'film_id', 'inner') \
            .join(film_category, 'film_id', 'inner') \
            .join(category, 'category_id', 'inner') \
            .filter(col('name') == 'Children') \
            .groupBy('actor_id', 'first_name', 'last_name') \
            .agg(count('actor_id').alias('amount_of_titles')) \
            .withColumn('placement', dense_rank().over(Window
                        .orderBy(col('amount_of_titles').desc()))) \
            .filter(col('placement') < 4) \
            .select('actor_id', 'first_name', 'last_name',
                    'amount_of_titles', 'placement') \
            .orderBy(col('amount_of_titles').desc())
query_5.show(truncate=False)


query_6 = customer \
            .join(address, 'address_id', 'inner') \
            .join(city, 'city_id', 'inner') \
            .groupBy('city_id', 'city') \
            .agg(sum('active').alias('active_users'),
                 count('active').alias('total_users')) \
            .withColumn('non_active_users', col('total_users') -
                                            col('active_users')) \
            .select('city_id', 'city', 'non_active_users', 'active_users') \
            .orderBy(col('non_active_users').desc())
query_6.show(truncate=False)


# prepare some table for last query with amount of hours of rent
rental_time = rental \
    .withColumn('amount_of_hours', (coalesce(unix_timestamp('return_date'),
                                    unix_timestamp('last_update')) -
                                    unix_timestamp('rental_date')) / 3600) \
    .select('rental_id', 'inventory_id', 'customer_id', 'amount_of_hours') \
    .orderBy(col('amount_of_hours').desc())


query_7 = category \
            .join(film_category, 'category_id', 'inner') \
            .join(film, 'film_id', 'inner') \
            .join(inventory, 'film_id', 'inner') \
            .join(rental_time, 'inventory_id', 'inner') \
            .join(customer, 'customer_id', 'inner') \
            .join(address, 'address_id', 'inner') \
            .join(city, 'city_id', 'inner') \
            .filter(col('city').like('A%') | col('city').like(r'%-%')) \
            .groupBy('city_id', 'city', 'category_id', 'name') \
            .agg(round(sum('amount_of_hours'), 2).alias('amount_of_hours')) \
            .withColumn('placement', row_number().over(Window
                        .partitionBy(col('city_id'))
                        .orderBy(col('amount_of_hours').desc()))) \
            .filter(col('placement') == 1) \
            .select('city_id', 'city', 'category_id',
                    'name', 'amount_of_hours', 'placement')
query_7.show(truncate=False)
