# Sort movies according to their duration in descending order.
import pandas as pd

movie_catalogue.head()
movie_catalogue['num_duration'] = movie_catalogue['duration'].str.extract('(\d+)')
movie_catalogue['num_duration'] = movie_catalogue['num_duration'].astype(int)
movie_catalogue = movie_catalogue.sort_values(by = 'num_duration', ascending = False)
movie_catalogue.drop('num_duration', axis = 1, inplace = True)
movie_catalogue.head(10)
