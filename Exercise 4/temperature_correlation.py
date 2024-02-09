"""
python3 temperature_correlation.py stations.json.gz city_data.csv output.svg
"""
import numpy as np
import pandas as pd
import sys
import math
import matplotlib.pyplot as plt


def haversine_dataframe(coord1, coord2, unit='mi'):
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    if lat1 > 90.0 or lat1 < -90.0:
        raise Exception(f"Latitude(s) out of range [-90,90]")
    if lon1 > 180.0 or lon1 < -180.0:
        raise Exception(f"Longitude(s) out of range [-180,180]")

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    lon1 = math.radians(lon1)
    lon2 = math.radians(lon2)

    if unit == "mi":
        world_radius = 3963
    elif unit == "km":
        world_radius = 6371
    else:
        raise Exception(f"Invalid unit. Should be 'mi' - miles or 'km' - kilometres ")

    lat_dist = lat2 - lat1
    lon_dist = lon2 - lon1
    a = math.sin((lat_dist/2)**2) + (math.cos(lat1) * math.cos(lat2) * math.sin((lon_dist/2)**2))
    c = 2 * math.asin(math.sqrt(a))
    distance = world_radius * c

    return distance

"""
Returns distance of one city to all stations.
- city: a city
- stations: data of all stations
"""
def distance(city, stations):
    city_lat, city_lon = city
    result = stations.apply(lambda station:
                   haversine_dataframe((city_lat, city_lon),
                                        (station['latitude'], station['longitude']),
                                        "mi"), axis=1)
    return result

"""
For each city, find the distances between it and all stations
- cities: data of all cities
- stations: data of all stations
"""
def all_distances(cities, stations):
    result = cities.apply(lambda city: distance((city['latitude'], city['longitude']), stations), axis=1)
    return result


"""
Find the closest station to each city
- cities: all city data
- distances: all distances data
"""
def closest_station(cities, distances):
    cities['station'] = distances.idxmin(axis=1)
    cities['distance'] = distances.min(axis=1)
    return

"""
Takes in a city and find the tmax of the station based on the city's closest station.
- city: one city index
- stations: all stations data
"""
def best_tmax(city, stations):
    station = stations.loc[city['station'],['avg_tmax']]
    return station

def main():
    # Load the data
    file_stations = sys.argv[1]
    file_cities = sys.argv[2]

    # Read the data
    data_stations = pd.read_json(file_stations, lines=True)
    data_cities = pd.read_csv(file_cities, index_col='name')

    # Divide original degrees by 10
    data_stations.set_index('station', inplace=True)
    data_stations['avg_tmax'] = data_stations['avg_tmax'].div(10)

    # Drop missing values, delete those rows
    data_cities.dropna(subset=['population'], inplace=True)
    data_cities.dropna(subset=['area'], inplace=True)

    # Convert area m^2 to km^2
    data_cities['area'] = data_cities['area'] * 0.000001

    # Drop areas > 10000km^2
    data_cities['too_large'] = data_cities['area'].ge(10000)
    data_cities['too_large'].replace(True, np.nan, inplace=True)
    data_cities.dropna(subset='too_large', inplace=True)

    # Find population density
    data_cities['density'] = data_cities['population'].div(data_cities['area'])

    # Find the weather station closest to each city
    distances = all_distances(data_cities, data_stations)
    closest_station(data_cities, distances)
    data_cities['avg_tmax'] = data_cities.apply(lambda city: best_tmax(city, stations=data_stations), axis=1)

    # Plot the average max. temp x population density
    plt.figure(figsize=(10,5))
    plt.scatter(data_cities['avg_tmax'],data_cities['density'], s = 2)
    plt.title('Temperature vs. Population Density')
    plt.xlabel('Average Max Temperature (\u00b0C)')
    plt.ylabel('Population Density (people/km\u00b2)')
    plt.show()

