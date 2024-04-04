import xml.dom.minidom

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import statsmodels.api as sm
import sys
from pykalman import KalmanFilter
import xml.dom.minidom
from xml.dom.minidom import getDOMImplementation

def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
        trkseg.appendChild(trkpt)
    
    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    
    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)
    
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')
"""
1. Extract latitude and longitude from each <trkpt> element
    - Ignore elevation, time, other fields.
2. Create DataFrame with columns: 'lat', 'lon' holding observations

Calculate Distances:
1. Haversine Formula 
Get the distances using latitude and longitude points

Write Function 'distance()'
1. Take the DataFrame and return the distance (in metres)
between the latitude and longitude points
DataFrame.shift

Print the literal distance described in the GPX file, round 2 dec. places
"""

def get_data(file):
    dom = xml.dom.minidom
    parsed_data = dom.parse(file)
    points = pd.DataFrame(columns=["lat","lon"])

    trkpt_data = parsed_data.getElementsByTagName('trkpt')
    for line in trkpt_data:
        points.loc[len(points)] = pd.Series({'lat':line.getAttribute('lat'), 'lon':line.getAttribute('lon')})

    points['lat'] = points['lat'].values.astype(float)
    points['lon'] = points['lon'].values.astype(float)

    return points

"""
Reference:
https://www.themathdoctors.org/distances-on-earth-2-the-haversine-formula/
"""
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

def distance(points, unit="mi"):

    # Create copies of columns that are shifted
    points['lat_shift'] = points['lat'].shift(-1)
    points['lon_shift'] = points['lon'].shift(-1)
    points['distance'] = points.apply(lambda r: haversine_dataframe( (r['lat'], r['lon']) ,(r['lat_shift'], r['lon_shift']), unit ) ,axis=1)
    points['distance'].fillna(0, inplace=True)

    return points


def smooth_points(points):

    kalman_data = points[['lat', 'lon']]
    lat_std = kalman_data['lat'].std()
    lon_std = kalman_data['lon'].std()

    initial_state = kalman_data.iloc[0]
    observation_covariance = np.diag([.005,
                                      .005]) ** 2

    transition_covariance = np.diag([lat_std,
                                     lon_std]) ** 2

    transition = [[1,0],       #lat
                  [0,1]]           #lon

    kf = KalmanFilter(
        initial_state_mean=initial_state,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition
    )
    kalman_smoothed, _ = kf.smooth(kalman_data)

    return pd.DataFrame(kalman_smoothed, columns=['lat','lon'])


def main():
    points = get_data(sys.argv[1])


    # Unsmoothed points, calculate distances
    points_unsmoothed = distance(points, "km")

    # Smooth points, calculate distances
    points_smoothed = smooth_points(points)
    points_smoothed = distance(points_smoothed, "km")

    print('Uniltered distance: %0.2f' % (points_unsmoothed['distance'].values.sum()))
    print('Filtered distance: %0.2f' % (points_smoothed['distance'].values.sum()))

    plt.figure(figsize=(6,4))
    plt.scatter(points_unsmoothed['lon'], points_unsmoothed['lat'], s=1, label='unsmoothed')
    plt.plot(points_smoothed['lon'], points_smoothed['lat'],'-g', label='smoothed')
    plt.title("GPS Tracking: Smoothing with Kalman Filter")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.show()

    output_gpx(points_smoothed, "out.gpx")

    return

if __name__ == '__main__':
    main()