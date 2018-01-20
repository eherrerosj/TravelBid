import os
import pandas as pd
import numpy as np
from scipy.spatial import kdtree
import _pickle as cPickle # do not touch
from sklearn.neighbors import NearestNeighbors
from math import cos, sin, pi, radians, asin, sqrt, ceil
from sklearn.preprocessing import normalize
import logging

logger = logging.getLogger()

def geo_spherical2cartesian(latitude, longitude):
    '''
    Note that both parameters must be in degrees
    '''
    d_center_earth = 6371 # assume earth is a sphere
    cos_theta = cos(float(latitude)*pi/180)
    sin_theta = sin(float(latitude)*pi/180)
    cos_phi = cos(float(longitude)*pi/180)
    sin_phi = sin(float(longitude)*pi/180)
    # Calculate cartesian coordinates
    x = d_center_earth * cos_theta * cos_phi
    y = d_center_earth * cos_theta * sin_phi
    z = d_center_earth * sin_theta
    return x, y, z

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the Earth (specified in decimal degrees)
    Example of use to calculate distance between 2 geo points:
        dist = haversine(52.3612646,4.8998865, 52.361300, 4.900700)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km

def load_hotels(hotels_csv_path):
    '''
    Read the csv of hotels
    '''
    global df_hotels
    df_hotels = pd.read_csv("./data/hotels.csv")


def process_hotels():
    global df_hotels
    df_hotels = df_hotels[(df_hotels.OverallRating > 6) & (df_hotels.Latitude > 0)].copy()
    df_hotels.drop_duplicates(inplace=True)
    df_hotels.reset_index(drop=True, inplace=True)
    
    if not os.path.exists("./data/"):
        os.makedirs("./data/")

    df_hotels.to_pickle("./data/df_hotels.pkl")


def index_hotels_geo():
    global df_hotels, hotels_kdtree
    df_hotels['geo_x'], df_hotels['geo_y'], df_hotels['geo_z'] = np.vectorize(geo_spherical2cartesian)(df_hotels.Latitude, df_hotels.Longitude)
    # KD-Tree https://en.wikipedia.org/wiki/K-d_tree
    cart_hotel_space = df_hotels[["geo_x", "geo_y", "geo_z"]]
    cart_hotel_space = [tuple(x) for x in cart_hotel_space.values]
    hotels_kdtree = kdtree.KDTree(cart_hotel_space)
    with open("./data/hotels_kdtree.pkl", 'wb') as handle:
        cPickle.dump(hotels_kdtree, handle)
    return hotels_kdtree


def top_n_neighbors(closest_hotels, input_landing_hotel_id, n, verbose=True):
    X_nn_test = closest_hotels[closest_hotels.HotelID != input_landing_hotel_id].copy()
    y_nn_test = closest_hotels[closest_hotels.HotelID == input_landing_hotel_id].copy()
    xy_nn_test = pd.concat([X_nn_test, y_nn_test], axis=0)
    if verbose:
        #display(y_nn_test)
        print("*"*20,"xy concat")
        print(xy_nn_test[["HotelName", "StarRating", "OverallRating"]])
    # Normalization
    X_nn_test_tf = normalize(xy_nn_test[["StarRating", "OverallRating"]],
                             axis=0)        
        
    neigh = NearestNeighbors(n_neighbors=n+1)
    neigh.fit(X_nn_test_tf)
    k_neighbors_d, k_neighbors_i = neigh.kneighbors(X_nn_test_tf)

    k_neighbors_d, k_neighbors_i = k_neighbors_d[0], k_neighbors_i[0]
    self_remove = np.argwhere( k_neighbors_i != xy_nn_test.shape[0]-1 )
    k_neighbors_d, k_neighbors_i = 100*(1-k_neighbors_d[self_remove].ravel()), k_neighbors_i[self_remove].ravel()
    results = xy_nn_test.iloc[k_neighbors_i,:]
    results.insert(0,"similarity",k_neighbors_d)
    if verbose:
        print("Calculating nearest neighbors...")
        print(k_neighbors_d, k_neighbors_i)
        print(len(k_neighbors_i),"elements")
        print("Recommendations from NN with concat normalized by col")
        print(results)
    return results


def top_n_kdtree(input_landing_hotel_id, n, verbose=False, rank_offset=15, max_ball_radius=5):
    latitude = df_hotels[df_hotels.HotelID == input_landing_hotel_id].Latitude.iloc[0]
    longitude = df_hotels[df_hotels.HotelID == input_landing_hotel_id].Longitude.iloc[0]
    x_input, y_input, z_input = geo_spherical2cartesian(float(latitude),float(longitude))
    hotel_name = df_hotels[df_hotels.HotelID == input_landing_hotel_id].HotelName.iloc[0]
    print("Finding recommendations for hotel " + hotel_name)
    # hotel selection bad to compare based on hotel density 
    (d_closest, i_closest) = hotels_kdtree.query((x_input, y_input, z_input), n+rank_offset)    
    i_closest_ball = hotels_kdtree.query_ball_point((x_input, y_input, z_input), max_ball_radius)     

    if verbose:
        print("The closest {} hotels are separated by these distances from the input hotel: {}".format(rank_offset + n, d_closest))
        print("{} hotels found for {}km around".format(len(i_closest_ball), max_ball_radius))
    if len(i_closest) > len(i_closest_ball):
        result = df_hotels.iloc[i_closest,:]
        result.insert(0,"distance_to_input_hotel",d_closest)
        if verbose: print("Using the {} hotels".format(len(i_closest)))    
    else:
        (d_closest, i_closest) = hotels_kdtree.query((x_input, y_input, z_input), len(i_closest_ball))
        result = df_hotels.iloc[i_closest,:]
        result.insert(0,"distance_to_input_hotel",d_closest)
        if verbose: print("Using {} hotels in a radius of {}km".format(len(i_closest_ball), max_ball_radius))
            
    # optional distance calculation, useful for potential corner cases
    result = top_n_neighbors(result, input_landing_hotel_id, n, verbose=verbose)
    

    
    # avoid landing hotel to be on the list
    result = result[result.HotelID != input_landing_hotel_id]
    result.rename(columns={'HotelID': 'recommended_hotel_id'}, inplace=True)
    result["HotelID"] = input_landing_hotel_id
    result = result[["HotelID", "recommended_hotel_id", "HotelName", "similarity", "distance_to_input_hotel"]].iloc[:n,]
    if verbose:
        print("This are the top hotels that are nearby:")
        print(result)
         
    
    return result