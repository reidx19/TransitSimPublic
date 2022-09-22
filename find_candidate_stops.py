# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:38:29 2022

@author: tpassmore6
"""

#%%import and directory

from pathlib import Path
import os
import partridge as ptg
import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import time
from itertools import combinations, chain, permutations
from scipy.spatial import cKDTree
from tqdm import tqdm
from datetime import datetime
import pickle

#set working directory to where data is stored
user_directory = os.fspath(Path.home())
file_directory = r"\GitHub\TransitSimPublic" #directory of bikewaysim outputs
homeDir = user_directory+file_directory
os.chdir(homeDir)

#%% load GTFS
path = 'gtfs.zip'
service_ids = ptg.read_busiest_date(path)[1]
view = {'trips.txt': {'service_id': service_ids}}

#get geodataframe
feed = ptg.load_geo_feed(path, view)

#change to local crs
feed.shapes.to_crs('epsg:2240',inplace=True)
feed.stops.to_crs('epsg:2240',inplace=True)

#buffer stops by 3 miles
buffered_stops = feed.stops.copy()
buffered_stops.geometry = buffered_stops.buffer(3*5280)
buffered_stops.set_geometry('geometry',inplace=True)

#export the rail routes paths
route_and_shape = feed.trips[['route_id','shape_id']].drop_duplicates()
color = pd.merge(feed.routes,route_and_shape,on='route_id')
final = pd.merge(feed.shapes,color,on='shape_id')
final.route_color = "#" + final.route_color.astype(str)

#final.to_file('testing.gpkg',layer='gtfs_shapes_ptg_color',driver='GPKG')


#%% get bounds of MARTA network to use for downloading OSM network

marta_study_area = buffered_stops.dissolve()
#marta_study_area.to_file(bike_dir+r'\base_shapefiles\studyareas\marta.gpkg',driver='GPKG')

#%% find TAZs within marta service sheds

centroids = gpd.read_file(r'Model_Traffic_Analysis_Zones_2020/Model_Traffic_Analysis_Zones_2020.shp')

#turn to points
centroids['geometry'] = gpd.points_from_xy(centroids['Longitde'], centroids['Latitude'])
centroids.set_geometry('geometry',inplace=True)
centroids.to_crs('epsg:2240',inplace=True)
centroids.FID_1 = centroids.FID_1.astype(str)
centroids = centroids[['FID_1','geometry']]

#intersect with marta study area
centroids = gpd.overlay(centroids,marta_study_area,how='intersection')

#export for map making
#centroids.to_file('base_layers.gpkg',layer='centroids')

#%%get dataframe with stop id, route id, and route_type
route_type = feed.routes[['route_id','route_type']]
stops = feed.stops[['stop_id','stop_name','geometry']]
trips = feed.trips[['trip_id','route_id']]
stop_times = feed.stop_times[['trip_id','stop_id']]
stop_and_route = stops.merge(stop_times,on='stop_id').merge(trips,on='trip_id').merge(route_type,on='route_id')    

#don't need trip id
stop_and_route.drop(columns=['trip_id'],inplace=True)
#drop duplicates so fewer stations to match to
stop_and_route.drop_duplicates(inplace=True)

#drop street_car
stop_and_route = stop_and_route[stop_and_route['route_type']!=0]

#
#stop_and_route.to_file('testing.gpkg',layer='gtfs_stops_ptg_color',driver='GPKG')

#delete gtfs data to save space
del feed, stops, trips, stop_times, buffered_stops, route_type

#%% find candidate stops for each taz

#initialize dict for candidate stops for each TAZ
candidate_stops_by_taz = pd.DataFrame()

#set initial biking threshold distance
bike_thresh = 3*5280 

#for each taz, calculate the euclidean distance to transit stops
for idx, row in centroids.iterrows():
    #make copy of stop_and_route dataframe
    candidate_stops = stop_and_route.copy()
    #calculate distance to all transit stops from centroid
    candidate_stops['distance'] = candidate_stops.distance(row.geometry)
    #knockout those beyond biking threshold
    candidate_stops = candidate_stops[candidate_stops['distance'] < bike_thresh]
    #add directionality bias (future)
    #remove matches that are the same route
    mask = candidate_stops.groupby('route_id')['distance'].idxmin().to_list()
    candidate_stops = candidate_stops.loc[mask]
    
    if len(candidate_stops) > 0:
        #add taz_id
        candidate_stops['FID_1'] = row.FID_1
        #append candidate stops
        candidate_stops_by_taz = candidate_stops_by_taz.append(candidate_stops)
        
#average number of candidate stops
avg_candidate  = round(candidate_stops_by_taz.groupby('FID_1')['stop_id'].count().mean(),0)
print(f'{avg_candidate} stops per TAZ on average')

#%% get od pairs to exlude (less than 3 miles apart)

#get list of tazs
list_of_ids = centroids['FID_1'].tolist()

#get od pairs
exclude = list(combinations(list_of_ids,2))

#create dataframe
exclude = pd.DataFrame.from_records(exclude, columns=['ori_id','dest_id'])

#add geo information back for calculating distance
exclude = pd.merge(exclude,centroids[['FID_1','geometry']],left_on='ori_id',right_on='FID_1')
exclude.drop(columns=['FID_1'],inplace=True)
exclude.rename(columns={'geometry':'ori_geo'},inplace=True)
exclude = pd.merge(exclude,centroids[['FID_1','geometry']],left_on='dest_id',right_on='FID_1')
exclude.drop(columns=['FID_1'],inplace=True)
exclude.rename(columns={'geometry':'dest_geo'},inplace=True)

#calculate distance between all pairs
ori_geo = gpd.GeoSeries(exclude.ori_geo,crs='epsg:2240')
dest_geo = gpd.GeoSeries(exclude.dest_geo,crs='epsg:2240')
exclude['euclid_distance'] = ori_geo.distance(dest_geo,align=True)

#only get od pairs that are at least 2 miles apart
exclude = exclude[exclude['euclid_distance']<2*5280]

#turn into set for efficiency
exclude = set(zip(exclude['ori_id'],exclude['dest_id']))

#drop distance
candidate_stops_by_taz.drop(columns=['distance'],inplace=True)

#%% import OSM network

#import the bike/road links one
links = gpd.read_file(r'osm_network.gpkg',layer='links',driver='GPKG')
nodes = gpd.read_file(r'osm_network.gpkg',layer='nodes',driver='GPKG')

#%% taz and transit snapping

#take in two geometry columns and find nearest gdB point from each
#point in gdA. Returns the matching distance too.
#MUST BE A PROJECTED COORDINATE SYSTEM
def ckdnearest(gdA, gdB, return_dist=True):  
    
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].reset_index(drop=True)
    
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ], 
        axis=1)
    
    if return_dist == False:
        gdf = gdf.drop(columns=['dist'])
    
    return gdf

#take in od data find the neearest node in the network
#returns od table with matched ids and distance to them
def snap_to_network(to_snap,network_nodes_raw):
    #record the starting time
    time_start = time.time()
    
    #create copy of network nodes
    network_nodes = network_nodes_raw.copy()
    
    #rename geometry columns
    to_snap.rename(columns={'geometry':'original'},inplace=True)
    to_snap.set_geometry('original',inplace=True)
    network_nodes.rename(columns={'geometry':'snapped'},inplace=True)
    network_nodes.set_geometry('snapped',inplace=True)
        
    #find closest network node from each orig/dest
    snapped_nodes = ckdnearest(to_snap, network_nodes)

    #filter columns
    snapped_nodes = snapped_nodes[to_snap.columns.to_list()+['N','dist']]
        
    #drop geo column
    snapped_nodes.drop(columns=['original'],inplace=True)
    
    print(f'snapping took {round(((time.time() - time_start)/60), 2)} minutes')
    return snapped_nodes

snapped_tazs = snap_to_network(centroids,nodes)
snapped_stops = snap_to_network(stop_and_route,nodes)

snapped_tazs.rename(columns={'N':'tazN','dist':'taz_snapdist'},inplace=True)
snapped_stops.rename(columns={'N':'stopsN','dist':'stops_snapdist'},inplace=True)

#add snapped nodes back into dataframe
candidate_stops_by_taz = candidate_stops_by_taz.merge(snapped_tazs[['FID_1','tazN','taz_snapdist']],on='FID_1').merge(
    snapped_stops[['stop_id','route_id','stopsN','stops_snapdist']],on=['stop_id','route_id'])


#%% create bike paths

def create_graph(links,impedance_col):
    DGo = nx.DiGraph()  # create directed graph
    for ind, row in links.iterrows():
        DGo.add_weighted_edges_from([(str(row['A']), str(row['B']), float(row[impedance_col]))],weight=impedance_col)   
    return DGo

def find_shortest(links,nodes,candidate_stops_by_taz,impedance_col):

    #record the starting time
    time_start = time.time()
    
    #create weighted network graph
    DGo = create_graph(links,impedance_col)
    
    #initialize empty dicts
    all_impedances = {}
    all_nodes = {}
    all_paths = {}
    
    #listcheck
    candidate_stops_by_taz['tup'] = list(zip(candidate_stops_by_taz.tazN,candidate_stops_by_taz.stopsN))
    listcheck = set(candidate_stops_by_taz['tup'].to_list())
    
    #from each unique origin
    print('Routing from TAZ to transit stops')
    for taz in tqdm(candidate_stops_by_taz.tazN.unique()):
        #run dijkstra's algorithm
        #the cutoff gets a little weird for other impedances
        impedances, paths = nx.single_source_dijkstra(DGo,taz,weight=impedance_col,cutoff=3*5280)    
        
        #filter dijkstra results
        for key in impedances.keys():
            #check if trip is in one of the ones we want
            if (taz,key) in listcheck:
                #for each trip id find impedance
                all_impedances[(taz,key)] = impedances[key]
              
                #convert from node list to edge list
                node_list = paths[key]
                edge_list = [node_list[i]+'_'+node_list[i+1] for i in range(len(node_list)-1)]
                
                #store
                all_nodes[(taz,key)] = node_list
                all_paths[(taz,key)] = edge_list

    #add impedance column to ods dataframe
    candidate_stops_by_taz[f'{impedance_col}_to'] = candidate_stops_by_taz['tup'].map(all_impedances)

    #from each unique transit stop
    print('Routing from transit stops to TAZ')
    for stop in tqdm(candidate_stops_by_taz.stopsN.unique()):
        #run dijkstra's algorithm
        impedances, paths = nx.single_source_dijkstra(DGo,stop,weight=impedance_col,cutoff=3*5280)    
        
        #filter dijkstra results
        for key in impedances.keys():
            #check if trip is in one of the ones we want
            if (key,stop) in listcheck:
                #for each trip id find impedance
                all_impedances[(stop,key)] = impedances[key]
              
                #convert from node list to edge list
                node_list = paths[key]
                edge_list = [node_list[i]+'_'+node_list[i+1] for i in range(len(node_list)-1)]
                
                #store
                all_nodes[(stop,key)] = node_list
                all_paths[(stop,key)] = edge_list

    #add impedance column to ods dataframe
    candidate_stops_by_taz[f'{impedance_col}_from'] = candidate_stops_by_taz['tup'].map(all_impedances)

    # #calculate betweenness centrality
    # node_btw_centrality = pd.Series(list(chain(*[all_nodes[key] for key in all_nodes.keys()]))).value_counts()
    # edge_btw_centrality = pd.Series(list(chain(*[all_paths[key] for key in all_paths.keys()]))).value_counts()
    
    # #add betweenness centrality as network attribute
    # nodes[f'{impedance_col}_btw_cntrlty'] = nodes['N'].map(node_btw_centrality)
    # links[f'{impedance_col}_btw_cntrlty'] = links['A_B'].map(edge_btw_centrality)
    
    # #get standardized betweenness centrality
    # nodes[f'{impedance_col}_std_btw_cntrlty'] = nodes[f'{impedance_col}_btw_cntrlty'] / nodes[f'{impedance_col}_btw_cntrlty'].sum()
    # links[f'{impedance_col}_std_btw_cntrlty'] = links[f'{impedance_col}_btw_cntrlty'] / links[f'{impedance_col}_btw_cntrlty'].sum()
 
    print('Creating paths')
    for key in tqdm(all_paths.keys()):
        if len(all_paths[key]) > 1:
            #get geo (this takes a bit)
            all_paths[key] = links[links['A_B'].isin(all_paths[key])].dissolve().geometry.item()
    
    print(f'Shortest path routing took {round(((time.time() - time_start)/60), 2)} minutes')
    
    return candidate_stops_by_taz, all_paths


#%%

links['length'] = links.length
impedance_col = 'length'
bikepaths, all_paths = find_shortest(links, nodes, candidate_stops_by_taz, impedance_col)

#%% pre-process

#drop if both are na
bikepaths.dropna(subset=['length_from','length_to'],how='all',inplace=True)

#turn distance into ints
bikepaths['length_to'] = bikepaths['length_to'].astype(int)
bikepaths['length_from'] = bikepaths['length_from'].astype(int)

#calculate arrival times to transit stations
bikespd = 8

#set start time
start_time = datetime(2022, 9, 22, 8, 0, 0, 0)

#get arrival time
bikepaths['arrival_time'] = start_time + pd.to_timedelta(bikepaths['length_to'] / 5280 / bikespd * 60 * 60, unit='s')

#figure out how to round start time
bikepaths['arrival_time'] = bikepaths['arrival_time'].dt.round('min')

#%%
#create tuple column
list_of_ids = list(zip(
    bikepaths['FID_1'],
    bikepaths['stop_id'],
    bikepaths['length_to'],
    bikepaths['length_from'],
    bikepaths['route_id'],
    bikepaths['route_type'],
    bikepaths['arrival_time']
    ))


#list comphehension version (faster)
print('Creating final pairs...')
time_start = time.time()
transit_od_pairs = [(i[0][0],i[1][0],i[0][1],i[1][1],i[0][6]) for i in permutations(list_of_ids,2) if \
                    (i[0][0]!=i[1][0]) & \
                        ((i[0][0],i[1][0]) not in exclude) & \
                            ((i[0][4] == 3) & \
                             ((i[1][4] == 3) & \
                                               (i[0][3] != i[1][3])) == False) & \
                                ((i[0][2]+i[1][3]) < bike_thresh)]
print(f'took {round(((time.time() - time_start)/60), 2)} minutes')

# #combinations versions (less memory)
# time_start = time.time()
# transit_od_pairs = [(i[0][0],i[1][0],i[0][1],i[1][1],i[0][6]) for i in combinations(list_of_ids,2) if \
#                     (i[0][0]!=i[1][0]) & \
#                         ((i[0][0],i[1][0]) not in exclude) & \
#                             ((i[0][4] == 3) & \
#                              ((i[1][4] == 3) & \
#                                                (i[0][3] != i[1][3])) == False) & \
#                                 ((i[0][2]+i[1][3]) < bike_thresh)]
# print(f'took {round(((time.time() - time_start)/60), 2)} minutes')



#return the following format
#(start_taz,end_taz,start_stop,end_stop,first_arrvial_time)

#condition 1
#checks for self pairs

#condition 2
#checks to see if pair is in excluded because origin and destination are too close

#conditions 3 and 4
#check to see if both routes are busses, if they are then check to see if their route number is the different

#condition 5
#check to see if from/to stop distance is less than the bike_threshold

#export as pickle
time_start = time.time()
print('Exporting final pairs...')
with open(r'transit_od_pairs.pkl','wb') as fh:
    pickle.dump(transit_od_pairs,fh)
print(f'took {round(((time.time() - time_start)/60), 2)} minutes')

#%%

#see if duplicates


# #%% save results

# import pickle

# def backup(thing,name):
#     with open(r'{name}.pkl', 'wb') as fh:
#         pickle.dump(thing,fh)
        
# def load_backup(name):
#     with open(r'{name}.pkl', 'rb') as fh:
#         thing = pickle.load(fh)
#     return thing
# #%%
# backup(bikepaths,'bikepaths')
# backup(edge_list,'edge_list')

# #%%
# bikepaths = load_backup('bikepaths')
# edge_list = load_backup('edge_list')


#%%for loop version (much longer)
# transit_od_pairs = list()
# for i in combinations(list_of_ids,2):
#     #check if bus route
#     cond_1 = i[0][4] == 3
#     cond_2 = i[1][4] == 3
#     #check if same route
#     cond_3 = i[0][3] != i[1][3]
#     #check total biking distance
#     cond_4 = (i[0][2]+i[1][2]) < bike_thresh
#     #check if trip is too short for transit
#     cond_5 = (i[0][0],i[1][0]) not in exclude
  
#     #bus route condition
#     if (cond_1 & cond_2 & cond_3) == False:
#         #bike distance condition 
#             if cond_4:
#                 if cond_5:
#                     transit_od_pairs.append(i)
#                 #trip distance condition (this one is slow)
#                 #if cond_5:
#                     #transit_od_pairs.append(i)
# print(f'construct took {round(((time.time() - time_start)/60), 2)} minutes')





