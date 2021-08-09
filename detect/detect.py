#!/usr/bin/env python3
import json
import csv
from geojson import Feature, Point, FeatureCollection
from GPSPhoto import gpsphoto

detection_results = "result.json"
csv_file = open('out.csv', 'w')
geojson_file = open('out.geojson', 'w')


def get_geojson(data):
    features = []
    for d in data:
        point = Point((d[5], d[4]))
        properties = {
            "filename": d[0],
            "class_id": d[1],
            "name": d[2],
            "confidence": d[3]
        }
        feature = Feature(geometry=point, properties=properties)
        features.append(feature)

    return FeatureCollection(features)


def detect():
    f = open(detection_results, 'r')
    data = json.load(f)
    out_data = []
    for d in data:
        exif = gpsphoto.getGPSData(d['filename'])
        latitude, longitude = exif['Latitude'], exif['Longitude']
        for o in d['objects']:
            out_data.append([d['filename'], o['class_id'], o['name'],
                             o['confidence'], latitude, longitude])
    write = csv.writer(csv_file)
    write.writerows(out_data)
    geojson_data = get_geojson(out_data)
    json.dump(geojson_data, geojson_file)


detect()
