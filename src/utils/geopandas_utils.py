import geopandas as gpd


def GeoSeries_to_GeoDataFrame(series):
    return gpd.GeoDataFrame({'geometry': series})

# from https://gis.stackexchange.com/questions/374864/creating-a-square-buffer-around-a-shapely-polygon
def to_square(polygon):
    from shapely.geometry import Point
    from math import sqrt

    minx, miny, maxx, maxy = polygon.bounds

    # get the centroid
    centroid = [(maxx + minx) / 2, (maxy + miny) / 2]
    # get the diagonal
    diagonal = sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2)

    return Point(centroid).buffer(diagonal / sqrt(2.) / 2., cap_style=3) # 3=square