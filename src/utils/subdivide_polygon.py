#### Python script for dividing any shapely polygon into smaller equal sized polygons
import time

import matplotlib.pyplot as plt
import numpy as np
from shapely.ops import split
import geopandas
from shapely.geometry import MultiPolygon, Polygon
from shapely import LineString
from tqdm import tqdm


def rhombus(square):
    """
    Naively transform the square into a Rhombus at a 45 degree angle
    """
    coords = square.boundary.coords.xy
    xx = list(coords[0])
    yy = list(coords[1])
    radians = 1
    points = list(zip(xx, yy))
    Rhombus = Polygon(
        [
            points[0],
            points[1],
            points[3],
            ((2 * points[3][0]) - points[2][0], (2 * points[3][1]) - points[2][1]),
            points[4],
        ]
    )
    return Rhombus


def get_squares_from_rect(RectangularPolygon, side_length=0.0025):
    """
    Divide a Rectangle (Shapely Polygon) into squares of equal area.

    `side_length` : required side of square

    """
    rect_coords = np.array(RectangularPolygon.boundary.coords.xy)
    y_list = rect_coords[1]
    x_list = rect_coords[0]
    y1 = min(y_list)
    y2 = max(y_list)
    x1 = min(x_list)
    x2 = max(x_list)
    width = x2 - x1
    height = y2 - y1

    xcells = int(np.round(width / side_length))
    ycells = int(np.round(height / side_length))

    yindices = np.linspace(y1, y2, ycells + 1)
    xindices = np.linspace(x1, x2, xcells + 1)
    horizontal_splitters = [
        LineString([(x, yindices[0]), (x, yindices[-1])]) for x in xindices
    ]
    vertical_splitters = [
        LineString([(xindices[0], y), (xindices[-1], y)]) for y in yindices
    ]
    result = RectangularPolygon
    for splitter in vertical_splitters:
        result = MultiPolygon(split(result, splitter))
    for splitter in horizontal_splitters:
        result = MultiPolygon(split(result, splitter))
    square_polygons = list(result.geoms)

    return square_polygons


def split_polygon(G, side_length=0.025, shape="square", thresh=0.9):
    """
    Using a rectangular envelope around `G`, creates a mesh of squares of required length.

    Removes non-intersecting polygons.


    Args:

    - `thresh` : Range - [0,1]

        This controls - the number of smaller polygons at the boundaries.

        A thresh == 1 will only create (or retain) smaller polygons that are
        completely enclosed (area of intersection=area of smaller polygon)
        by the original Geometry - `G`.

        A thresh == 0 will create (or retain) smaller polygons that
        have a non-zero intersection (area of intersection>0) with the
        original geometry - `G`

    - `side_length` : Range - (0,infinity)
        side_length must be such that the resultant geometries are smaller
        than the original geometry - `G`, for a useful result.

        side_length should be >0 (non-zero positive)

    - `shape` : {square/rhombus}
        Desired shape of subset geometries.


    """
    assert side_length > 0, "side_length must be a float>0"
    start = time.time()
    Rectangle = G.envelope
    squares = get_squares_from_rect(Rectangle, side_length=side_length)
    print(f"end squares {time.time() - start}")
    start = time.time()
    SquareGeoDF = geopandas.GeoDataFrame(geopandas.GeoDataFrame(squares).rename(columns={0: "geometry"})) # needed for gpd[@0.11:]
    Geoms = SquareGeoDF[SquareGeoDF.intersects(G)].geometry.values
    if shape == "rhombus":
        Geoms = [rhombus(g) for g in Geoms]
        geoms = [g for g in Geoms if ((g.intersection(G)).area / g.area) >= thresh]
    elif shape == "square":
        # geoms = []
        # for g in tqdm(Geoms):
        #     if ((g.intersection(G)).area / g.area) >= thresh:
        #         geoms.append(g)
        geoms = [g for g in Geoms if ((g.intersection(G)).area / g.area) >= thresh]
    print(f"end squares2 {time.time() - start}")

    return geoms

if __name__ == '__main__':

    # Reading geometric data

    # geo_filepath = "/data/geojson/pc_14.geojson"
    # GeoDF = geopandas.read_file(geo_filepath)
    strip_box = [(-1485.27, 721.91), (150.37, 55.81), (168.88, 118.72), (-1422.36, 773.71)]
    G = Polygon(strip_box)
    GeoDF = geopandas.GeoDataFrame([G], columns=['geometry'])

    # Selecting random shapely-geometry

    # G = np.random.choice(GeoDF.geometry.values)

    squares = split_polygon(G, shape='square', thresh=0.1, side_length=100)
    rhombuses = split_polygon(G, shape='rhombus', thresh=0.1, side_length=100)

    squares_gdf = geopandas.GeoDataFrame(squares).rename(columns={0: "geometry"})
    print(squares_gdf)
    squares_gdf.plot(edgecolor="grey")
    plt.show()
    plt.close()
    plt.clf()
    rhombuses_gdf = geopandas.GeoDataFrame(rhombuses).rename(columns={0: "geometry"})
    print(rhombuses_gdf)
    rhombuses_gdf.plot(edgecolor="grey")
    plt.show()
