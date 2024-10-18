import numpy as np

def sph2cart(r, lat, lon):
    """
    Transform spherical (meters, degrees) to cartesian (meters)
    Args:
        r: meters
        lat: degrees
        lon: degrees

    Returns:
    Cartesian xyz (meters)
    """
    x = r * cosd(lon) * cosd(lat)
    y = r * sind(lon) * cosd(lat)
    z = r * sind(lat)

    return x, y, z

# transform cartesian to spherical (meters, radians)
def cart2sph(xyz):
    # print("cart2sph in",np.array(xyz))

    rtmp = np.linalg.norm(np.array(xyz).reshape(-1, 3), axis=1)
    lattmp = np.arcsin(np.array(xyz).reshape(-1, 3)[:, 2] / rtmp)
    lontmp = np.arctan2(np.array(xyz).reshape(-1, 3)[:, 1], np.array(xyz).reshape(-1, 3)[:, 0])

    return rtmp, lattmp, lontmp


def sind(x):
    return np.sin(np.deg2rad(x))


def cosd(x):
    return np.cos(np.deg2rad(x))


def unproject_stereographic(x, y, lon0, lat0, R):
    """
    Stereographic Coordinates unprojection
    Args:
        x: stereo coord
        y: stereo coord
        lon0: center of the projection (longitude, deg)
        lat0: center of the projection (latitude, deg)
        R: planet radius

    Returns:
    Longitude and latitude (deg) of points in cylindrical coordinates
    """
    rho = np.sqrt(np.power(x, 2) + np.power(y, 2))
    c = 2 * np.arctan2(rho, 2 * R)

    lat = np.rad2deg(np.arcsin(np.cos(c) * sind(lat0) + (cosd(lat0) * y * np.sin(c)) / rho))
    lon = np.mod(
        lon0 + np.rad2deg(np.arctan2(x * np.sin(c), cosd(lat0) * rho * np.cos(c) - sind(lat0) * y * np.sin(c))), 360)

    lat = np.where(x**2+y**2 == 0, lat0, lat)
    lon = np.where(x**2+y**2 == 0, lon0, lon)

    # if (x == 0).any() and (y == 0).any():
    #     print("coming here")
    #     #    if x == 0 and y == 0:
    #     return lon0, lat0
    # else:
    return lon, lat

def project_stereographic(lon, lat, lon0, lat0, R=1):
    """
    project cylindrical coordinates to stereographic xy from central lon0/lat0
    :param lon: array of input longitudes (deg)
    :param lat: array of input latitudes (deg)
    :param lon0: center longitude for the projection (deg)
    :param lat0: center latitude for the projection (deg)
    :param R: planetary radius (km)
    :return: stereographic projection xy coord from center (km)
    """

    cosd_lat = cosd(lat)
    cosd_lon_lon0 = cosd(lon - lon0)
    sind_lat = sind(lat)

    k = (2. * R) / (1. + sind(lat0) * sind_lat + cosd(lat0) * cosd_lat * cosd_lon_lon0)
    x = k * cosd_lat * sind(lon - lon0)
    y = k * (cosd(lat0) * sind_lat - sind(lat0) * cosd_lat * cosd_lon_lon0)

    return x, y