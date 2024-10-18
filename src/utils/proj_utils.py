def project_coord(input, crs, inverse=False):

    import pyproj

    p = pyproj.Proj(crs)
    print(input)
    a, b = input # a,b = lon,lat or x,y depending on crs and inverse

    return p(a, b, inverse=inverse)

if __name__ == '__main__':

    lonlat_lst = [(120., 82), (120., 84), (137., 84), (137., 82)]

    # set projection to NP stereo for Mercury
    crs_merc_np_stereo = '+proj=stere +lat_0=90 +lon_0=0 +lat_ts=90 +k=1 +x_0=0 +y_0=0 +units=m +a=2440e3 +b=2440e3 +no_defs'  # km, else for meters a*1000, b*1000

    for lonlat in lonlat_lst:
        out = project_coord(input = lonlat, crs=crs_merc_np_stereo, inverse= False)
        # print("x,y=",out)
        x, y = out
        # import numpy as np
        # print(np.linalg.norm([x,y]))
        print(x,"   ",y)