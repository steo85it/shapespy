import time
from functools import partial
import numpy as np
from sklearn.neighbors import KDTree

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import xarray as xr

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().

    FROM: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def calc_cross(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    v3 = np.cross(v1, v2)
    return v3 / np.linalg.norm(v3)

def PCA_unit_vector(array, pca=PCA(n_components=3)):
    pca.fit(array)
    eigenvalues = pca.explained_variance_
    return pca.components_[np.argmin(eigenvalues)]

## NEW ##
def calc_angle_with_xy(vectors):
    '''
    Assuming unit vectors!
    '''
    l = np.sum(vectors[:, :2] ** 2, axis=1) ** 0.5
    return np.arctan2(vectors[:, 2], l)


def surfnorm(X, Y, Z, method='standard', npoints=3):
    """
    https://stackoverflow.com/questions/70711741/most-efficient-way-to-calculate-point-wise-surface-normal-from-a-numpy-grid
    Parameters
    ----------
    x, y: xarray dataarray coords, np.meshgrid(x, y) --> z.shape
    z: xarray dataarray data

    Returns
    -------

    """

    # # Grab some test data.
    # X, Y, Z = axes3d.get_test_data(0.25)
    # print(X.shape, Y.shape, Z.shape)

    X, Y, Z = map(lambda x: x.flatten(), [X, Y, Z])
    # print(X.shape, Y.shape, Z.shape)
    # exit()


    data = np.array([X, Y, Z]).T

    tree = KDTree(data, metric='minkowski')  # minkowski is p2 (euclidean)

    # Get indices and distances:
    # start = time.time()
    dist, ind = tree.query(data, k=npoints)  # k=3 points including itself
    combinations = data[ind]

    if method == 'standard':
        normals = list(map(lambda x: calc_cross(*x), combinations))
        # print(combinations[10])
        # print(normals[10], np.round(time.time()-start, 2))
    elif method == 'lazy_map':
        # lazy with map
        # start = time.time()
        # dist, ind = tree.query(data, k=3)  # k=3 points including itself
        # combinations = data[ind]
        normals = list(map(PCA_unit_vector, combinations))
        # print(normals2[10], np.round(time.time()-start, 2))
    elif method == 'functools_map':
        # start = time.time()
        # dist, ind = tree.query(data, k=5)  # k=3 points including itself
        # combinations = data[ind]
        # map with functools
        pca = PCA(n_components=3)
        normals = list(map(partial(PCA_unit_vector, pca=pca), combinations))
        # print(combinations[10])
        # print(normals3[10], np.round(time.time()-start, 2))

    n = np.array(normals)
    n[calc_angle_with_xy(n) < 0] *= -1

    u, v, w = n.T

    return u, v, w

if __name__ == '__main__':

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    root = "/home/sberton2/Lavoro/code/geotools/examples/SYLSFS/"
    auxdir = f"{root}aux/"
    outdir = f"{root}out/"
    prioridem = f"{auxdir}SLDEM15_cut.tif"

    # load a priori dem M0 and clip to region
    dem_ds_st = xr.open_dataset(prioridem) * 1e3 - 1737.4e3
    # crs_dem = dem_ds_st.rio.crs
    dem_ds_st = dem_ds_st.rio.clip_box(minx=-423500, miny=-1089800,
                                       maxx=-422000, maxy=-1088300)

    # upscale by a factor 12
    resampling_factor = 12
    new_x = np.linspace(dem_ds_st.x[0], dem_ds_st.x[-1], dem_ds_st.dims["x"] * resampling_factor)
    new_y = np.linspace(dem_ds_st.y[0], dem_ds_st.y[-1], dem_ds_st.dims["y"] * resampling_factor)
    dem_ds_st = dem_ds_st.interp(y=new_y, x=new_x)
    dem_resol = dem_ds_st.rio.resolution()[0] / resampling_factor
    dem_ds_st = dem_ds_st
    demvar = [x for x in dem_ds_st.data_vars][0]
    dem_ds_st = dem_ds_st.rename_vars({demvar: 'M'})

    da = dem_ds_st.M[0]
    x = da.coords['x'].values
    y = da.coords['y'].values
    X, Y = np.meshgrid(x, y)
    Z = da.values
    u, v, w = surfnorm(X, Y, Z)
    print(np.vstack([u, v, w]).T[10])


    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.set_aspect('equal')
    # # Make the grid
    # plt.plot(X, Y, Z, '.')
    # plt.show(block=False)
    # ax.quiver(X, Y, Z, u, v, w, length=10, normalize=True)
    # set_axes_equal(ax)
    # plt.show()

