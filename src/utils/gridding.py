import numpy as np
import shapely


def new_grid(n_cells_per_side, bounds, n_cells_x=None, n_cells_y=None):
    """

    Parameters
    ----------
    n_cells int, number of cells per grid side (n_cells+1)
    bounds [double,4] xmin, ymin, xmax, ymax
    n_cells_x, n_cells_y int, number of cells per grid side (n_cells+1) along x and y (specify if different)

    Returns
    -------
    grid_cells [n_cells+1,n_cells+1]
    """
    # create gridding
    # https://james-brennan.github.io/posts/fast_gridding_geopandas/
    # total area for the grid
    xmin, ymin, xmax, ymax = bounds

    if (n_cells_x is None) and (n_cells_y is None):
        # how many cells across and down
        cell_size = (xmax - xmin) / n_cells_per_side
        cell_size_x = cell_size
        cell_size_y = cell_size
    elif (n_cells_x is None):
        # how many cells across and down
        cell_size_x = (xmax - xmin) / n_cells_per_side
        cell_size_y = (ymax - ymin) / n_cells_y
    elif (n_cells_y is None):
        # how many cells across and down
        cell_size_x = (xmax - xmin) / n_cells_x
        cell_size_y = (ymax - ymin) / n_cells_per_side
    else:
        cell_size_x = (xmax - xmin) / n_cells_x
        cell_size_y = (ymax - ymin) / n_cells_y

    # projection of the grid
    # create the cells in a loop
    grid_cells = []
    for x0 in np.arange(xmin, xmax, cell_size_x):
        for y0 in np.arange(ymin, ymax, cell_size_y):
            # bounds
            x1 = x0 + cell_size_x
            y1 = y0 + cell_size_y
            grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))

    return grid_cells