import xarray as xr
import numpy as np
# from skspatial.objects import Plane, Points
import matplotlib.pyplot as plt
# from line_profiler_pycharm import profile
from tqdm import tqdm

# @profile
def plane_fit(xv, yv, zv):
    """
    Fit a plane z = ax + by + c to the given points using least squares.
    """
    A = np.c_[xv, yv, np.ones(xv.shape)]
    # Solving for [a, b, c] in the plane equation z = ax + by + c using least squares
    C, _, _, _ = np.linalg.lstsq(A, zv, rcond=None)
    return C

# @profile
def plane_fit_fast(xv, yv, zv):
    """
    Fit a plane z = ax + by + c using normal equations for faster computation.
    If the matrix is singular, fallback to np.linalg.lstsq for a more stable solution.
    """
    A = np.c_[xv, yv, np.ones_like(xv)]

    # Normal equation components
    ATA = A.T @ A
    ATz = A.T @ zv

    try:
        # Try solving the normal equations
        coeffs = np.linalg.solve(ATA, ATz)
    except np.linalg.LinAlgError:
        # If the matrix is singular, fall back to np.linalg.lstsq
        coeffs, *_ = np.linalg.lstsq(A, zv, rcond=None)

    return coeffs

# @profile
def plane_fit_regularized(xv, yv, zv, regularization=1e-8):
    """
    Fit a plane z = ax + by + c using normal equations with regularization
    to avoid singular matrices.
    """
    A = np.c_[xv, yv, np.ones_like(xv)]

    # Normal equation components
    ATA = A.T @ A
    ATz = A.T @ zv

    # Ensure ATA is a float array to avoid integer casting errors
    ATA = ATA.astype(np.float64)

    # Add regularization to diagonal to avoid singular matrix
    ATA += np.eye(ATA.shape[0]) * regularization

    # Solve the regularized system
    coeffs = np.linalg.solve(ATA, ATz)

    return coeffs


# @profile
def compute_residuals(xv, yv, zv, plane_params):
    """
    Compute the residuals of the points from the plane z = ax + by + c.
    """
    a, b, c = plane_params
    # Calculate the predicted z values from the plane equation
    z_pred = a * xv + b * yv + c
    # Calculate residuals (z - z_pred)
    residuals = zv - z_pred
    return residuals


#@profile
def slope_detrended_std(constructed_data, window_size, subset_points=10):
    """
    Compute the standard deviation of the detrended slope using a fixed subset of points to fit the plane.

    Parameters
    ----------
    constructed_data: numpy.ndarray
        Array with constructed rolling window data.
    window_size: int
        Size of the rolling window.
    subset_points: int, optional
        Number of points to use for fitting the best fit plane. Default is 10.

    Returns
    -------
    float
        Standard deviation of the detrended slope.
    """
    zv_flat = constructed_data.ravel()
    valid_mask = ~np.isnan(zv_flat)
    zv_valid = zv_flat[valid_mask]

    if zv_valid.size == 0:
        return np.nan

    # Randomly sample subset_points from the valid points
    num_valid_points = len(zv_valid)
    num_points = min(subset_points, num_valid_points)
    # sample_indices = np.random.choice(np.arange(num_valid_points), size=num_points, replace=False)
    # Use linspace to select a fixed set of equally spaced sample points
    sample_indices = np.linspace(0, num_valid_points - 1, num_points, dtype=int)

    # Subset the valid points
    zv_sample = zv_valid[sample_indices]
    valid_indices = np.nonzero(valid_mask)[0]
    sampled_indices = valid_indices[sample_indices]

    # Convert the sampled linear indices into 2D indices
    yv_sample, xv_sample = np.unravel_index(sampled_indices, constructed_data.shape)

    # Fit the plane using the sampled points
    plane_params = plane_fit_regularized(xv_sample, yv_sample, zv_sample)

    # Compute residuals using the fitted plane parameters
    residuals = compute_residuals(xv_sample, yv_sample, zv_sample, plane_params)

    # Return the standard deviation of the residuals
    return np.std(residuals)


# @profile
def compute_sdr(dem_data, winsize=3):
    """
    Compute slope detrended roughness for the given digital elevation model (DEM) data.

    Parameters
    ----------
    dem_data: xarray.DataArray
        Digital elevation model (DEM) data.
    winsize: int, optional
        Size of the rolling window (default is 3).

    Returns
    -------
    xarray.DataArray
        Roughness map for the given window size.
    """
    # Construct the rolling windows
    rolling_dims = {'x': 'x_win', 'y': 'y_win'}  # Define rolling dimension names
    constructed = dem_data.rolling(x=winsize, y=winsize, center=True).construct(rolling_dims)
    print(constructed)

    # Apply slope_detrended_std function over moving window
    roughness = xr.apply_ufunc(
        slope_detrended_std,
        constructed,
        winsize,  # Pass the baseline/window size as an additional argument
        input_core_dims=[[rolling_dims['x'], rolling_dims['y']], []],  # Add an empty list for the new argument
        vectorize=True,
        output_dtypes=[float]
    )

    return roughness


local = False

# Example usage with mock data
if local:
    pdir = "/home/sberton2/Lavoro/projects/A3CLR/"
    # set paths to computed/simulated and measured/real images
else:
    pdir = "/panfs/ccds02/nobackup/people/sberton2/RING/code/sfs_helper/examples/HLS/A3CLS/proc/tile_4/sel_0/products/"

meas_path = f"{pdir}A304_GLDELEV_001.tif"
comp_path = f"{pdir}A304_GLDELEV_001.tif"
prods_name = "A304_GLDELEV_001"

# Read and process the DEM data
dem_ = xr.open_dataarray(comp_path).isel(band=0).coarsen(x=1, boundary='trim').mean().coarsen(y=1,
                                                                                               boundary='trim').mean()
print(dem_)

# Compute roughness for different baselines
baselines = np.arange(50, 51, 3)  # Example baselines
roughness_maps = {}

for baseline in tqdm(baselines):
    roughness_maps[baseline] = compute_sdr(dem_, winsize=baseline)

# Plotting the roughness map for each baseline
fig, axes = plt.subplots(nrows=len(roughness_maps), ncols=1, figsize=(10, 5 * len(roughness_maps)))

for i, (baseline, roughness) in enumerate(roughness_maps.items()):
    if len(roughness_maps) > 1:
        ax = axes[i]
    else:
        ax = axes

    roughness.plot(ax=ax, robust=True, add_labels=True, cmap='viridis')
    ax.set_title(f'Roughness Map (Baseline {baseline})')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

plt.tight_layout()

# Save the figure
plt.savefig(f'{pdir}roughness_maps_{prods_name}.png')

# Display the figure
plt.show()

# Plotting the power spectrum of roughness as a function of the baseline
fig, axes = plt.subplots(nrows=len(baselines), ncols=1, figsize=(10, len(baselines) * 3))
for i, baseline in enumerate(baselines):
    power_spectrum = np.abs(np.fft.fftshift(np.fft.fft2(roughness_maps[baseline])))
    axes[i].imshow(np.log(power_spectrum), cmap='viridis')
    axes[i].set_title(f'Power Spectrum of Roughness (Baseline {baseline})')
    axes[i].set_xlabel('Frequency X')
    axes[i].set_ylabel('Frequency Y')

plt.tight_layout()
plt.savefig(f'{pdir}power_spectrum_{prods_name}.png')

# Compute total power per baseline
total_power_per_baseline = {}
for baseline in baselines:
    fft_2d = np.fft.fft2(roughness_maps[baseline])
    power_spectrum = np.abs(fft_2d) ** 2
    total_power = np.sum(power_spectrum)
    total_power_per_baseline[baseline] = total_power

# Plotting total power vs baseline
plt.figure(figsize=(8, 6))
plt.plot(list(total_power_per_baseline.keys()), list(total_power_per_baseline.values()), marker='o')
plt.xlabel('Baseline (window size)')
plt.ylabel('Total Power')
plt.title('Topography Roughness Power Spectrum')
plt.grid(True)
plt.savefig(f'{pdir}total_power_vs_baseline_{prods_name}.png')

plt.show()
