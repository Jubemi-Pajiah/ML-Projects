import rioxarray
import numpy as np
import os

base_dir = "Flood_Hotspots_Project/resampled_40m"
dem_path = os.path.join(base_dir, "dem_40m.tif")
slope_path = os.path.join(base_dir, "slope_40m.tif")

# Load DEM
dem = rioxarray.open_rasterio(dem_path, masked=True).squeeze()

# Get cell size in meters
xres = abs(dem.rio.resolution()[0])
yres = abs(dem.rio.resolution()[1])

# Compute gradients
dzdx, dzdy = np.gradient(dem, yres, xres)

# Compute slope in degrees
slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))

# Convert to DataArray and copy spatial attributes
slope_da = dem.copy(data=slope)
slope_da.rio.to_raster(slope_path)

print("Slope raster saved as:", slope_path)
