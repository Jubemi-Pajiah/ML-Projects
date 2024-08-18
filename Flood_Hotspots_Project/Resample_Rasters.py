import os
import rasterio
from rasterio.enums import Resampling
from tqdm import tqdm
from config import raw_dir,out_dir

os.makedirs(out_dir, exist_ok=True)

target_res = 40
rasters = [f for f in os.listdir(raw_dir) if f.endswith(".tif")]

print(f"Found {len(rasters)} raster(s) in '{raw_dir}'")
print("Starting resampling...\n")

for raster_name in tqdm(rasters, desc="Resampling rasters"):
    input_path = os.path.join(raw_dir, raster_name)
    output_path = os.path.join(out_dir, raster_name.replace(".tif", "_40m.tif"))

    print(f"Resampling: {raster_name}")

    with rasterio.open(input_path) as src:
        # compute new transform, width, and height
        scale_x = src.res[0] / target_res
        scale_y = src.res[1] / target_res
        new_transform = src.transform * src.transform.scale(scale_x, scale_y)
        new_width = int(src.width * scale_x)
        new_height = int(src.height * scale_y)

        # choose resampling type
        if "cover" in raster_name.lower() or "flood" in raster_name.lower():
            resampling = Resampling.nearest
        else:
            resampling = Resampling.bilinear

        kwargs = src.meta.copy()
        kwargs.update({
            "transform": new_transform,
            "width": new_width,
            "height": new_height
        })

        with rasterio.open(output_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                dst.write(
                    src.read(
                        i,
                        out_shape=(src.count, new_height, new_width),
                        resampling=resampling
                    ),
                    i
                )

    print(f"Finished: {raster_name} → saved to resampled_40m/\n")

print("All rasters successfully resampled to 40 m resolution!")
