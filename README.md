# ML Projects

Two applied machine learning projects on environmental and geoscience problems, built end to end in Python: one geospatial (flood risk mapping for Lagos) and one computer vision (rock and mineral identification).

| Project | Problem | Approach | Result |
| --- | --- | --- | --- |
| [Lagos Flood Hotspots](Flood_Hotspots_Project) | Where in Lagos are flooding conditions most likely? | Elevation, rainfall, land cover and population rasters aligned to a 40 m UTM grid (EPSG:32631), stacked into a pixel-level dataset, Random Forest classifier | 98.2% overall accuracy on a held-out test set of about 596,000 pixels. Recall on flooded pixels is 73%, precision 99%. QGIS-ready probability and risk-class maps |
| [Rock Identification](Rock_Identification_Poject) | Can a model name a rock from a photo? | MobileNetV2 transfer learning with two output heads (rock name and rock family), class weighting, two-phase training on 13,000+ images | Rock family (igneous, metamorphic, sedimentary) about 65 to 67% accuracy. Exact rock name about 39 to 41% top-1 and 58 to 59% top-3 across many classes |

## Lagos Flood Hotspots

![Flood risk probability map for Lagos](Flood_Hotspots_Project/assets/Flood%20Risk%20Probability.png)

- **Data:** SRTM elevation, CHIRPS monthly rainfall (2020 to 2021), ESA WorldCover, WorldPop population, and digitized flood labels rasterized to the DEM grid.
- **Pipeline:** reproject, clip, resample and normalize rasters, build the dataset, train, then generate probability maps, risk classes, monthly rainfall scenarios and influence maps.
- **Read this as a pilot:** it uses a single flood label layer, so it shows the workflow and the relative influence of rainfall, terrain and population rather than an operational forecast.

Full method, metrics and run instructions: [Flood_Hotspots_Project/README.md](Flood_Hotspots_Project/README.md)

## Rock Identification

- **Task:** classify rock and mineral photos into specific types (for example basalt, granite, coal) and into the three geological families at the same time.
- **Model:** MobileNetV2 pretrained on ImageNet, frozen phase followed by fine-tuning, weighted loss for rare classes, exported as `.keras` and TensorFlow SavedModel with label maps.
- **What I learned:** fine-grained rock names are hard from photos alone. A second version with stronger augmentation and longer training did not beat the first, which is documented in the project README along with the reproducible V2 pipeline.

Full method, metrics and run instructions: [Rock_Identification_Poject/README.md](Rock_Identification_Poject/README.md)

## Tech

Python, scikit-learn, rasterio, pandas, NumPy, Matplotlib, TensorFlow, Keras, QGIS

## Run locally

```bash
git clone https://github.com/Jubemi-Pajiah/ML-Projects.git
cd ML-Projects
pip install -r requirements.txt
```

Each project folder documents its own steps. Raw rasters and image datasets are not included in the repository.

## Author

Jubemi Pajiah, software developer and published researcher (hydrogeology and environmental geochemistry).
[jubemi.com](https://jubemi.com) | [LinkedIn](https://www.linkedin.com/in/jubemi-pajiah-626b7323b/) | [Google Scholar](https://scholar.google.com/citations?hl=en&user=z6iMmPgAAAAJ) | info@jubemi.com
