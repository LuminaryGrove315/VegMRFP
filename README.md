The Python libraries we are using include 2DGS, BBSplat, and reduced-3dgs; please refer to their respective websites:

https://github.com/graphdeco-inria/reduced-3dgs
https://github.com/david-svitov/BBSplat
https://github.com/hbb1/2d-gaussian-splatting

1. Use SAM to remove the background from plant images, leaving only the clean plant.

2. Build a 2DGS dataset.

3. Run the STAGE1-2DGS code (Step 1) to generate the plant PLY file.

4. Copy the STAGE1 PLY file into STAGE2-DBSCAN/data and run main.py to get the segmentation results.

5. Copy the segmentation results into the data directories of both files under STAGE3, and run the programs to generate PLY files.

6. Merge the two PLY files to produce the final result.

Please download the vegetation datasets from our project website:

https://luminarygrove315.github.io/VegMRFP/
