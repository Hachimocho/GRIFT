from skimage import io
from skimage.metrics import structural_similarity
from skimage.color import rgb2gray
import os
import glob
import sys
import tqdm
from itertools import combinations
import csv
from math import comb



# Specify the main directory
main_dir = '/home/brg2890/data/datasets/FaceForensics++/Faces'

# Use glob to match the pattern '*/faces/*'
file_list = glob.glob(main_dir + '/**/*', recursive=True)

# Filter out directories, keep only files
file_list = [f for f in file_list if os.path.isfile(f)]


with open('combinations.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    num_combinations = comb(len(file_list), 2)

    print(num_combinations)
    for path1, path2 in tqdm.tqdm(combinations(file_list, 2)):
        image1 = io.imread(path1, as_gray=True)
        image2 = io.imread(path2, as_gray=True)

        # Compute SSIM between two images
        ssim_score = structural_similarity(image1, image2, data_range=1.0)

        # Save SSIM score to csv file
        writer.writerow([path1, path2, str(ssim_score)])