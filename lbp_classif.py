import numpy as np
import cv2

from skimage.feature import local_binary_pattern

#Settings
radius = 3
n_points = 8 * radius
METHOD = 'uniform'

#Training Images
PalmInput = ["Palm1", "Palm8", "Palm9", "Palm10"]
NonPalmInput = ["NonPalm4", "NonPalm7", "NonPalm8"]

#Testing Images
TestInput = ["Palm1", "Palm2", "Palm3", "Palm4", "Palm5", "Palm6", "Palm7", "Palm8", "Palm9", "Palm10",
             "NonPalm1", "NonPalm2", "NonPalm3", "NonPalm4", "NonPalm5", "NonPalm6", "NonPalm7", "NonPalm8", "NonPalm9",
             "NonPalm10"]

def kullback_leibler_divergence(p, q):
    #The lower the value the better the match for this
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

def matching(refs, image):
    matched_value = 10
    matched_name = None
    lbp = local_binary_pattern(image, n_points, radius, METHOD)
    n_bins = int(lbp.max() + 1)
    #get histogram of test image
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    for name, ref in refs.items():
        #get histogram for reference image
        ref_hist, _ = np.histogram(ref, density=True, bins=n_bins, range=(0, n_bins))
        value = kullback_leibler_divergence(hist, ref_hist)
        #checking for the best match image texture
        if value < matched_value:
            matched_value = value
            matched_name = name.split("_")[0]
    return matched_name

refs = {}
i = 0
for file in PalmInput:
    infile = cv2.imread(file+".jpg",0)
    refs.update({f"Palm_{i}": local_binary_pattern(infile, n_points, radius, METHOD)})
    i += 1
i = 0
for file in NonPalmInput:
    infile = cv2.imread(file + ".jpg",0)
    refs.update({f"Non-Palm_{i}": local_binary_pattern(infile, n_points, radius, METHOD)})
    i += 1

for file in TestInput:
    infile = cv2.imread(file+".jpg",0)
    print("Image: "+file+" Best Match: "+matching(refs, infile))







