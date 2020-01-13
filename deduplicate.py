import glob
from PIL import Image
import cv2
import os
from skimage.measure import compare_ssim
import numpy as np
import sys

# Rotate images if needed
# input a list of files path
# return a list of correct oriented files path
def rotate_image(input_path_list):
    try:
        output_path_list = list()
        extension = input_path_list[0][-4:]
        for path_img in input_path_list:
            image = Image.open(path_img)
            if hasattr(image, '_getexif'):
                orientation = 0x0112
                exif = image._getexif()
                if exif is not None:
                    orientation = exif[orientation]
                    rotations = {
                        3: Image.ROTATE_180,
                        6: Image.ROTATE_270,
                        8: Image.ROTATE_90
                    }
                    if orientation in rotations:
                        image = image.transpose(rotations[orientation])
            path_img_rotate = path_img.replace(extension,"_rotate"+extension)
            image.save(path_img_rotate)
            output_path_list.append(path_img_rotate)
        return output_path_list
    except Exception as e:
        print("Error in rotate image : {}".format(e))
        return []

# Remove white band around images
# input a list of files path
# return a list of no white bands files path
def remove_white_bands(input_path_list):
    try:
        output_path_list = list()
        extension = input_path_list[0][-4:]
        for path_img in input_path_list:
            img = cv2.imread(path_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_blur = cv2.bilateralFilter(img, d = 7, sigmaSpace = 75, sigmaColor =75)
            img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
            a = img_gray.max()
            _,thresh = cv2.threshold(img_gray, a/2+120, a,cv2.THRESH_BINARY_INV)
            _,contours,_ = cv2.findContours(image = thresh, mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key = cv2.contourArea, reverse = True)
            c_0 = contours[0]
            x, y, w, h = cv2.boundingRect(c_0)
            img_copy = img.copy()
            tmp_img = img[y:y+h+1, x:x+w+1]
            path_img_crop = path_img.replace(extension,"_crop"+extension)
            im = Image.fromarray(tmp_img)
            im.save(path_img_crop)
            output_path_list.append(path_img_crop)
        return output_path_list
    except Exception as e:
        print("Error in remove white bands : {}".format(e))
        return []

# Resize images to compare matrices of same shape
# input a list of files path
# return a list of resized files path
def resize_images(input_path_list):
    try:
        output_path_list = list()
        extension = input_path_list[0][-4:]
        size = max([Image.open(path_img).size for path_img in input_path_list])
        for path_img in input_path_list:
            img = Image.open(path_img)
            img = img.resize(size, Image.ANTIALIAS)
            path_img_resize = path_img.replace(extension,"_resize"+extension)
            img.save(path_img_resize)
            output_path_list.append(path_img_resize)
        return output_path_list
    except Exception as e:
        print("Error in resize images : {}".format(e))
        return []

# Preprocessing step for both images
# input a list of files paths
# return a list of preprocessed files paths
def preprocessing_images(input_path_list):
    rotate_path_list = rotate_image(input_path_list)
    nowhite_path_list = remove_white_bands(rotate_path_list)
    output_path_list = resize_images(nowhite_path_list)
    return output_path_list

# Mean squared error between images
# input two matrices
# return mean square error between then
def mse(matA, matB):
    # the lower the error, the more "similar" the two images are
    err = np.sum((matA.astype("float") - matB.astype("float")) ** 2)
    err = err/float(matA.shape[0] * matA.shape[1])
    return err
 
# Basic comparison step for both images
# input two matrices
# return mean square error and structural similary
def basic_metrics(matA, matB):
    m = mse(matA, matB)
    s = compare_ssim(matA, matB,multichannel=True)
    return m,s

# Compute key points matching
# input two matrices
# return the number of keypoints
def extract_size_match(matA, matB):
    try:
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(matA,None)
        kp2, des2 = sift.detectAndCompute(matB,None)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        vect1 = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        vect2 = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        return len(vect1)
    except Exception as e:
        print("Error in extract size match : {}".format(e))
        exit()


# Remove temporary created files
# input a list of files paths
# return nothing
def clean_folder(input_path_list):
    for path_elt in input_path_list:
        extension = path_elt[-4:]
        template = ".".join(path_elt.split(".")[:-1])+"_*"+extension
        files_remove = glob.glob(template)
        for filepath in files_remove:
            os.remove(filepath)

# Main function for the comparison of two images
# input two images paths
# return True if both images are similar, False otherwise
def main(path_img_1,path_img_2):
    file_list = preprocessing_images([path_img_1,path_img_2])
    if file_list != []:
        img1 = cv2.imread(file_list[0])
        img2 = cv2.imread(file_list[1])
        m,s = basic_metrics(img1, img2)
        clean_folder([path_img_1,path_img_2])
        if m == 0:
            return True
        #use threshold
        elif s > 0.45:
            return True
        else:
            size_m_kp = extract_size_match(img1, img2)
            #use threshold
            if size_m_kp >= 10:
                return True
            else:
                return False
    else:
        return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Wrong number of arguments")
        exit()
    else:
        input_img1 = sys.argv[1]
        input_img2 = sys.argv[2]
        if os.path.exists(input_img1) and os.path.exists(input_img2):
            match = main(input_img1,sys.argv[2])
            print(match)
        else:
            print("Input path does not exist")
            exit()