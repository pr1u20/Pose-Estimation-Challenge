import numpy as np
import json
import os
from PIL import Image
from matplotlib import pyplot as plt
import math
import random
import time
from skimage import measure
from collections import defaultdict
import cv2
from skimage.filters import sobel
from skimage import segmentation
from scipy import ndimage as ndi

# deep learning framework imports
try:
    from tensorflow.keras.utils import Sequence
    from tensorflow.keras.preprocessing import image as keras_image
    has_tf = True
except ModuleNotFoundError:
    has_tf = False

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
    has_pytorch = True
except ImportError:
    has_pytorch = False


class Camera:

    """" Utility class for accessing camera parameters. """
    
    speed_root = 'speedplus/'
    
    with open(os.path.join(speed_root, 'camera.json'), 'r') as f:
        camera_params = json.load(f)

    fx = camera_params['fx'] # focal length[m]
    fy = camera_params['fy'] # focal length[m]
    nu = camera_params['Nu'] # number of horizontal[pixels]
    nv = camera_params['Nv'] # number of vertical[pixels]
    ppx = camera_params['ppx'] # horizontal pixel pitch[m / pixel]
    ppy = camera_params['ppy'] # vertical pixel pitch[m / pixel]
    fpx = fx / ppx  # horizontal focal length[pixels]
    fpy = fy / ppy  # vertical focal length[pixels]
    k = camera_params['cameraMatrix']
    K = np.array(k) # cameraMatrix
    dcoef = camera_params['distCoeffs']


def process_json_dataset(root_dir):
    with open(os.path.join(root_dir, 'synthetic', 'train.json'), 'r') as f:
        train_images_labels = json.load(f)

    with open(os.path.join(root_dir, 'synthetic', 'validation.json'), 'r') as f:
        test_image_list = json.load(f)

    with open(os.path.join(root_dir, 'sunlamp', 'test.json'), 'r') as f:
        sunlamp_image_list = json.load(f)

    with open(os.path.join(root_dir, 'lightbox', 'test.json'), 'r') as f:
        lightbox_image_list = json.load(f)

    partitions = {'validation': [], 'train': [], 'sunlamp': [], 'lightbox': []}
    labels = {}

    for image_ann in train_images_labels:
        partitions['train'].append(image_ann['filename'])
        labels[image_ann['filename']] = {'q': image_ann['q_vbs2tango_true'], 'r': image_ann['r_Vo2To_vbs_true']}

    for image in test_image_list:
        partitions['validation'].append(image['filename'])
        labels[image['filename']] = {'q': image['q_vbs2tango_true'], 'r': image['r_Vo2To_vbs_true']}

    for image in sunlamp_image_list:
        partitions['sunlamp'].append(image['filename'])

    for image in lightbox_image_list:
        partitions['lightbox'].append(image['filename'])

    return partitions, labels


def quat2dcm(q):

    """ Computing direction cosine matrix from quaternion, adapted from PyNav. """

    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm

def convert_to_pixel_from_distance(p_cam):
    # getting homogeneous coordinates
    points_camera_frame = p_cam / p_cam[2]

    x0, y0 = (points_camera_frame[0], points_camera_frame[1])

    # apply distortion
    dist = Camera.dcoef

    r2 = x0*x0 + y0*y0
    cdist = 1 + dist[0]*r2 + dist[1]*r2*r2 + dist[4]*r2*r2*r2
    x1  = x0*cdist + dist[2]*2*x0*y0 + dist[3]*(r2 + 2*x0*x0)
    y1  = y0*cdist + dist[2]*(r2 + 2*y0*y0) + dist[3]*2*x0*y0

    # projection to image plane
    x = Camera.K[0,0]*x1 + Camera.K[0,2]
    y = Camera.K[1,1]*y1 + Camera.K[1,2]

    return x, y


def project(q, r):

        """ Projecting points to image frame to draw axes """

        # reference points in satellite frame for drawing axes
        p_axes = np.array([[0, 0, 0, 1],
                           [0.3, 0, 0, 1],
                           [0, 0.3, 0, 1],
                           [0, 0, 0.15, 1]])
        points_body = np.transpose(p_axes)

        # transformation to camera frame
        pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
        p_cam = np.dot(pose_mat, points_body)
        
        # to adjust the magnitude of vectors
        desired_lenght = (1/1.5)*np.array([0, 0.8, 0.8, 0.62])/4
        
        c = np.nan_to_num([(desired_lenght[i] / np.sum((p_cam[:, 0] - p_cam[:, i])**2))**(1/2) for i in range(4)])
        vectors = np.zeros((3,4))
        for i in range(4):
            vector = c[i]*(p_cam[:, 0] - p_cam[:, i])
            p_cam[:, i] = p_cam[:, 0] - vector
            vectors[:, i] = vector
    
        #print("Lenght: ",[sum(((p_cam[:, 0] - p_cam[:, i]))**2) for i in range(4)])
        
        # get the vectors that join the center of the spacecraft its vertices
        
        p_cam_add = np.zeros((3,8))
        l = 0
        p_cam_add[:, l] = p_cam[:, 0] + vectors[:,1] + vectors[:,2] + 0*vectors[:,3]
        l = 1
        p_cam_add[:, l] = p_cam[:, 0] - vectors[:,1] + vectors[:,2] + 0*vectors[:,3]
        l = 2
        p_cam_add[:, l] = p_cam[:, 0] + vectors[:,1] - vectors[:,2] + 0*vectors[:,3]
        l = 3
        p_cam_add[:, l] = p_cam[:, 0] - vectors[:,1] - vectors[:,2] + 0*vectors[:,3]
        l = 4
        p_cam_add[:, l] = p_cam[:, 0] + vectors[:,1] + vectors[:,2] - vectors[:,3]
        l = 5
        p_cam_add[:, l] = p_cam[:, 0] - vectors[:,1] + vectors[:,2] - vectors[:,3]
        l = 6
        p_cam_add[:, l] = p_cam[:, 0] + vectors[:,1] - vectors[:,2] - vectors[:,3]
        l = 7
        p_cam_add[:, l] = p_cam[:, 0] - vectors[:,1] - vectors[:,2] - vectors[:,3]
        
        x, y = convert_to_pixel_from_distance(p_cam)
        x_add, y_add = convert_to_pixel_from_distance(p_cam_add)

        return x, y, x_add, y_add
    
def transform_pixel_coordinates(x, y, change_size):
    
    x = x * change_size[0]
    y = y * change_size[1]
    
    return x, y
    

def find_vertices(x_add, y_add):
    centroid =(sum(x_add)/len(x_add),sum(y_add)/len(y_add))
    # Save mins and max coordinates of vertices
    min_maxs = (min(x_add), max(x_add), min(y_add), max(y_add))
    n_ver = len(x_add)
    arr1 = x_add.reshape((n_ver, 1))
    arr2 = y_add.reshape((n_ver, 1))
    arr = np.concatenate((arr1, arr2), axis=1)
    index_del = []  #Save indexes where coordinates are not vertices of polygon
    lenght_2 = True
    count = 0
    # Iterate to find the not vertices
    while lenght_2:
        for i in range(n_ver):
            point_arr = np.delete(arr, i, axis = 0)
            np.random.shuffle(point_arr)
            bool_val = is_inside_polygon(points = point_arr , p = arr[i], INT_MAX = min_maxs[1] + 1)
            if bool_val and i not in index_del: 
                index_del.append(i)
        
        count += 1

        if len(index_del) >= 2 or count > 10:
            lenght_2 = False

    arr = np.delete(arr, index_del, axis = 0) # Delete coordinates that are not vertices
    arr = arr.tolist()
    # sort by polar angle
    arr.sort(key=lambda p: math.atan2(p[1]-centroid[1],p[0]-centroid[0]))
    
    return np.array(arr), min_maxs

# Given three collinear points p, q, r,  
# the function checks if point q lies 
# on line segment 'pr' 
def onSegment(p:tuple, q:tuple, r:tuple) -> bool:
      
    if ((q[0] <= max(p[0], r[0])) &
        (q[0] >= min(p[0], r[0])) &
        (q[1] <= max(p[1], r[1])) &
        (q[1] >= min(p[1], r[1]))):
        return True
          
    return False
  
# To find orientation of ordered triplet (p, q, r). 
# The function returns following values 
# 0 --> p, q and r are collinear 
# 1 --> Clockwise 
# 2 --> Counterclockwise 
def orientation(p:tuple, q:tuple, r:tuple) -> int:
      
    val = (((q[1] - p[1]) * 
            (r[0] - q[0])) - 
           ((q[0] - p[0]) * 
            (r[1] - q[1])))
             
    if val == 0:
        return 0
    if val > 0:
        return 1 # Collinear
    else:
        return 2 # Clock or counterclock
  
def doIntersect(p1, q1, p2, q2):
      
    # Find the four orientations needed for  
    # general and special cases 
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
  
    # General case
    if (o1 != o2) and (o3 != o4):
        return True
      
    # Special Cases 
    # p1, q1 and p2 are collinear and 
    # p2 lies on segment p1q1 
    if (o1 == 0) and (onSegment(p1, p2, q1)):
        return True
  
    # p1, q1 and p2 are collinear and 
    # q2 lies on segment p1q1 
    if (o2 == 0) and (onSegment(p1, q2, q1)):
        return True
  
    # p2, q2 and p1 are collinear and 
    # p1 lies on segment p2q2 
    if (o3 == 0) and (onSegment(p2, p1, q2)):
        return True
  
    # p2, q2 and q1 are collinear and 
    # q1 lies on segment p2q2 
    if (o4 == 0) and (onSegment(p2, q1, q2)):
        return True
  
    return False
  
# Returns true if the point p lies  
# inside the polygon[] with n vertices 
def is_inside_polygon(points:list, p:tuple, INT_MAX = 10000) -> bool:
      
    n = len(points)
      
    # There must be at least 3 vertices
    # in polygon
    if n < 3:
        return False
          
    # Create a point for line segment
    # from p to infinite
    extreme = (INT_MAX, p[1])
    count = i = 0
      
    while True:
        next = (i + 1) % n
          
        # Check if the line segment from 'p' to  
        # 'extreme' intersects with the line  
        # segment from 'polygon[i]' to 'polygon[next]' 
        if (doIntersect(points[i],
                        points[next], 
                        p, extreme)):
                              
            # If the point 'p' is collinear with line  
            # segment 'i-next', then check if it lies  
            # on segment. If it lies, return true, otherwise false 
            if orientation(points[i], p, 
                           points[next]) == 0:
                return onSegment(points[i], p, 
                                 points[next])
                                   
            count += 1
              
        i = next
          
        if (i == 0):
            break
          
    # Return true if count is odd, false otherwise 
    return (count % 2 == 1)

def points_inside(min_max, arr, resolution, N_size):
    
    x_min, x_max = min_max[0], min_max[1]
    y_min, y_max = min_max[2], min_max[3]
    
    coors_sat = []
    
    values = np.zeros(N_size)
    
    lim_x_min, lim_x_max = int(x_min), int(x_max)
    lim_y_min, lim_y_max = int(y_min), int(y_max)
    
    if x_min < 0:
        lim_x_min = 0
    if x_max > N_size[0]:
        lim_x_max = N_size[0]
    if y_min < 0:
        lim_y_min = 0
    if y_max > N_size[1]:
        lim_y_max = N_size[1]
        


    for col in range(lim_x_min - int(resolution/2), lim_x_max , resolution):
        for row in range(lim_y_min - int(resolution/2), lim_y_max, resolution):
          
            bool_val = is_inside_polygon(points = arr , p = (col, row), INT_MAX = x_max + 1)
            
            if bool_val:
                coors_sat.append([col, row])
                values[int(col - (resolution/2)): int(col + (resolution/2)), int(row - (resolution/2)): int(row + (resolution/2))] = 1
            
            else: 
                values[int(col - (resolution/2)): int(col + (resolution/2)), int(row - (resolution/2)): int(row + (resolution/2))] = 0
                

    return values, coors_sat

def random_different_coordinates(coords, size_x, size_y, pad):
    """ Returns a random set of coordinates that is different from the provided coordinates, 
    within the specified bounds.
    The pad parameter avoids coordinates near the bounds."""
    good = False
    while not good:
        good = True
        c1 = random.randint(pad + 1, size_x - (pad + 1))
        c2 = random.randint(pad + 1, size_y -( pad + 1))
        if [c1, c2] in coords:
            good = False
            break
    return (c1,c2)

def extract_neighborhood(x, y, arr, radius):
    """ Returns a 1-d array of the values within a radius of the x,y coordinates given """
    # .ravel()
    return arr[(x - radius) : (x + radius + 1), (y - radius) : (y + radius + 1)]

def check_coordinate_validity(x, y, size_x, size_y, pad):
    """ Check if a coordinate is not too close to the image edge """
    return x >= pad and y >= pad and x + pad < size_x and y + pad < size_y

def generate_labeled_data(im_array, annotation, radius):
    """ For one frame and one annotation array, returns a list of labels 
    (1 for true object and 0 for false) and the corresponding features as an array.
    nb_false controls the number of false samples
    radius defines the size of the sliding window (e.g. radius of 1 gives a 3x3 window)"""
    
    features,labels = [],[]
    
    training_n = 4
    
    # True samples
    for obj in annotation[::round(len(annotation) / training_n)]:
        # For some reason the order of coordinates is inverted in the annotation files
        if check_coordinate_validity(obj[1],obj[0],im_array.shape[0],im_array.shape[1],radius):
            features.append(extract_neighborhood(obj[1],obj[0],im_array,radius))
            labels.append(1)
            
    # False samples
    for i in range(training_n * 3):
        c = random_different_coordinates(annotation,im_array.shape[1],im_array.shape[0],radius)
        features.append(extract_neighborhood(c[1],c[0],im_array,radius))
        labels.append(0)
        
    #np.stack(features,axis=1)
    return np.array(labels), np.array(features)


def get_XY(u, v, z):
    """
    Calculate the X and Y position of the centroid of the satellite. In other words, find the 
    first two values of the the position vector given the last one.
    
    Inputs: u: x coordinate for the centroid of the satellite
            v: y coordinate for the centroid of the satellite
            z: distance in the z direction to the centroid
     
    Return: X: x distance from origin (focal point) to centroid of the satellite
            Y: y distance from origin (focal point) to centroid of the satellite
    """
    
    Nu = Camera.nu
    Nv = Camera.nv

    H = Camera.K

    """Use the following function to take into account distortion and get a more accurate value of the camera matrix --->
    H, _ = cv2.getOptimalNewCameraMatrix(H, tuple(Camera.dcoef), (Nu, Nv), alpha = 0)"""
    
    H, _ = cv2.getOptimalNewCameraMatrix(H, tuple(Camera.dcoef), (Nu, Nv), alpha = 0)
    
    X =   0.8*z * ((u) - H[0][2]) / H[0][0]
    Y =  0.8*z * ((v) - H[1][2]) / H[1][1]
    
    return X, Y

def score_all(y_pred, y_true):
    """Given list of all predictions and labels calculate the pose score"""

    error_r_all = 0
    error_q_all = 0

    m = len(y_pred)

    try:
        assert(y_pred.shape == y_true.shape)

    except AttributeError:
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        assert(y_pred.shape == y_true.shape)


    for i in range(m):
        r_pred = y_pred[i][4:]
        q_pred = y_pred[i][:4]
        r_true = y_true[i][4:]
        q_true = y_true[i][:4]

        error_r = np.sum((r_true - r_pred)**2 / r_true**2)**(1/2)

        if error_r < (0.002173 * np.sum(r_true**2)**(1/2)):
            error_r = 0

        unit_vector_1 = quat2dcm(q_pred)[0]
        unit_vector_2 = quat2dcm(q_true)[0]
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        error_q  = np.arccos(dot_product)

        if error_q  < (0.169 * np.pi / 180):
            error_q = 0


        error_r_all += error_r
        error_q_all += error_q

    pose_score = (error_r_all + error_q_all) / m

    return pose_score

def orientation_error(q_true, q_pred):
    """Calculate the orientation error used in the Pose Estimation Challenge"""
    unit_vector_1 = quat2dcm(q_pred)[0]
    unit_vector_2 = quat2dcm(q_true)[0]
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    error_q  = np.arccos(dot_product)

    if error_q  < (0.169 * np.pi / 180):
        error_q = 0

    return error_q

def position_error(r_true, r_pred):
    """Calculate the position error used in the Pose Estimation Challenge"""
    error_r = np.sum((r_true - r_pred)**2 / r_true**2)**(1/2)

    if error_r < (0.002173 * np.sum(r_true**2)**(1/2)):
        error_r = 0

    return error_r

class Score():
    """Object to save the scores of individual images. Allows us to get the avarage score at any given 
    moment. Used to test our predictions on the validation dataset"""
    def __init__(self):
        self.error_r_all = 0
        self.error_q_all = 0
        self.pose_score = 0
        self.iter = 0
        
    
    def append_score(self, y_pred, y_true):
        """Add indidual score to previously processed scores"""
        
        try:
            assert(y_pred.shape == y_true.shape)

        except AttributeError:
            y_pred = np.array(y_pred)
            y_true = np.array(y_true)
            assert(y_pred.shape == y_true.shape)
        
        r_pred = y_pred[4:]
        q_pred = y_pred[:4]
        r_true = y_true[4:]
        q_true = y_true[:4]
        
        error_r = position_error(r_true, r_pred)
        error_q  = orientation_error(q_true, q_pred)
            
        self.error_r_all += error_r
        self.error_q_all += error_q
        
        self.pose_score = self.error_r_all + self.error_q_all
        
        self.iter += 1
        
    def finalize(self):
        """Show avarage scores"""
        print(f"Total pose score: {(self.pose_score / self.iter)}")
        print(f"Position score: {(self.error_r_all / self.iter)}")
        print(f"Orientation score: {(self.error_q_all / self.iter)}")
        

def threeshold_pixel(unit_inpix):
    """Input segmented image and make 1s the pixels with values larger 
    than the threeshold and 0s values lower than threeshold"""
    
    threeshold_1 = np.median(unit_inpix) + np.std(unit_inpix)*1.5
    threeshold_2 = np.max(unit_inpix) - np.max(unit_inpix)*0.3
    threeshold = (threeshold_1 + threeshold_2) / 2
    unit_inpix = np.where(unit_inpix < threeshold, 0, unit_inpix)
    unit_inpix = np.where(unit_inpix > threeshold, 1, unit_inpix)
    unit_inpix = unit_inpix
    
    return unit_inpix


def fill_spacecraft(unit_inpix, radius = 2, threeshold = 11):
    """
    The spacecraft is a regular shape, so this function gives a value of 1 to pixels that are sorrounded by 
    many filled pixels.
    """
    continue_filling = True
    filled_pix = 0

    while continue_filling:
        for x in range(radius, 50 - radius):
            for y in range(radius, 50 - radius):
                sum_surrounding = np.sum(extract_neighborhood(x, y, unit_inpix, radius))
                if sum_surrounding > threeshold:
                    unit_inpix[x, y] = 1

        if np.sum(unit_inpix) != filled_pix: 
            filled_pix = np.sum(unit_inpix)

        else: continue_filling = False
            
    return unit_inpix


def delete_object(image, pixels):
    
    """Given the pixel coordinates of an object, make its pixels equal 0"""
    
    for coor in pixels:
        image[coor[0]][coor[1]] = 0
    return image


class Clean():
    
    """Object to clean the segemented image (output from CNNs)"""
    
    def __init__(self, unit_inpix):
        self.unit_inpix = unit_inpix
        
    def delete_small_obj(self, unit_inpix):
        
        """Delete all the small objects"""
        
        centroids, object_sizes, object_dict = centroids_sizes(unit_inpix)
        object_sizes = np.array(list(object_sizes.values()))
        centroids = np.array(list(centroids.values()))
        object_dict = np.array(list(object_dict.values()))

        indexes = np.argsort(object_sizes)
        self.object_sizes = object_sizes[indexes]
        self.centroids = centroids[indexes]
        self.object_dict = object_dict[indexes]

        for obj in range(len(self.centroids) - 4):
            unit_inpix = delete_object(unit_inpix, self.object_dict[obj])

        return unit_inpix
    
    def correct_obj(self):
        
        """Choose the object that corresponds to the spacecraft"""
        
        unit_inpix = self.delete_small_obj(self.unit_inpix)
        
        if len(self.object_sizes) >= 2:
            
            if self.object_sizes[-1] > 20*self.object_sizes[-2]:
                for i in range(2, len(self.object_dict) + 1):
                    unit_inpix = delete_object(unit_inpix, self.object_dict[-i])
                
            else:
                center_pixel = np.array(unit_inpix.shape) / 2
                min_dist = 10000
                arg_min_dist = None
                
                for i in range(1, len(self.centroids)+1):
                    
                    dist = np.sum(abs(self.centroids[-i] - center_pixel))
                    
                    if dist <= min_dist:
                        min_dist = dist
                        arg_min_dist = i
                        
                
                for i in range(1, len(self.centroids)+1):

                    if i != arg_min_dist:
                        unit_inpix = delete_object(unit_inpix, self.object_dict[-i])
        
        return unit_inpix
    
def borders_1(avg_unipix):
    
    """Find the borders of segemented image in a specififc way and fill the inside to 
    find the spacecraft"""
    
    unit_inpix = ndi.gaussian_filter(avg_unipix, 3)
    sobel_pix = sobel(cv2.resize(avg_unipix, (50,50)))
    markers = np.zeros_like(sobel_pix)
    markers[sobel_pix < np.median(sobel_pix) + np.std(sobel_pix)*1.8] = 1
    markers[sobel_pix > np.median(sobel_pix) + np.std(sobel_pix)*1.8] = 2
    markers = segmentation.watershed(sobel_pix, markers)
    unit_inpix = markers - 1
    #unit_inpix = canny(markers)
    unit_inpix = ndi.binary_fill_holes(unit_inpix).astype(int)
    #unit_inpix = fill_spacecraft(unit_inpix)

    Clean_obj = Clean(unit_inpix)
    unit_inpix = Clean_obj.delete_small_obj(unit_inpix)
    unit_inpix_1 = Clean_obj.correct_obj()
    
    return unit_inpix_1

def borders_2(unit_inpix_50_1):
    
    """Find the borders of segemented image in a specififc way and fill the inside to 
    find the spacecraft"""
    
    #unit_inpix = ndi.gaussian_filter(unit_inpix_50_1, 3)
    sobel_pix = sobel(unit_inpix_50_1)
    markers = np.zeros_like(sobel_pix)
    markers[sobel_pix < np.median(sobel_pix) + np.std(sobel_pix)*0.3] = 1
    markers[sobel_pix > np.median(sobel_pix) + np.std(sobel_pix)*0.3] = 2
    markers = segmentation.watershed(sobel_pix, markers)
    unit_inpix = markers - 1
    #unit_inpix = canny(markers)
    unit_inpix = ndi.binary_fill_holes(unit_inpix).astype(int)
    #unit_inpix = fill_spacecraft(unit_inpix)

    Clean_obj = Clean(unit_inpix)
    unit_inpix = Clean_obj.delete_small_obj(unit_inpix)
    unit_inpix_2 = Clean_obj.correct_obj()
    
    return unit_inpix_2

def borders_3(unit_inpix_60_1):
    
    """Find the borders of segemented image in a specififc way and fill the inside to 
    find the spacecraft"""
    
    #unit_inpix = ndi.gaussian_filter(unit_inpix_60_1, 3)
    sobel_pix = sobel(cv2.resize(unit_inpix_60_1, (50,50)))
    markers = np.zeros_like(sobel_pix)
    markers[sobel_pix < np.median(sobel_pix) + np.std(sobel_pix)*1.5] = 1
    markers[sobel_pix > np.median(sobel_pix) + np.std(sobel_pix)*1.5] = 2
    markers = segmentation.watershed(sobel_pix, markers)
    unit_inpix = markers - 1
    #unit_inpix = canny(markers)
    unit_inpix = ndi.binary_fill_holes(unit_inpix).astype(int)
    #unit_inpix = fill_spacecraft(unit_inpix)

    Clean_obj = Clean(unit_inpix)
    unit_inpix = Clean_obj.delete_small_obj(unit_inpix)
    unit_inpix_3 = Clean_obj.correct_obj()
    
    return unit_inpix_3





    
class SatellitePoseEstimationDataset:

    """ Class for dataset inspection: easily accessing single images, and corresponding ground truth pose data. """

    def __init__(self, newsize = None, root_dir='speedplus/'):
        self.partitions, self.labels = process_json_dataset(root_dir)
        self.root_dir = root_dir  # path to speed+'
        self.newsize = newsize
        self.oldsize = (1920, 1200)
        self.change_size =  np.array(self.newsize) / np.array(self.oldsize)
        
    def get_image(self, i=0, split='train', rgb = False):

        """ Loading image as PIL image. """
        self.split = split

        img_name = self.partitions[self.split][i]
        if split=='train':
            img_name = os.path.join(self.root_dir, 'synthetic','images', img_name)
        elif split=='validation':
            img_name = os.path.join(self.root_dir, 'synthetic','images', img_name)
        elif split=='sunlamp':
            img_name = os.path.join(self.root_dir, 'sunlamp','images', img_name)
        elif split=='lightbox':
            img_name = os.path.join(self.root_dir, 'lightbox','images', img_name)
        else:
            print()
            # raise error?
        
        image = Image.open(img_name)
        if self.newsize != None: image = image.resize(self.newsize)
        if rgb == True: image = image.convert('RGB')
        
        return image

    def get_pose(self, i=0, partition = "train"):

        """ Getting pose label for image. """

        img_id = self.partitions[partition][i]
        q, r = self.labels[img_id]['q'], self.labels[img_id]['r']
        return q, r

    def visualize(self, i, partition='train', ax=None):

        """ Visualizing image, with ground truth pose with axes projected to training image. """

        if ax is None:
            ax = plt.gca()
        img = self.get_image(i)
        ax.imshow(img)

        # no pose label for test
        if partition == 'train':
            q, r = self.get_pose(i)
            xa, ya, x_add, y_add = project(q, r)
            xa, ya = transform_pixel_coordinates(xa, ya, self.change_size)
            x_add, y_add = transform_pixel_coordinates(x_add, y_add, self.change_size)
            ax.plot(x_add, y_add, "or")
            width_head = 30 * self.change_size[0]
            ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=width_head , color='r')
            ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=width_head, color='g')
            ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=width_head, color='b')

        return
    
    def visualize_unit(self, i, show_im = False, split = "train"):
        
        """Visulaize image of 1s and 0s. 1s orresponding to spacecraft and 0 to background."""
        #t1 = time.time()
        
        #print("Get pose ...")
        q, r = self.get_pose(i, split)
        
        #print("Getting dots ...")
        xa, ya, x_add, y_add = project(q, r)
        
        xa, ya = transform_pixel_coordinates(xa, ya, self.change_size)
        x_add, y_add = transform_pixel_coordinates(x_add, y_add, self.change_size)
        #print("Finding vertices ...")
        verts, min_maxs = find_vertices(x_add, y_add)
        #plt.plot(arr[:,0], arr[:,1], "or")
        #plt.show()
        #print("Processing ...")

        values = measure.grid_points_in_poly(self.newsize, verts)
        
        coords = np.column_stack(np.where(values == 1))
        
        #print("Plotting ... ")
        if show_im: plt.imshow(values.T)
        #t2 = time.time()
        #print(t2 - t1)
        return values, coords
    
    def generate_labeled_set(self, radius, batch = None, partition = "train"):
        
        """Generate labeled data for a list of sequences in a given path"""
        "Set batch equal to None if you want to download data from all the images in the Dataset"
        
        labels,features = [],[]
        
        training_lenght = len(self.partitions[partition])
        
        if batch == None:
            # If no batch size given, the batch will become the number of training examples
            print("Dataset occupies a lot of space. Make sure you can download it.")
            
            batch = training_lenght
        
        for num_im in range(batch):

                #if batch is given the image number becomes random
                if batch != None: num_im = random.randint(0, training_lenght - 1)
                
                _, coor_sat = self.visualize_unit(num_im, show_im = False, split = partition)
             
                d = generate_labeled_data(np.array(self.get_image(num_im, split = partition)),
                                        coor_sat,
                                        radius)
                labels.append(d[0])
                features.append(d[1])
                
        
        labels = np.concatenate(labels,axis=0)
        features = np.expand_dims(np.concatenate(features,axis=0), axis = -1)
        
        if batch == None:
            
            labels_file = f"Labels_{self.newsize[0]}_{self.newsize[1]}_{batch}_{radius}"
            features_file = f"Features_{self.newsize[0]}_{self.newsize[1]}_{batch}_{radius}"
            
            np.save(labels_file, labels)
            np.save(features_file, features)
            
        # Return features and labels used to train ML models.
        return features, labels
    
    def guess_unit(self, q, r, show_im = False):
        
        """Given orientaion and position of spacecraft, project it into image and show
        and image of 1s (where the spacraft is) and 0s (background) """
        
        # Get the position in the image of the 8 vertices of the 3d spacecraft
        _, _, x_add, y_add = project(q, r)
        
        # convert vertices pixel position from original image to the reduced image
        x_add, y_add = transform_pixel_coordinates(x_add, y_add, self.change_size)
        
        # The 3d spacecraft projected into 2d can be modelled by a polygon of 6 or 4 vertices.
        # We need to find the vertices that correspond to the edges of the 2d polygon and get rid of edges inside polygon.
        
        verts, min_maxs = find_vertices(x_add, y_add)


        # Given vertices, generate and image with 1s corresponding to spacecraft.
        values = measure.grid_points_in_poly(self.newsize, verts)
        
        values = values.T

        if show_im: plt.imshow(values)

        return values
    

    
    
def centroids_sizes(pred, bg = 0):
    """Given an image of 1s and 0s, caclulate the centroids and sizes of all objects"""
    
    conn_comp=measure.label(pred, background=bg)
    object_dict=defaultdict(list) #Keys are the indices of the connected components and values are arrrays of their pixel coordinates 
    for (x,y),label in np.ndenumerate(conn_comp):
            if label != bg:
                object_dict[label].append([x,y])
    # Mean coordinate vector for each object, except the "0" label which is the background
    centroids={label: np.mean(np.stack(coords),axis=0) for label,coords in object_dict.items()}
    object_sizes={label: len(coords) for label,coords in object_dict.items()}
    
    return centroids, object_sizes, object_dict


class Position():
    """Calculate centroids and extract the coordinates that correspond to the spacecraft."""
    def __init__(self, pred):
        self.pred = pred
        width = len(pred)
        im_size = (width, width)
        dataset_root_dir ='speedplus/' # path to speed+'
        self.dataset = SatellitePoseEstimationDataset(root_dir=dataset_root_dir, newsize = im_size)
        

    def extract_centroid(self, pred):
        """From an image of 0s and 1s return the coordinates of the centroid of the largest body (satellite)"""
        centroids, object_sizes, _ = centroids_sizes(pred, bg = 0)

        index = np.argmax(list(object_sizes.values())) # Get index of the coordinates of the largest body (satellite)

        self.centroid = centroids[index + 1]
    
    def pixel_equivalent(self):
        """Convert centroid of image of specific size (ex. (100, 100)) to the corresponding point
        on an image of size (1920, 1200)"""
        self.centroid = self.centroid.tolist()
        self.centroid.reverse()
        self.u, self.v = self.centroid / self.dataset.change_size
    
    
    def XYZ_components(self, z):
        """Given the z distance to the satellite, return the x and y componets in meters"""
        self.extract_centroid(self.pred)
        self.pixel_equivalent()
        
        X, Y = get_XY(self.u, self.v, z)
        
        return np.array([X, Y, z])
    

if has_pytorch:
    class PyTorchSatellitePoseEstimationDataset(Dataset):

        """ SPEED dataset that can be used with DataLoader for PyTorch training. """

        def __init__(self, split='train', speed_root='datasets/', transform=None):

            if not has_pytorch:
                raise ImportError('Pytorch was not imported successfully!')

            if split not in {'train', 'validation', 'sunlamp', 'lightbox'}:
                raise ValueError('Invalid split, has to be either \'train\', \'validation\', \'sunlamp\' or \'lightbox\'')

            if split in {'train', 'validation'}:
                self.image_root = os.path.join(speed_root, 'synthetic', 'images')
                with open(os.path.join(speed_root, "synthetic", split + '.json'), 'r') as f:
                    label_list = json.load(f)
            else:
                self.image_root = os.path.join(speed_root, split, 'images')
                with open(os.path.join(speed_root, split, 'test.json'), 'r') as f:
                    label_list = json.load(f)

            self.sample_ids = [label['filename'] for label in label_list]
            self.train = split == 'train'

            if self.train:
                self.labels = {label['filename']: {'q': label['q_vbs2tango_true'], 'r': label['r_Vo2To_vbs_true']}
                               for label in label_list}
            self.split = split
            self.transform = transform

        def __len__(self):
            return len(self.sample_ids)

        def __getitem__(self, idx):
            sample_id = self.sample_ids[idx]
            img_name = os.path.join(self.image_root, sample_id)

            # note: despite grayscale images, we are converting to 3 channels here,
            # since most pre-trained networks expect 3 channel input
            pil_image = Image.open(img_name).convert('RGB')

            if self.train:
                q, r = self.labels[sample_id]['q'], self.labels[sample_id]['r']
                y = np.concatenate([q, r])
            else:
                y = sample_id

            if self.transform is not None:
                torch_image = self.transform(pil_image)
            else:
                torch_image = pil_image

            return torch_image, y
else:
    class PyTorchSatellitePoseEstimationDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError('Pytorch is not available!')

if has_tf:
    class KerasDataGenerator(Sequence):

        """ DataGenerator for Keras to be used with fit_generator (https://keras.io/models/sequential/#fit_generator)"""

        def __init__(self, preprocessor, label_list, speed_root, batch_size=32, dim=(224, 224), n_channels=3, shuffle=True):

            # loading dataset
            self.image_root = os.path.join(speed_root, 'synthetic', 'images')

            # Initialization
            self.preprocessor = preprocessor
            self.dim = dim
            self.batch_size = batch_size
            self.labels = self.labels = {label['filename']: {'q': label['q_vbs2tango_true'], 'r': label['r_Vo2To_vbs_true']}
                                         for label in label_list}
            self.list_IDs = [label['filename'] for label in label_list]
            self.n_channels = n_channels
            self.shuffle = shuffle
            self.indexes = None
            self.on_epoch_end()

        def __len__(self):

            """ Denotes the number of batches per epoch. """

            return int(np.floor(len(self.list_IDs) / self.batch_size))

        def __getitem__(self, index):

            """ Generate one batch of data """

            # Generate indexes of the batch
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

            # Find list of IDs
            list_IDs_temp = [self.list_IDs[k] for k in indexes]

            # Generate data
            X, y = self.__data_generation(list_IDs_temp)

            return X, y

        def on_epoch_end(self):

            """ Updates indexes after each epoch """

            self.indexes = np.arange(len(self.list_IDs))
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def __data_generation(self, list_IDs_temp):

            """ Generates data containing batch_size samples """

            # Initialization
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
            y = np.empty((self.batch_size, 7), dtype=float)

            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                img_path = os.path.join(self.image_root, ID)
                img = keras_image.load_img(img_path, target_size=(224, 224))
                x = keras_image.img_to_array(img)
                x = self.preprocessor(x)
                X[i,] = x

                q, r = self.labels[ID]['q'], self.labels[ID]['r']
                y[i] = np.concatenate([q, r])

            return X, y
else:
    class KerasDataGenerator:
        def __init__(self, *args, **kwargs):
            raise ImportError('tensorflow.keras is not available! Please install tensorflow.')