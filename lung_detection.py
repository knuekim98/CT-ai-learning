import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os
import os.path
import scipy.ndimage
from skimage import measure, morphology, segmentation, transform # image processing
import pandas as pd
import cv2
from matplotlib.patches import Circle


def read_patient_data(patient_folder):
    patient_data = []
    for dcmfile in os.listdir(patient_folder):
        patient_data.append([pydicom.read_file(patient_folder + '/' + dcmfile), int(dcmfile[2:5])])
    patient_data.sort(key = lambda slice: float(slice[0].ImagePositionPatient[2]))
    return patient_data
    

mode = 'val'    
data_folder = f'./RIDER Lung CT/{mode}-data/'
patients = os.listdir(data_folder)

'''
idx = 95
new_idx = 0
patient_data_sample = read_patient_data(data_folder + patients[0])
print(patient_data_sample[idx]) # print out the metadata of a sample slice

for i in range(len(patient_data_sample)):
    if patient_data_sample[i][1] == idx:
        new_idx = i
        break
print(new_idx)
final_idx = new_idx

patient_data_sample = list(zip(*patient_data_sample))[0]
'''

#plt.imshow(patient_data_sample[new_idx].pixel_array, cmap=plt.cm.bone) # slice image
#plt.show()

def get_hounsfield_unit_array(patient_data):
    nx = patient_data[0].Rows
    ny = patient_data[0].Columns
    nz = len(patient_data)
    hu_array = np.zeros((nx, ny, nz), dtype=np.float32)
    for slice_index in range(len(patient_data)):
        intercept = patient_data[slice_index].RescaleIntercept
        slope = patient_data[slice_index].RescaleSlope
        pixel_array = patient_data[slice_index].pixel_array
        pixel_array[pixel_array == -2000] = 0
        hu_array[:,:,slice_index] = slope * pixel_array + intercept
        
    return hu_array.astype(np.int16)

#hu_array_sample = get_hounsfield_unit_array(patient_data_sample)
#print('Maximum HU: {:d}, Minmum HU: {:d}'.format(np.amax(hu_array_sample), np.amin(hu_array_sample)))


def resample_data(patient_data, new_spacings):
    # obtain original spacing in z-direction
    original_z_spacing = np.abs(patient_data[0].ImagePositionPatient[2]
                                - patient_data[1].ImagePositionPatient[2])
    # obtain rescaled HU array
    hu_array = get_hounsfield_unit_array(patient_data)
    
    original_spacings = np.array([float(patient_data[0].PixelSpacing[0]), float(patient_data[0].PixelSpacing[1]), original_z_spacing], dtype='float32')
    new_shape = np.round(hu_array.shape * original_spacings / new_spacings)
    zoom_factor = new_shape / hu_array.shape
    #print(zoom_factor)
    
    return scipy.ndimage.zoom(hu_array, zoom_factor, mode='nearest')

new_spacings = [1, 1, 1] # unit: mm
'''
resampled_patient_data_sample = resample_data(patient_data_sample, new_spacings)
new_idx *= zoom_z
new_idx = round(new_idx)
'''
#print(new_idx)

'''
plt.imshow(resampled_patient_data_sample[:,:,new_idx], cmap=plt.cm.bone)
plt.show()
'''

def create_markers(data):
    # create lung markers
    lung_marker = data < -400
    # remove the outside air
    for s in range(data.shape[2]):
        lung_marker[:,:,s] = segmentation.clear_border(lung_marker[:,:,s])
    lung_marker_labels = measure.label(lung_marker)
    potential_lung_regions = measure.regionprops(lung_marker_labels)
    potential_lung_regions.sort(key = lambda region: region.area)

    # only retain the largest regions (ideally corresponding to the lung)
    #if (len(potential_lung_regions) > 1):
    #    assert (potential_lung_regions[-1].area > 5 * potential_lung_regions[-2].area)
    for region in potential_lung_regions[:-1]:
        coords_tuple = tuple([tuple(coord) for coord in region.coords.transpose()])
        lung_marker_labels[coords_tuple] = 0
    lung_marker = lung_marker_labels > 0

    # create outside markers
    outside_marker_inbound = scipy.ndimage.binary_dilation(lung_marker, iterations=8)
    outside_marker_outbound = scipy.ndimage.binary_dilation(lung_marker, iterations=35)
    outside_marker = outside_marker_outbound ^ outside_marker_inbound
    
    # create watershed markers (lung: 2, outside: 1)
    watershed_marker = lung_marker.astype(np.int16) * 2 + outside_marker.astype(np.int16) * 1
    
    return lung_marker, outside_marker, watershed_marker

#lung_marker, outside_marker, watershed_marker = create_markers(resampled_patient_data_sample)

'''
plt.figure(figsize=(10,5))
sample_slice_index = 100
plt.subplot(131)
plt.imshow(lung_marker[:,:,sample_slice_index], cmap=plt.cm.bone)
plt.title('lung marker')
plt.subplot(132)
plt.imshow(outside_marker[:,:,sample_slice_index], cmap=plt.cm.bone)
plt.title('outside marker')
plt.subplot(133)
plt.imshow(watershed_marker[:,:,sample_slice_index], cmap=plt.cm.bone)
plt.title('watershed marker')
plt.show()
'''

def slice_lung_segmentation(slice, lung_marker, watershed_marker):
    # find edges with Sober filters
    sober_x = scipy.ndimage.sobel(slice, axis=0)
    sober_y = scipy.ndimage.sobel(slice, axis=1)
    sober = np.hypot(sober_x, sober_y)

    # apply watershed algorithm to find watershed basins flooded from the markers
    watershed = segmentation.watershed(sober, watershed_marker)
    
    # outline of watershed
    border = scipy.ndimage.morphological_gradient(watershed, size=(3,3)).astype(bool)
    
    # apply black tophat filter to include possible nodules near the lung border
    black_tophat = [[0, 0, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 0, 0]]
    black_tophat = scipy.ndimage.iterate_structure(black_tophat, 5)
    border += scipy.ndimage.black_tophat(border, structure=black_tophat)
    
    # combine the lung and the regions near its border which may contain nodules
    lung = np.bitwise_or(lung_marker, border)
    # remove the small air pockets inside the lung
    # lung = scipy.ndimage.morphology.binary_closing(lung, structure=np.ones((5,5)), iterations=4)
    lung = scipy.ndimage.binary_fill_holes(lung)
    
    return lung, sober, watershed, border

'''
lung_marker, outside_marker, watershed_marker = create_markers(resampled_patient_data_sample)

sample_slice_index = new_idx
sample_slice = resampled_patient_data_sample[:,:,sample_slice_index]
lung, sober, watershed, border = slice_lung_segmentation(
    sample_slice,
    lung_marker[:,:,sample_slice_index],
    watershed_marker[:,:,sample_slice_index],
    )

plt.figure(figsize=(10,5))
plt.subplot(231)
plt.imshow(sober, cmap=plt.cm.bone)
plt.title('sober filter')
plt.subplot(232)
plt.imshow(watershed, cmap=plt.cm.bone)
plt.title('watershed')
plt.subplot(233)
plt.imshow(border, cmap=plt.cm.bone)
plt.title('border')
plt.subplot(234)
plt.imshow(lung, cmap=plt.cm.bone)
plt.title('final lung region')
plt.subplot(235)
segmented_slice = np.ones(sample_slice.shape) * 30
segmented_slice[lung] = sample_slice[lung]
plt.imshow(segmented_slice, cmap=plt.cm.bone)
plt.title('segmented lung')

plt.tight_layout()
plt.show()
'''

def lung_segmentation(data):
    # set HU outside the lung region to 30 (similar to the tissue HU around the lung)
    segmented_lung = np.ones(data.shape) * 30
    segmented_lung_marker = np.zeros(data.shape)
    
    lung_marker, outside_marker, watershed_marker = create_markers(data)
    
    for slice_index in range(data.shape[2]):
        slice_marker,_,_,_ = slice_lung_segmentation(data[:,:,slice_index],
                                                     lung_marker[:,:,slice_index],
                                                     watershed_marker[:,:,slice_index])
        segmented_lung_marker[:,:,slice_index] = slice_marker
    segmented_lung[segmented_lung_marker.astype(bool)] = data[segmented_lung_marker.astype(bool)]

    # crop the blank borders
    '''
    for x_begin in range(data.shape[0]):
        slice = segmented_lung_marker[x_begin,:,:]
        if not np.array_equal(slice, np.zeros(slice.shape)): break
    for x_end in range(-1,-data.shape[0]-1,-1):
        slice = segmented_lung_marker[x_end,:,:]
        if not np.array_equal(slice, np.zeros(slice.shape)): break
    for y_begin in range(data.shape[1]):
        slice = segmented_lung_marker[:,y_begin,:]
        if not np.array_equal(slice, np.zeros(slice.shape)): break
    for y_end in range(-1,-data.shape[1]-1,-1):
        slice = segmented_lung_marker[:,y_end,:]
        if not np.array_equal(slice, np.zeros(slice.shape)): break
    for z_begin in range(data.shape[2]):
        slice = segmented_lung_marker[:,:,z_begin]
        if not np.array_equal(slice, np.zeros(slice.shape)): break
    for z_end in range(-1,-data.shape[2]-1,-1):
        slice = segmented_lung_marker[:,:,z_end]
        if not np.array_equal(slice, np.zeros(slice.shape)): break

    if x_end == -1: x_end = -2
    if y_end == -1: y_end = -2
    if z_end == -1: z_end = -2

    return segmented_lung[x_begin:(x_end+1),y_begin:(y_end+1),z_begin:(z_end+1)]
    '''
    return segmented_lung

'''
segmented_lung_sample = lung_segmentation(resampled_patient_data_sample)
plt.imshow(segmented_lung_sample[:,:,new_idx], cmap=plt.cm.bone)
plt.show()
'''

def reshape_data(segmented_lung, desired_shape):
    shape = list(segmented_lung.shape)
    diff = np.array(desired_shape) - np.array(shape)
    pad_width = []
    for i in range(3):
        before = (abs(diff[i])/2).astype(np.int16)
        after = abs(diff[i]) - before
        # make sure the size is not larger than the desired size
        # if (i != 2): assert (diff[i] >= 0), \
        #    "shape %s is larger than desired shape %s for axis %s" % (shape[i], desired_shape[i], i)
        if diff[i] >= 0:
            pad_width.append((before, after))
        else:
            pad_width.append((0,0))
            # trim the array when the actual size is greater than the desrized size
            if i == 0: segmented_lung = segmented_lung[before:(-after),:,:]
            if i == 1: segmented_lung = segmented_lung[:,before:(-after),:]
            # trim the bottom part since studies show that lung nodules are less likely to
            # be cancerous in bottom and middle lobes
            if i == 2: segmented_lung = segmented_lung[:,:,abs(diff[i]):]
                
            print('Data trimmed for axis ' + str(i))

    # pad constant values when the actual size is less than the desired size
    reshaped_data = np.lib.pad(segmented_lung, tuple(pad_width), 'constant', constant_values=30)
    assert (list(reshaped_data.shape) == desired_shape), \
        'data shape is (%s, %s, %s) but desired shape is (%s, %s, %s)' \
        % (reshaped_data.shape[0], reshaped_data.shape[1], reshaped_data.shape[2], \
        desired_shape[0], desired_shape[1], desired_shape[2])
    
    return reshaped_data

desired_shape = [350, 350, 400]
'''
reshaped_sample = reshape_data(segmented_lung_sample, desired_shape)

final_idx *= (400/len(patient_data_sample))
final_idx = round(final_idx)
'''

def normalize_data(reshaped_data, min_hu, max_hu):
    normalized_data = (reshaped_data - min_hu) / (max_hu - min_hu)
    normalized_data[normalized_data < 0.] = 0.
    normalized_data[normalized_data > 1.] = 1.
    return normalized_data
    
min_hounsfield_unit = -1000.
max_hounsfield_unit = 400.

'''
normalized_sample = normalize_data(reshaped_sample,
                                   min_hounsfield_unit,
                                   max_hounsfield_unit)
'''

mean = 0.3
def zero_centering(data, mean):
    return data - mean

'''
preprocessed_sample = zero_centering(normalized_sample, mean)
plt.imshow(preprocessed_sample[:,:,final_idx], cmap=plt.cm.bone)
plt.show()
'''


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


import tensorflow as tf
writer_train = tf.io.TFRecordWriter(f'./RIDER Lung CT/tfrecord/{mode}.tfr')
with open(data_folder+'ans.txt', 'r') as f:
    ans = eval(f.read())
for patient in patients[:-1]:
    idxlist = {}

    patient_data = read_patient_data(data_folder + patient)
    for idx in ans[patient].keys():
        for i in range(len(patient_data)):
            if patient_data[i][1] == idx:
                idxlist[idx]=round(i*300/len(patient_data))
                break
    print(idxlist)

    patient_data = list(zip(*patient_data))[0]
    resampled_patient_data = resample_data(patient_data, new_spacings)
    print("resample done")

    segmented_lung = lung_segmentation(resampled_patient_data)
    print("segment done")

    ds = [224., 224., 300.]
    factor = (ds[0]/segmented_lung.shape[0], ds[1]/segmented_lung.shape[1], ds[2]/segmented_lung.shape[2])
    print(factor)
    reshaped_patient_data = scipy.ndimage.zoom(segmented_lung, factor, mode='nearest')

    normalized_patient_data = normalize_data(reshaped_patient_data,
                                      min_hounsfield_unit,
                                      max_hounsfield_unit)
    preprocessed_patient_data = zero_centering(normalized_patient_data, mean)
    
    for i in idxlist.keys():
        img = preprocessed_patient_data[:,:,idxlist[i]]
        img_min = np.min(img)
        img_max = np.max(img)
        img = img - img_min
        img = img / (img_max-img_min)
        img *= 2**8-1
        img = img.astype(np.uint8)
        

        x_value = ans[patient][i][0]*224/512
        y_value = ans[patient][i][1]*224/512

        example = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(img.tobytes()),
            'x': _float_feature(x_value),
            'y': _float_feature(y_value)
        }))
        writer_train.write(example.SerializeToString())

writer_train.close()
