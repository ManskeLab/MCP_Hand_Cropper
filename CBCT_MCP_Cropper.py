import SimpleITK as sitk
import numpy as np
import os
import argparse
import shutil

from itertools import filterfalse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hand_image_path", type=str, help="Image (path + filename)")
    args = parser.parse_args()

    hand_image_path = args.hand_image_path
    hand_image = sitk.Normalize(sitk.ReadImage(hand_image_path))
    print("Image read!")

    region_based_levelset(hand_image)

    # print("Starting Crop...")
    # region_based_levelset(hand_image)

    return

def get_means(slice_image, vert1, vert2):

    slice = sitk.Image(slice_image)
    rows = iter(range(slice.GetHeight()))
    cols = range(slice.GetWidth())

    sum_in = 0
    num_vox_in = 0
    sum_out = 0
    num_vox_out = 0
    for row in rows:
        for col in cols:
            if ((col>=vert1[0] and col<=vert2[0]) and (row>=vert1[1] and row<=vert2[1])):
                sum_in += slice[col, row] 
                num_vox_in += 1
            else:
                sum_out += slice[col, row] 
                num_vox_out += 1
                slice[col, row] = -5


    if num_vox_in == 0:
        mean_in = False
    else:
        mean_in = sum_in/num_vox_in

    if num_vox_out == 0:
        mean_out = False
    else:
        mean_out = sum_out/num_vox_out
    sitk.WriteImage(slice, os.path.join(temp_dir, 'region_inv'+str(mean_in)+'.nii'))

    return mean_in, mean_out

def get_edge_max(slice_image, vert1, vert2):

    slice_image = sitk.GetArrayFromImage(slice_image)
    sub_vert = vert1-vert2
    axis = np.where(sub_vert==0)[0]

    if not (len(axis) == 1):
        raise ValueError("Inputs do not exist along a bounding box's edge.")
    
    axis = int(not(axis[0]))
    sliced_range = 0

    if (sub_vert[axis] < 0):
        sliced_range = slice(vert1[axis], vert2[axis]+1)
    else:
        sliced_range = slice(vert2[axis], vert1[axis]+1)

    sum_max_5 = 0
    edge = 0
    if axis:
        edge = slice_image[sliced_range, vert1[0]]
    else:
        edge = slice_image[vert1[1], sliced_range]

    edge= np.sort(edge)
    sum_max_5 = np.mean(edge)
        
    # for i in range(5):
    #     sum_max_5 += edge[-1-i]
    
    return sum_max_5

def calc_levelset(levelset, slice_image, vertex_low_temp, vertex_high_temp, mean_in_new, mean_out_new, width, height):

    speed_cols = 0
    for col in range(width):
        end1 = np.array([col, 0])
        end2 = np.array([col, height])

        col_rep = get_edge_max(slice_image, end1, end2)
        if levelset[0, col] > 0:
            speed_cols += col_rep - mean_in_new
        elif levelset[0, col] < 0:
            speed_cols += col_rep - mean_in_new

    speed_rows = 0
    for row in range(height):
        end1 = np.array([0, row])
        end2 = np.array([width, row])

        row_rep = get_edge_max(slice_image, end1, end2)
        if levelset[1, row] > 0:
            speed_rows += row_rep - mean_in_new
        elif levelset[1, row] < 0:
            speed_rows += row_rep - mean_in_new

    speed = 0
    for col in range(width):
        end1 = np.array([col, 0])
        end2 = np.array([col, height])

        col_rep = get_edge_max(slice_image, end1, end2)
        if levelset[0, col] > 0:
            speed += col_rep - mean_in_new
        elif levelset[0, col] < 0:
            speed += col_rep - mean_in_new

    for row in range(height):
        end1 = np.array([col, 0])
        end2 = np.array([col, height])

        col_rep = get_edge_max(slice_image, end1, end2)
        if levelset[0, col] > 0:
            speed += col_rep - mean_in_new
        elif levelset[0, col] < 0:
            speed += col_rep - mean_in_new

        
def region_based_levelset(hand_image):

    slice = hand_image[:, :, 15]

    height = slice.GetHeight()
    width = slice.GetWidth()

    vertex_low = [0, 0]
    vertex_high = [width, height]

    vertex_low_temp = [244, 109]
    vertex_high_temp = [313,148]

    mean_in = -9999
    mean_out = 9999 

    levelset = 0*sitk.Image(slice)
    levelset[1:-1,1:-1] += slice[1:-1,1:-1]+min(slice)

    while(True):
        slice_image = sitk.Image(slice)
        mean_in_new, mean_out_new = get_means(slice_image, vertex_low_temp, vertex_high_temp)
        if(mean_out_new == False):
            vertex_low_temp = [int(0.4*width), int(0.4*height)]
            vertex_high_temp = [int(0.7*width), int(0.7*height)]
        else:
            corner1 = np.array(vertex_high_temp)
            corner2 = np.array([vertex_high_temp[0], vertex_low_temp[1]])
            corner3 = np.array(vertex_low_temp)
            corner4 = np.array([vertex_low_temp[0], vertex_high_temp[1]])

            edge1_max_avg = get_edge_max(slice_image, corner1, corner2)
            edge2_max_avg = get_edge_max(slice_image, corner2, corner3)
            edge3_max_avg = get_edge_max(slice_image, corner3, corner4)
            edge4_max_avg = get_edge_max(slice_image, corner4, corner1)
            print(corner1)
            slice_image[int(corner1[0]), int(corner1[1])] = 999
            slice_image[int(corner2[0]), int(corner2[1])] = 998
            slice_image[int(corner3[0]), int(corner3[1])] = 997
            slice_image[int(corner4[0]), int(corner4[1])] = 996
            sitk.WriteImage(slice_image, os.path.join(temp_dir, 'test.nii'))

            # F = speed_function(slice_image, vertex_low_temp, vertex_high_temp, mean_in_new, mean_out_new)

            levelset = calc_levelset(levelset, slice_image, vertex_low_temp, vertex_high_temp, mean_in_new, mean_out_new)
            
            F_edge1 = round((edge1_max_avg+0.6*(mean_in_new-1.5) + edge1_max_avg-5*(mean_out_new+1))*4)
            F_edge2 = round((edge2_max_avg+5*(mean_in_new-1.5) + edge2_max_avg-5*(mean_out_new+1))*4)
            F_edge3 = round((edge3_max_avg+5*(mean_in_new-1.5) + edge3_max_avg-5*(mean_out_new+1))*4)
            F_edge4 = round((edge4_max_avg+5*(mean_in_new-1.5) + edge4_max_avg-5*(mean_out_new+1))*4)

            print('Mean Inside Bounding box: {}'.format(mean_in_new))
            print('Mean Outside Bounding box: {}'.format(mean_out_new))
            print('Mean Product: (u1+u2=){}, (u1*u2=){}'.format(mean_in_new+mean_out_new, mean_in_new*mean_out_new))
            print('Top 5 Max Average Edge1: {} - Force: {}'.format(edge1_max_avg, F_edge1))
            print('Top 5 Max Average Edge2: {} - Force: {}'.format(edge2_max_avg, F_edge2))
            print('Top 5 Max Average Edge3: {} - Force: {}'.format(edge3_max_avg, F_edge3))
            print('Top 5 Max Average Edge4: {} - Force: {}\n\n'.format(edge4_max_avg, F_edge4))

            vertex_high_temp = [vertex_high_temp[0]+F_edge1, vertex_high_temp[1]+F_edge4]
            vertex_low_temp = [vertex_low_temp[0]-F_edge3, vertex_low_temp[1]-F_edge2]

            if mean_in_new-mean_out_new > 2.2:
                break

    return

if __name__ == '__main__':
    temp_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'tmp')
    if(os.path.isdir(temp_dir)):
        shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)
    main()
