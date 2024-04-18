import SimpleITK as sitk
import argparse
import os

def get_bounding_boxes(knuckle, temp_dir):
    '''
    
    '''
    print("Normalize and smooth using median filter...")
    knuckle_norm = sitk.Normalize(knuckle)
    knuckle_med = sitk.Median(knuckle_norm)
    sitk.WriteImage(knuckle_med, os.path.join(temp_dir, 'knuckle.nii.gz'))

    print("Running canny edge filter...")
    canny_edge = sitk.CannyEdgeDetection(knuckle_med, lowerThreshold=0.88, upperThreshold=0.999, variance = 3*[0.001*knuckle_med.GetSpacing()[0]])
    sitk.WriteImage(canny_edge, os.path.join(temp_dir, 'canny.nii.gz'))

    print("Gaussian Smoothing...")
    knuckle_smooth = sitk.SmoothingRecursiveGaussian(knuckle_med, 0.2)
    sitk.WriteImage(knuckle_smooth, os.path.join(temp_dir, 'smoothed.nii.gz'))

    print("Binary Thresholding...")
    thresh = sitk.BinaryThreshold(knuckle_smooth, 1, 99)
    sitk.WriteImage(thresh, os.path.join(temp_dir, 'thresh.nii.gz'))

    print("Combining masks...")
    knuckle_mask = sitk.Cast(canny_edge, sitk.sitkUInt8) | sitk.Cast(thresh, sitk.sitkUInt8)
    
    print("Running erode and dilate filter...")
    erode_filter = sitk.BinaryErodeImageFilter()
    erode_filter.SetKernelRadius(2)
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelRadius(6)

    knuckle_mask = erode_filter.Execute(knuckle_mask)
    knuckle_mask = dilate_filter.Execute(knuckle_mask)
    
    print("Running connected component and sorting by size...")
    connected_comp = sitk.ConnectedComponent(knuckle_mask, knuckle_mask)
    connected_comp = sitk.RelabelComponent(connected_comp, sortByObjectSize=True)
    connected_comp = sitk.Threshold(connected_comp, 0, 10, 0)
    sitk.WriteImage(connected_comp, os.path.join(temp_dir, 'conn.nii.gz'))

    print("Running label statistics filter...")
    label_stat_filter = sitk.LabelStatisticsImageFilter()
    label_stat_filter.Execute(connected_comp, connected_comp)
    num_labels = label_stat_filter.GetNumberOfLabels()

    bounding_boxes = []

    print("Finding bounding boxes...")
    for label in range(1, num_labels):
        bounding_box = label_stat_filter.GetBoundingBox(label)

        overlap_flag = False
        for idx in range(len(bounding_boxes)):
            bb = bounding_boxes[idx]
            bb_x_range = bb[0:2]
            current_x_range = bounding_box[0:2]

            if overlap(bb_x_range, current_x_range) > 0.1:
                overlap_flag = True
                bb = (min(bb[0], bounding_box[0]), max(bb[1], bounding_box[1]), min(bb[2], bounding_box[2]), max(bb[3], bounding_box[3]), min(bb[4], bounding_box[4]), max(bb[5], bounding_box[5]))
                bounding_boxes[idx] = bb

        if not overlap_flag:
            bounding_boxes.append(bounding_box)

        print("Label: ", label)
        print("Bounding Box: ", bounding_box)
        print()

    return bounding_boxes

def crop_using_bounding_box(knuckle, bounding_boxes, stack, dir):
    '''
    
    '''

    idx = 1
    sorted_bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0], reverse=True)
    for bb in sorted_bounding_boxes:
        if idx > 5:
            break
        print(bb)
        cropped_img = knuckle[bb[0]-10:bb[1]+10, bb[2]-10:bb[3]+10, bb[4]:bb[5]]

        sitk.WriteImage(cropped_img, os.path.join(os.path.join(dir, 'mcp{}'.format(idx)), '{}_mcp_{}.nii.gz'.format(stack, idx)))
        idx += 1

def overlap(a, b):
    '''
    
    '''
    min_a = min(a)
    max_a = max(a)
    min_b = min(b)
    max_b = max(b)

    overlap = max(0, min(max_a, max_b) - max(min_a, min_b))
    range_a = max_a - min_a
    range_b = max_b - min_b
    tot_range = range_a + range_b

    return 2*overlap/tot_range 

if __name__ == '__main__':
    '''
    
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dst_img_path", type=str, help="Image (path + filename)")
    parser.add_argument("input_mid_img_path", type=str, help="Image (path + filename)")
    parser.add_argument("input_prx_img_path", type=str, help="Image (path + filename)")
    parser.add_argument("out_dir", type=str, help="Output Directory")
    args = parser.parse_args()

    input_dst_img_path = args.input_dst_img_path
    input_mid_img_path = args.input_mid_img_path
    input_prx_img_path = args.input_prx_img_path
    out_dir = args.out_dir

    input_filename = input_dst_img_path.split('/')[-1]
    ext_idx = input_filename.find('.')
    input_filename=input_filename[0:ext_idx]

    temp_dst_dir = os.path.join(out_dir, 'temp_' + input_filename)
    if not os.path.exists(temp_dst_dir):
        os.mkdir(temp_dst_dir)

    input_filename = input_mid_img_path.split('/')[-1]
    ext_idx = input_filename.find('.')
    input_filename=input_filename[0:ext_idx]

    temp_mid_dir = os.path.join(out_dir, 'temp_' + input_filename)
    if not os.path.exists(temp_mid_dir):
        os.mkdir(temp_mid_dir)

    input_filename = input_prx_img_path.split('/')[-1]
    ext_idx = input_filename.find('.')
    input_filename=input_filename[0:ext_idx]

    temp_prx_dir = os.path.join(out_dir, 'temp_' + input_filename)
    if not os.path.exists(temp_prx_dir):
        os.mkdir(temp_prx_dir)

    mcp1_dir = os.path.join(out_dir, 'mcp1')
    if not os.path.exists(mcp1_dir):
        os.mkdir(mcp1_dir)

    mcp2_dir = os.path.join(out_dir, 'mcp2')
    if not os.path.exists(mcp2_dir):
        os.mkdir(mcp2_dir)

    mcp3_dir = os.path.join(out_dir, 'mcp3')
    if not os.path.exists(mcp3_dir):
        os.mkdir(mcp3_dir)

    mcp4_dir = os.path.join(out_dir, 'mcp4')
    if not os.path.exists(mcp4_dir):
        os.mkdir(mcp4_dir)

    mcp5_dir = os.path.join(out_dir, 'mcp5')
    if not os.path.exists(mcp5_dir):
        os.mkdir(mcp5_dir)

    print("Reading stacks...")
    knuckle_dst = sitk.ReadImage(input_dst_img_path)
    knuckle_mid = sitk.ReadImage(input_mid_img_path)
    knuckle_prx = sitk.ReadImage(input_prx_img_path)

    print("Get bounding boxes...")
    bounding_boxes_dst = get_bounding_boxes(knuckle_dst, temp_dst_dir)
    bounding_boxes_mid = get_bounding_boxes(knuckle_mid, temp_mid_dir)
    bounding_boxes_prx = get_bounding_boxes(knuckle_prx, temp_prx_dir)

    print("Crop using bounding boxes...")
    crop_using_bounding_box(knuckle_dst, bounding_boxes_dst, 'dst', out_dir)
    crop_using_bounding_box(knuckle_mid, bounding_boxes_mid, 'mid', out_dir)
    crop_using_bounding_box(knuckle_prx, bounding_boxes_prx, 'prx', out_dir)

