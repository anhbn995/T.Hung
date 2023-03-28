#input : images, label, box
#output: trainnig dataset
import os
from utils import _generate_maskfile, crop_raster_by_shape, gen_trainning_dataset
import glob

if __name__=="__main__":
    tmp_dir = r'D:\HIEUUUUUU\land_cover\data_biuldup\DATA_train\tmp'

    # step1: gen mask file
    images_dir = r"D:\HIEUUUUUU\land_cover\data_train\build_up\image"
    labels_dir = r"D:\HIEUUUUUU\land_cover\data_train\build_up\labels_buildup"
    mask_dir = r"D:\HIEUUUUUU\land_cover\data_train\build_up\masks"
    box_dir = r"D:\HIEUUUUUU\land_cover\data_train\build_up\boxs"

    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    for _file in glob.glob(images_dir + '\\*.tif'):
        name_file = _file.split('\\')[-1].split('.')[0]
        labelshapefile = f'{labels_dir}\\{name_file}.shp'
        
        maskfile = f'{mask_dir}\\{name_file}.tif'

        _generate_maskfile(_file, labelshapefile, maskfile)

        #step2: crop mask and image with box
        box_path = f'{box_dir}\\{name_file}.shp'
        try:
            result_dir_images = r'D:\HIEUUUUUU\land_cover\data_train\build_up\image_crop'
            crop_raster_by_shape(_file, box_path, result_dir_images)

            result_dir_masks = r'D:\HIEUUUUUU\land_cover\data_train\build_up\mask_crop'
            crop_raster_by_shape(maskfile, box_path, result_dir_masks)
        except:
            # print('sai')
            print(_file)
            

    #step3: generate trainning dataset
    #note : chỉ chạy gen trainning dataset sau khi crop hết tất cả các mẫu
    output_dir = r'D:\HIEUUUUUU\land_cover\data_train\build_up\training_dataset'
    gen_trainning_dataset(result_dir_images, result_dir_masks, output_dir)