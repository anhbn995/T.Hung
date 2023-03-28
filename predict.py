import numpy as np
import rasterio
from rasterio.windows import Window
import threading
from tqdm import tqdm
import concurrent.futures
import warnings, cv2, os
import tensorflow as tf
# import Vectorization
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects
from utils import get_range_value, create_list_id
from tensorflow.compat.v1.keras.backend import set_session
from model import unet_basic

warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))

def predict(model, path_image, path_predict, size=128):
    print(path_image)
    # qt_scheme = get_quantile_schema(path_image)
    with rasterio.open(path_image) as raster:
        meta = raster.meta
        
        meta.update({'count': 1, 'nodata': 0,"dtype":"uint8"})
        height, width = raster.height, raster.width
        input_size = size
        stride_size = input_size - input_size //4
        padding = int((input_size - stride_size) / 2)
        list_coordinates = []
        for start_y in range(0, height, stride_size):
            for start_x in range(0, width, stride_size): 
                x_off = start_x if start_x==0 else start_x - padding
                y_off = start_y if start_y==0 else start_y - padding
                    
                end_x = min(start_x + stride_size + padding, width)
                end_y = min(start_y + stride_size + padding, height)
                
                x_count = end_x - x_off
                y_count = end_y - y_off
                list_coordinates.append(tuple([x_off, y_off, x_count, y_count, start_x, start_y]))
        with rasterio.open(path_predict,'w+', **meta, compress='lzw') as r:
            read_lock = threading.Lock()
            write_lock = threading.Lock()

            def process(coordinates):
                x_off, y_off, x_count, y_count, start_x, start_y = coordinates
                read_wd = Window(x_off, y_off, x_count, y_count)
                with read_lock:
                    values = raster.read(window=read_wd)
                if raster.profile["dtype"]=="uint8":
                    # print('zo'*10, 'uint8')
                    image_detect = values[0:4].transpose(1,2,0).astype(int)
                else:
                    # datas = []
                    # for chain_i in range(4):
                    #     # band_qt = qt_scheme[chain_i]
                    #     band = values[chain_i]

                    #     # cut_nor = np.interp(band, (band_qt.get('p2'), band_qt.get('p98')), (1, 255)).astype(int)
                    #     cut_nor = get_range_value(band)
                    #     datas.append(cut_nor)
                    datas = get_range_value(values)
                    image_detect = np.transpose(datas, (1,2,0))

                img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
                mask = np.pad(np.ones((stride_size, stride_size), dtype=np.uint8), ((padding, padding),(padding, padding)))
                shape = (stride_size, stride_size)
                if y_count < input_size or x_count < input_size:
                    img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
                    mask = np.zeros((input_size, input_size), dtype=np.uint8)
                    if start_x == 0 and start_y == 0:
                        img_temp[(input_size - y_count):input_size, (input_size - x_count):input_size] = image_detect
                        mask[(input_size - y_count):input_size-padding, (input_size - x_count):input_size-padding] = 1
                        shape = (y_count-padding, x_count-padding)
                    elif start_x == 0:
                        img_temp[0:y_count, (input_size - x_count):input_size] = image_detect
                        if y_count == input_size:
                            mask[padding:y_count-padding, (input_size - x_count):input_size-padding] = 1
                            shape = (y_count-2*padding, x_count-padding)
                        else:
                            mask[padding:y_count, (input_size - x_count):input_size-padding] = 1
                            shape = (y_count-padding, x_count-padding)
                    elif start_y == 0:
                        img_temp[(input_size - y_count):input_size, 0:x_count] = image_detect
                        if x_count == input_size:
                            mask[(input_size - y_count):input_size-padding, padding:x_count-padding] = 1
                            shape = (y_count-padding, x_count-2*padding)
                        else:
                            mask[(input_size - y_count):input_size-padding, padding:x_count] = 1
                            shape = (y_count-padding, x_count-padding)
                    else:
                        img_temp[0:y_count, 0:x_count] = image_detect
                        mask[padding:y_count, padding:x_count] = 1
                        shape = (y_count-padding, x_count-padding)
                        
                    image_detect = img_temp
                mask = (mask!=0)

                if np.count_nonzero(image_detect) > 0:
                    if len(np.unique(image_detect)) <= 2:
                        pass
                    else:
                        y_pred = model.predict(image_detect[np.newaxis,...])
                        y_pred = (y_pred[0,...,0] > 0.5).astype(np.uint8) 
                        
                        # y_pred = 1 - y_pred
                        y = y_pred[mask].reshape(shape)
                        
                        with write_lock:
                            r.write(y[np.newaxis,...], window=Window(start_x, start_y, shape[1], shape[0]))
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                results = list(tqdm(executor.map(process, list_coordinates), total=len(list_coordinates)))

def Morphology(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # dilation  
    # img = cv2.dilate(data,kernel,iterations = 1)
    # opening
    #     img = cv2.morphologyEx(data, cv2.MORPH_OPEN, kernel)
    # for i in range(10):
    #     img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel2)
    # closing
    #     for _ in range(2):
    img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)
    return img
    
if __name__=="__main__":
    # model_path = r"/home/skm/SKM16/Tmp/HIEEU/weights_landcover/model_dat_trong_v3.h5"
    # folder_image_path = r'/home/skm/SKM16/Tmp/HIEEU/germany_test'
    # folder_output_path = r'/home/skm/SKM16/Tmp/HIEEU/germany_test/model_dat_trong_v3'
    # # image_path = '/home/boom/data/green_cover/green_demo/images/2020.tif'
    
    # # output_tif = '/home/boom/data/green_cover/green_demo/results/test.tif'
    # size_model = 256
    # model = unet_basic((size_model, size_model, 4))
    # model.load_weights(model_path)
    # # predict(model, image_path, output_tif)

    # model_path = r"/home/skm/SKM16/X/Lo/Samples_V4_2023_03_12/gen_cut128stride100_TruotLo/model/gen_cut128stride100_TruotLo.h5"
    # folder_image_path = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection'
    # folder_output_path = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection/gen_cut128stride100_TruotLo'
    
    model_path = r'/home/skm/SKM16/Tmp/HIEEU/weights_landcover/model_building256_2.h5'
    folder_image_path = r'/home/skm/SKM16/Tmp/HIEEU/test2'
    folder_output_path = r'/home/skm/SKM16/Tmp/HIEEU/test2/test2_kp'
    
    
    # image_path = '/home/boom/data/green_cover/green_demo/images/2020.tif'
    
    # output_tif = '/home/boom/data/green_cover/green_demo/results/test.tif'
    size_model = 128
    model = unet_basic((size_model, size_model, 4))
    model.load_weights(model_path)


    list_image=create_list_id(folder_image_path)
    
    if not os.path.exists(folder_output_path):
        os.makedirs(folder_output_path)
    for image_path in tqdm(list_image):
        image_name = os.path.basename(image_path)     
        fp_img = os.path.join(folder_image_path, image_name)+".tif"
        # if fp_img=="/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Full_Images_LandslideDetection/S2B_L2A_20190221_Full.tif":
            # continue
        outputpredict = os.path.join(folder_output_path,image_name) +".tif"
        # print("1233456789", outputpredict)
        
        if not os.path.exists(outputpredict):
            print(fp_img)
            predict(model, fp_img, outputpredict, size_model)
        else:
            pass