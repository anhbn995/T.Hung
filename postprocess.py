import numpy as np
import rasterio
def merge_result(list_img, output_tif):
    with rasterio.open(list_img[0]) as src:
        h,w = src.height, src.width
        meta = src.meta
    result_data = np.zeros((h,w)).astype('uint8')
    for idx,img in enumerate(list_img):
        data_img = rasterio.open(img).read(1)
        result_data[data_img==1]= idx + 1
    with rasterio.open(output_tif, "w", **meta) as dest:
        dest.write(result_data, indexes=1)
    return True

if __name__ == '__main__':
    list_image = [
        '/home/boom/data/green_cover/green_demo/results/2020.tif',
        '/home/boom/data/green_cover/green_demo/results/2019.tif'
    ]
    out_tif = '/home/boom/data/green_cover/green_demo/results/merge.tif'
    merge_result(list_image, out_tif)