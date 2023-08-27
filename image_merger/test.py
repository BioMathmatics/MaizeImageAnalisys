from image_merger import ImageMerger
from imageio.v2 import imread, imwrite
import time

image1_list = [
    ['examples/image1/leftup.png', 'examples/image1/rightup.png'],
    ['examples/image1/leftdown.png', 'examples/image1/rightdown.png']
]
path_to_save_1 = 'examples/image1/res_image1.png'

# image2_list = [
#     ['examples/image2/image2_11.jpg', 'examples/image2/image2_21.jpg'],
#     ['examples/image2/image2_12.jpg', 'examples/image2/image2_22.jpg'],
#     ['examples/image2/image2_13.jpg', 'examples/image2/image2_23.jpg'],
#     ['examples/image2/image2_14.jpg', 'examples/image2/image2_24.jpg']
# ]

image2_list = [
    ['examples/image2/image2_11.jpg', 'examples/image2/image2_12.jpg',
     'examples/image2/image2_13.jpg', 'examples/image2/image2_14.jpg'],
    ['examples/image2/image2_21.jpg', 'examples/image2/image2_22.jpg',
     'examples/image2/image2_23.jpg', 'examples/image2/image2_24.jpg']
]
path_to_save_2 = 'examples/image2/res_image2.png'

image3_list = [
    ['examples/image3/image3_11.jpg', 'examples/image3/image3_21.jpg',
     'examples/image3/image3_31.jpg', 'examples/image3/image3_41.jpg'],
    ['examples/image3/image3_12.jpg', 'examples/image3/image3_22.jpg',
     'examples/image3/image3_32.jpg', 'examples/image3/image3_42.jpg'],
    ['examples/image3/image3_13.jpg', 'examples/image3/image3_23.jpg',
     'examples/image3/image3_33.jpg', 'examples/image3/image3_43.jpg'],
    ['examples/image3/image3_14.jpg', 'examples/image3/image3_24.jpg',
     'examples/image3/image3_34.jpg', 'examples/image3/image3_44.jpg'],
    ['examples/image3/image3_15.jpg', 'examples/image3/image3_25.jpg',
     'examples/image3/image3_35.jpg', 'examples/image3/image3_45.jpg']
]
path_to_save_3 = 'examples/image3/res_image3.png'

start = time.time()

merger = ImageMerger()

for parts, path_to_save in zip([image1_list, image2_list, image3_list], [path_to_save_1, path_to_save_2, path_to_save_3]):
    parts_im = []
    for line in parts:
        im_list = [imread(i) for i in line]
        parts_im.append(
            merger.merge_from_list(im_list)
        )
    res_im = merger.merge_from_list(parts_im)
    imwrite(path_to_save, res_im)

stop = time.time()

print(f'time of work {round(stop - start, 2)} sec.')
