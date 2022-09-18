#!/usr/bin/env python3

import math

from PIL import Image as Image

# NO ADDITIONAL IMPORTS ALLOWED!


# 获取图片的某个像素点
def get_pixel(image, x, y):
    width = image['width']   # 一行有多少个元素
    #height = image['height']    # 一列有多少个元素
    # x行第y个元素
    return image['pixels'][x * width + y]


#设置图片的某个像素点(x,y)为c
def set_pixel(image, x, y, c):
    width = image['width']
    image['pixels'][x * width + y] = c


#用func方法改变图片的每个像素
def apply_per_pixel(image, func):
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': [],
    }
    for i in range(image['height'] * image['width']):
        result['pixels'].append(0)
    # print("length of result == " + str(len(result['pixels'])))
    # print("image height ==" + str(image['height']))
    # print("image width == " + str(image['width']))
    for x in range(image['height']):    #x是行
        for y in range(image['width']):     #y是列
            color = get_pixel(image, x, y)
            newcolor = func(color)
            #print("x == " + str(x))
            #print("y == " + str(y))
            set_pixel(result, x, y, newcolor)    #你妈的在这里把xy写反了，调了我半天
    return result


def inverted(image):
    return apply_per_pixel(image, lambda c: 255-c)


# HELPER FUNCTIONS

# kernel 二维数组   n*n  n为奇数
def correlate(image, kernel, boundary_behavior):
    """
    Compute the result of correlating the given image with the given kernel.
    `boundary_behavior` will one of the strings 'zero', 'extend', or 'wrap',
    and this function will treat out-of-bounds pixels as having the value zero,
    the value of the nearest edge, or the value wrapped around the other edge
    of the image, respectively.

    if boundary_behavior is not one of 'zero', 'extend', or 'wrap', return
    None.

    Otherwise, the output of this function should have the same form as a 6.101
    image (a dictionary with 'height', 'width', and 'pixels' keys), but its
    pixel values do not necessarily need to be in the range [0,255], nor do
    they need to be integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    DESCRIBE YOUR KERNEL REPRESENTATION HERE
    """
    if boundary_behavior != 'zero' and boundary_behavior != 'extend' and boundary_behavior != 'wrap':
        return None
    height = image['height']
    width = image['width']
    n = len(kernel)
    a = (n-1) // 2
    extended_image_pixels = [[0]*(width + n - 1) for i in range(height + n-1)]
    two_image_pixels = [([0] * width) for i in range(height)]
    # print(two_image_pixels)
    # print(extended_image_pixels)
    for i in range(len(image['pixels'])):
        two_image_pixels[i // width ][i % width ] = image['pixels'][i]
    # print(two_image_pixels)

    ## 先扩展边界情况
    # 中间
    for i in range(a, a+height):
        for j in range(a, a+width):
            extended_image_pixels[i][j] = two_image_pixels[i-a][j-a]
    #print(extended_image_pixels)
    # 左上
    for i in range(a):
        for j in range(a):
            if boundary_behavior == 'zero':
                extended_image_pixels[i][j] = 0
            if boundary_behavior == 'extend':
                extended_image_pixels[i][j] = extended_image_pixels[a][a]
            if boundary_behavior == 'wrap':
                extended_image_pixels[i][j] = extended_image_pixels[a+height-1][a+width-1]
    # 上
    for i in range(a):
        for j in range(a, a+width):
            if boundary_behavior == 'zero':
                extended_image_pixels[i][j] = 0
            if boundary_behavior == 'extend':
                extended_image_pixels[i][j] = extended_image_pixels[a][j]
            if boundary_behavior == 'wrap':
                extended_image_pixels[i][j] = extended_image_pixels[height - 1 + a][j]
    # 右上
    for i in range(a):
        for j in range(a+width, 2*a+width):
            if boundary_behavior == 'zero':
                extended_image_pixels[i][j] = 0
            if boundary_behavior == 'extend':
                extended_image_pixels[i][j] = extended_image_pixels[a][a+width-1]
            if boundary_behavior == 'wrap':
                extended_image_pixels[i][j] = extended_image_pixels[height - 1 + a][a]
    # 左
    for i in range(a, a+height):
        for j in range(a):
            if boundary_behavior == 'zero':
                extended_image_pixels[i][j] = 0
            if boundary_behavior == 'extend':
                extended_image_pixels[i][j] = extended_image_pixels[i][a]
            if boundary_behavior == 'wrap':
                extended_image_pixels[i][j] = extended_image_pixels[i][a+width-1]
    # 左下
    for i in range(a+height, 2*a+height):
        for j in range(a):
            if boundary_behavior == 'zero':
                extended_image_pixels[i][j] = 0
            if boundary_behavior == 'extend':
                extended_image_pixels[i][j] = extended_image_pixels[a+height-1][a]
            if boundary_behavior == 'wrap':
                extended_image_pixels[i][j] = extended_image_pixels[a][a+width-1]
    # 下
    for i in range(a+height, a+a+height):
        for j in range(a, a+width):
            if boundary_behavior == 'zero':
                extended_image_pixels[i][j] = 0
            if boundary_behavior == 'extend':
                extended_image_pixels[i][j] = extended_image_pixels[a+height-1][j]
            if boundary_behavior == 'wrap':
                extended_image_pixels[i][j] = extended_image_pixels[a][j]
    # 右下
    for i in range(a+height, a+a+height):
        for j in range(a+width, 2*a+width):
            if boundary_behavior == 'zero':
                extended_image_pixels[i][j] = 0
            if boundary_behavior == 'extend':
                extended_image_pixels[i][j] = extended_image_pixels[a+height-1][a+width-1]
            if boundary_behavior == 'wrap':
                extended_image_pixels[i][j] = extended_image_pixels[a][a]
    # 右
    for i in range(a, a+height):
        for j in range(a+width, 2*a+width):
            if boundary_behavior == 'zero':
                extended_image_pixels[i][j] = 0
            if boundary_behavior == 'extend':
                extended_image_pixels[i][j] = extended_image_pixels[i][a+width-1]
            if boundary_behavior == 'wrap':
                extended_image_pixels[i][j] = extended_image_pixels[i][a]

    # print("two pixels ==")
    # print(two_image_pixels)
    # print("Extend == ")
    #print(extended_image_pixels)
    # for i in range(len(extended_image_pixels)):
    #     print(extended_image_pixels[i])

    ## 计算 并将结果加到result_pixels中
    res_pixels = []
    # for i in range(a, a+height):
    #     for j in range(a, a+width):
    #         res = 0
    #         for k in range(i-a, i+a+1):
    #             for p in range(j-a, j-a+1):   # extend某一行的每个数
    #                 for col_in_kernel in range(n):  # 与kernel某一列的每个数相乘
    #                     res = res + extended_image_pixels[i][j] * kernel[k-(i-a)][col_in_kernel]
    #         res_pixels.append(res)
    for i in range(a, a+height):
        for j in range(a, a+width): # 枚举每个要计算的像素点  即未拓展的点
            res = 0         # 当前是(i,j)点
            # print("i == "+str(i)+"  j == "+str(j))
            for k in range(n):
                for p in range(n):
                    #print("kernel  "+str(k)+"  "+str(p)+"    extend  "+str(i+k-a)+"  "+str(j+p-a))
                    #print("kernel: "+str(kernel[k][p])+"   extend  "+str(extended_image_pixels[i+k-a][j+p-a]))
                    res = res + kernel[k][p] * extended_image_pixels[i+k-a][j+p-a]
            res_pixels.append(res)
    result = {
        'width': width,
        'height': height,
        'pixels': res_pixels
    }
    #print("res_pixels == ")
    # print(res_pixels)
    return result

def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the 'pixels' list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    length = image['width'] * image['height']
    pixels = image['pixels']
    for i in range(length):
        if pixels[i] > 255:
            pixels[i] = 255
        if pixels[i] < 0:
            pixels[i] = 0;
        pixels[i] = round(pixels[i])
    result = {
        'width': image['width'],
        'height': image['height'],
        'pixels': pixels
    }
    return result

# FILTERS

def create_kernel(n):
    kernel = [[0]*n for i in range(n)]
    for i in range(n):
        for j in range(n):
            kernel[i][j] = 1/n**2
            print(kernel[i][j])
    #print(kernel)
    return kernel

def blurred(image, n):
    """
    Return a new image representing the result of applying a box blur (with
    kernel size n) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)
    kernel = create_kernel(n)

    # then compute the correlation of the input image with that kernel
    res = correlate(image, kernel, 'extend')

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.
    return round_and_clip_image(res)

def sharpened(image, n):
    blurred_img = blurred(image, n)
    pixels = image['pixels'].copy()
    for i in range(len(pixels)):
        pixels[i] = 2*pixels[i] - blurred_img['pixels'][i]
    res = {
        'width': image['width'],
        'height': image['height'],
        'pixels': pixels
    }
    return round_and_clip_image(res)


def edges(image):
    kx = [[-1,0,1],[-2,0,2],[-1,0,1]]
    ky = [[-1, -2, -1],[0,0,0], [1,2,1]]
    Ox = correlate(image, kx, 'extend')
    Oy = correlate(image, ky, 'extend')
    res_pixels = []
    for i in range(image['height']):
        for j in range(image['width']):
            res_pixels.append(math.sqrt(Ox['pixels'][i*Ox['width']+j]**2+Oy['pixels'][i*Ox['width']+j]**2))
    res = {
        'width': image['width'],
        'height': image['height'],
        'pixels': res_pixels
    }
    return round_and_clip_image(res)

# HELPER FUNCTIONS FOR LOADING AND SAVING IMAGES

def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith('RGB'):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == 'LA':
            pixels = [p[0] for p in img_data]
        elif img.mode == 'L':
            pixels = list(img_data)
        else:
            raise ValueError('Unsupported image mode: %r' % img.mode)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_greyscale_image(image, filename, mode='PNG'):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode='L', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == '__main__':
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.
    #im = load_greyscale_image("test_images/cat.png")
    #image = blurred(im, 2)
    # i = {
    #     'height': 3,
    #     'width': 2,
    #     'pixels': [0, 50, 50, 100, 100, 255],
    # }
    # #kernel = [[0,0,0,0,0],[0,0,0,0,0],[1,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
    # kernel = [[0,0.2,0],[0.2,0.2,0.2],[0,0.2,0]]
    # img = correlate(i, kernel, 'zero')
    # image = round_and_clip_image(img)
    # print(image['pixels'])
    # save_greyscale_image(image, "new.png")
    create_kernel(3)


