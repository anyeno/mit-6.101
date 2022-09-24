#!/usr/bin/env python3

# NO ADDITIONAL IMPORTS!
# (except in the last part of the lab; see the lab writeup for details)
import math
from PIL import Image


# last lab


# 获取图片的某个像素点
def get_pixel(image, x, y):
    width = image['width']  # 一行有多少个元素
    # height = image['height']    # 一列有多少个元素
    # x行第y个元素
    return image['pixels'][x * width + y]


# 设置图片的某个像素点(x,y)为c
def set_pixel(image, x, y, c):
    width = image['width']
    image['pixels'][x * width + y] = c


# 用func方法改变图片的每个像素
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
    for x in range(image['height']):  # x是行
        for y in range(image['width']):  # y是列
            color = get_pixel(image, x, y)
            newcolor = func(color)
            # print("x == " + str(x))
            # print("y == " + str(y))
            set_pixel(result, x, y, newcolor)  # 你妈的在这里把xy写反了，调了我半天
    return result


def inverted(image):
    return apply_per_pixel(image, lambda c: 255 - c)


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
    a = (n - 1) // 2
    extended_image_pixels = [[0] * (width + n - 1) for i in range(height + n - 1)]
    two_image_pixels = [([0] * width) for i in range(height)]
    # print(two_image_pixels)
    # print(extended_image_pixels)
    for i in range(len(image['pixels'])):
        two_image_pixels[i // width][i % width] = image['pixels'][i]
    # print(two_image_pixels)

    ## 先扩展边界情况
    # 中间
    for i in range(a, a + height):
        for j in range(a, a + width):
            extended_image_pixels[i][j] = two_image_pixels[i - a][j - a]
    # print(extended_image_pixels)
    # 左上
    for i in range(a):
        for j in range(a):
            if boundary_behavior == 'zero':
                extended_image_pixels[i][j] = 0
            if boundary_behavior == 'extend':
                extended_image_pixels[i][j] = extended_image_pixels[a][a]
            if boundary_behavior == 'wrap':
                extended_image_pixels[i][j] = extended_image_pixels[a + height - 1][a + width - 1]
    # 上
    for i in range(a):
        for j in range(a, a + width):
            if boundary_behavior == 'zero':
                extended_image_pixels[i][j] = 0
            if boundary_behavior == 'extend':
                extended_image_pixels[i][j] = extended_image_pixels[a][j]
            if boundary_behavior == 'wrap':
                extended_image_pixels[i][j] = extended_image_pixels[height - 1 + a][j]
    # 右上
    for i in range(a):
        for j in range(a + width, 2 * a + width):
            if boundary_behavior == 'zero':
                extended_image_pixels[i][j] = 0
            if boundary_behavior == 'extend':
                extended_image_pixels[i][j] = extended_image_pixels[a][a + width - 1]
            if boundary_behavior == 'wrap':
                extended_image_pixels[i][j] = extended_image_pixels[height - 1 + a][a]
    # 左
    for i in range(a, a + height):
        for j in range(a):
            if boundary_behavior == 'zero':
                extended_image_pixels[i][j] = 0
            if boundary_behavior == 'extend':
                extended_image_pixels[i][j] = extended_image_pixels[i][a]
            if boundary_behavior == 'wrap':
                extended_image_pixels[i][j] = extended_image_pixels[i][a + width - 1]
    # 左下
    for i in range(a + height, 2 * a + height):
        for j in range(a):
            if boundary_behavior == 'zero':
                extended_image_pixels[i][j] = 0
            if boundary_behavior == 'extend':
                extended_image_pixels[i][j] = extended_image_pixels[a + height - 1][a]
            if boundary_behavior == 'wrap':
                extended_image_pixels[i][j] = extended_image_pixels[a][a + width - 1]
    # 下
    for i in range(a + height, a + a + height):
        for j in range(a, a + width):
            if boundary_behavior == 'zero':
                extended_image_pixels[i][j] = 0
            if boundary_behavior == 'extend':
                extended_image_pixels[i][j] = extended_image_pixels[a + height - 1][j]
            if boundary_behavior == 'wrap':
                extended_image_pixels[i][j] = extended_image_pixels[a][j]
    # 右下
    for i in range(a + height, a + a + height):
        for j in range(a + width, 2 * a + width):
            if boundary_behavior == 'zero':
                extended_image_pixels[i][j] = 0
            if boundary_behavior == 'extend':
                extended_image_pixels[i][j] = extended_image_pixels[a + height - 1][a + width - 1]
            if boundary_behavior == 'wrap':
                extended_image_pixels[i][j] = extended_image_pixels[a][a]
    # 右
    for i in range(a, a + height):
        for j in range(a + width, 2 * a + width):
            if boundary_behavior == 'zero':
                extended_image_pixels[i][j] = 0
            if boundary_behavior == 'extend':
                extended_image_pixels[i][j] = extended_image_pixels[i][a + width - 1]
            if boundary_behavior == 'wrap':
                extended_image_pixels[i][j] = extended_image_pixels[i][a]

    # print("two pixels ==")
    # print(two_image_pixels)
    # print("Extend == ")
    # print(extended_image_pixels)
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
    for i in range(a, a + height):
        for j in range(a, a + width):  # 枚举每个要计算的像素点  即未拓展的点
            res = 0  # 当前是(i,j)点
            # print("i == "+str(i)+"  j == "+str(j))
            for k in range(n):
                for p in range(n):
                    # print("kernel  "+str(k)+"  "+str(p)+"    extend  "+str(i+k-a)+"  "+str(j+p-a))
                    # print("kernel: "+str(kernel[k][p])+"   extend  "+str(extended_image_pixels[i+k-a][j+p-a]))
                    res = res + kernel[k][p] * extended_image_pixels[i + k - a][j + p - a]
            res_pixels.append(res)
    result = {
        'width': width,
        'height': height,
        'pixels': res_pixels
    }
    # print("res_pixels == ")
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
    pixels = image['pixels'].copy()
    for i in range(length):
        if pixels[i] >= 255:
            pixels[i] = 255
        elif pixels[i] <= 0:
            pixels[i] = 0
        else:
            pixels[i] = round(pixels[i])
    result = {
        'width': image['width'],
        'height': image['height'],
        'pixels': pixels
    }
    # print("res_round == ")
    # print(result)
    return result


# FILTERS

def create_kernel(n):
    kernel = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(n):
            kernel[i][j] = 1 / n ** 2
            print(kernel[i][j])
    # print(kernel)
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
        pixels[i] = 2 * pixels[i] - blurred_img['pixels'][i]
    res = {
        'width': image['width'],
        'height': image['height'],
        'pixels': pixels
    }
    return round_and_clip_image(res)


def edges(image):
    kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    ky = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    Ox = correlate(image, kx, 'extend')
    Oy = correlate(image, ky, 'extend')
    res_pixels = []
    w = Ox['width']
    for i in range(image['height']):
        for j in range(image['width']):
            res_pixels.append(math.sqrt(Ox['pixels'][i * w + j] ** 2 + Oy['pixels'][i * w + j] ** 2))
    res = {
        'width': image['width'],
        'height': image['height'],
        'pixels': res_pixels
    }
    return round_and_clip_image(res)


# VARIOUS FILTERS


# 彩色图像拆分成三个灰度图像
def split(image):
    width = image['width']
    height = image['height']
    pixels = image['pixels']
    length = len(pixels)
    three_pixels = [[0] * length for i in range(3)]
    for i in range(length):
        for j in range(3):
            three_pixels[j][i] = pixels[i][j]
    res = []
    for i in range(3):
        res.append({
            'width': width,
            'height': height,
            'pixels': three_pixels[i]
        })
    return res


# 3个灰度图像合并成一个彩色图像  images是数组
def combine(images):
    width = images[0]['width']
    height = images[0]['height']
    res_pixels = []
    for i in range(len(images[0]['pixels'])):
        res_pixels.append((images[0]['pixels'][i], images[1]['pixels'][i], images[2]['pixels'][i]))
    res = {
        'height': height,
        'width': width,
        'pixels': res_pixels
    }
    # print(res)
    return res


def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """

    def color_filter(image):
        three_images = split(image)
        for i in range(3):
            three_images[i] = filt(three_images[i])
        res = combine(three_images)
        return res

    return color_filter


def make_blur_filter(n):
    def blur_filter(image, m=n):
        return blurred(image, m)

    return blur_filter


def make_sharpen_filter(n):
    def sharp_filter(image, m=n):
        return sharpened(image, m)

    return sharp_filter


def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """

    def final_filter(image):
        res = image.copy()
        for i in range(len(filters)):
            res = filters[i](res)
        return res

    return final_filter


# SEAM CARVING

# Main Seam Carving Implementation

def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image. Returns a new image.
    """
    res = image.copy()

    for i in range(ncols):
        grey = greyscale_image_from_color_image(res)
        energy = compute_energy(grey)
        cem = cumulative_energy_map(energy)
        seam = minimum_energy_seam(cem)
        res = image_without_seam(res, seam)

    return res


# Optional Helper Functions for Seam Carving


def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    pixels = image['pixels']
    res_pixels = []
    for i in range(len(pixels)):
        res_pixels.append(round(pixels[i][0] * 0.299 + pixels[i][1] * 0.587 + pixels[i][2] * 0.114))
    res = {
        'width': image['width'],
        'height': image['height'],
        'pixels': res_pixels
    }
    return res


def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    res = edges(grey)

    return res


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy
    function), computes a "cumulative energy map" as described in the lab 2
    writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    width = energy['width']
    height = energy['height']
    pixels = energy['pixels']
    res_pixels = [[0] for i in range(width * height)]
    for i in range(height):
        for j in range(width):
            if i == 0:
                res_pixels[i * width + j] = pixels[i * width + j]
            else:
                if j == 0:
                    res_pixels[i * width + j] = pixels[i * width + j] + min(res_pixels[(i - 1) * width + j],
                                                                            res_pixels[(i - 1) * width + j + 1])
                elif j == width - 1:
                    res_pixels[i * width + j] = pixels[i * width + j] + min(res_pixels[(i - 1) * width + j - 1],
                                                                            res_pixels[(i - 1) * width + j])
                else:
                    res_pixels[i * width + j] = pixels[i * width + j] + min(res_pixels[(i - 1) * width + j - 1],
                                                                            res_pixels[(i - 1) * width + j],
                                                                            res_pixels[(i - 1) * width + j + 1])
    res = {
        'width': width,
        'height': height,
        'pixels': res_pixels
    }
    return res


def minimum_energy_seam(cem):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 2 writeup).
    """
    pixels = cem['pixels']
    width = cem['width']
    height = cem['height']
    min_col = 0
    min_val = pixels[(height - 1) * width + min_col]
    for j in range(1, width):
        if pixels[(height - 1) * width + j] < min_val:
            min_val = pixels[(height - 1) * width + j]
            min_col = j
    res = []
    res.insert(0, min_col)
    # print("height == "+str(height))
    for i in range(height - 2, -1, -1):
        # print("i == "+str(i))
        left = min_col - 1
        right = min_col + 2
        if min_col == 0:
            left = 0
        elif min_col == width - 1:
            right = width
            # 一开始写成 right = width - 1   浪费我一下午时间  草！！！
        in_min_col = left
        in_min_val = pixels[i * width + in_min_col]
        for j in range(left + 1, right):
            if pixels[i * width + j] < in_min_val:
                in_min_val = pixels[i * width + j]
                in_min_col = j
        res.insert(0, in_min_col)
        min_col = in_min_col
    # print(res)
    return res


def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    pixels = image['pixels']
    res_pixels = []
    width = image['width']
    # print("seam == ")
    # print(seam)
    x = set()
    for i in range(len(seam)):
        x.add(i * width + seam[i])
    for i in range(len(pixels)):
        if i not in x:
            res_pixels.append(pixels[i])
    res = {
        'width': image['width'] - 1,
        'height': image['height'],
        'pixels': res_pixels
    }
    # print("pixels == ")
    # print(pixels)
    # print("res_pixels == ")
    # print(res_pixels)
    return res


# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES


def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img = img.convert("RGB")  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        w, h = img.size
        return {"height": h, "width": w, "pixels": pixels}


def save_color_image(image, filename, mode="PNG"):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode="RGB", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns an instance of this class
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [
                round(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) for p in img_data
            ]
        elif img.mode == "LA":
            pixels = [p[0] for p in img_data]
        elif img.mode == "L":
            pixels = list(img_data)
        else:
            raise ValueError("Unsupported image mode: %r" % img.mode)
        w, h = img.size
        return {"height": h, "width": w, "pixels": pixels}


def save_greyscale_image(image, filename, mode="PNG"):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode="L", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == "__main__":
    img = load_color_image("test_images/pattern.png")
    grey = greyscale_image_from_color_image(img)
    print(grey)
    energy = compute_energy(grey)
    print(energy)
    cem = cumulative_energy_map(energy)
    print(cem)
    seam = minimum_energy_seam(cem)
    print(seam)
