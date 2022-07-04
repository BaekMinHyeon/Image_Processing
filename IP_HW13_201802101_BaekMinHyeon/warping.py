import cv2
import numpy as np


def transform(transform_mat, img):
    src_h, src_w = img.shape
    y_scale = transform_mat[1, 1]
    x_scale = transform_mat[0, 0]

    dst_h = max(int(src_h*y_scale+0.5), src_h)
    dst_w = max(int(src_w*x_scale+0.5), src_w)
    dst = np.zeros((dst_h, dst_w), img.dtype)

    for y in range(src_h):
        for x in range(src_w):
            dst_coordinate = transform_mat @ np.array([x, y, 1])
            x_ = int(np.clip(dst_coordinate[0], 0, dst_h-1))
            y_ = int(np.clip(dst_coordinate[1], 0, dst_w-1))

            dst[y_, x_] = img[y, x]

    return dst


'''
def transform(transform_mat, img):
    ###############################################################
    # TODO                                                        #
    # Forward transform 구현                                       #
    # dst의 크기는 scale이 1 이상일 때에만 때에만 변화하도록 함           #
    #  -> 구현 편의를 위함                                           #
    # 각 픽셀마다 dst에 대응되는 좌표를 계산해 transform 수행            #
    ###############################################################
    src_h, src_w = img.shape
    y_scale = transform_mat[1, 1]
    x_scale = transform_mat[0, 0]

    dst_h = ???
    dst_w = ???
    dst = np.zeros((dst_h, dst_w), img.dtype)

    for y in range(src_h):
        for x in range(src_w):
            dst[???, ???] = img[y, x]


    return dst
'''



if __name__ == '__main__':
    #img = np.zeros((256, 256)).astype('uint8')
    #img[50:100, 50:100] = 255

    img = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

    transform_mat = np.array([[1, 0, -50],
                              [0, 1, -100],
                              [0, 0, 1]])


    dst = transform(transform_mat, img)

    cv2.imshow('original', img)
    cv2.imshow('transformed', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

