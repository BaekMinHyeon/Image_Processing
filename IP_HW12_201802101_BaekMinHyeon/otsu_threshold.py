import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_threshold(img, th=120):
    ######################################################
    # TODO                                               #
    # 실습시간에 배포된 코드 사용                             #
    ######################################################
    dst = np.zeros(img.shape, img.dtype)
    dst[img >= th] = 255
    dst[img < th] = 0
    return dst


def my_otsu_threshold(img):
    hist, bins = np.histogram(img.ravel(),256,[0,256])
    p = hist / np.sum(hist) + 1e-7

    ######################################################
    # TODO                                               #
    # Otsu 방법을 통해 threshold 구한 후 이진화 수행          #
    # cv2의 threshold 와 같은 값이 나와야 함                 #
    ######################################################
    B = list()
    q1 = list()
    q2 = list()
    m1 = list()
    m2 = list()
    l = len(p)
    j = np.mgrid[0:len(p)]
    m = np.sum(j*p)
    q1.append(p[0])
    q2.append(1-q1[0])
    m1.append(0)
    m2.append((m-q1[0]*m1[0])/q2[0])
    for i in range(l-1):
        B.append(q1[i]*q2[i]*(m1[i]-m2[i])**2)
        q1.append(q1[i]+p[i+1])
        q2.append(1-q1[i+1])
        m1.append((q1[i]*m1[i]+(i+1)*p[i+1])/q1[i+1])
        m2.append((q2[i]*m2[i]-(i+1)*p[i+1])/q2[i+1])
    B.append(q1[255]*q2[255]*(m1[255]-m2[255])**2)
    th = np.argmax(B)
    dst = apply_threshold(img, th)

    return th, dst

if __name__ == '__main__':
    img = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)

    th_cv2, dst_cv2 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    th_my, dst_my = my_otsu_threshold(img)

    print('Threshold from cv2: {}'.format(th_cv2))
    print('Threshold from my: {}'.format(th_my))

    cv2.imshow('original image', img)
    cv2.imshow('cv2 threshold', dst_cv2)
    cv2.imshow('my threshold', dst_my)

    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


