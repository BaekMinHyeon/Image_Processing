import numpy as np
import cv2
import time

def my_get_Gaussian2D_mask(msize, sigma=1):
    #########################################
    # ToDo
    # 2D gaussian filter 만들기
    #########################################
    y, x = np.mgrid[-(msize//2):msize//2+1, -(msize//2):msize//2+1]
    '''
    y, x = np.mgrid[-1:2, -1:2]
    y = [[-1,-1,-1],
         [ 0, 0, 0],
         [ 1, 1, 1]]
    x = [[-1, 0, 1],
         [-1, 0, 1],
         [-1, 0, 1]]
    '''

    #2차 gaussian mask 생성
    gaus2D = np.e**(-(x**2+y**2)/(2*sigma**2))/(2*np.pi*sigma**2)
    #mask의 총 합 = 1
    gaus2D = gaus2D/np.sum(gaus2D)
    return gaus2D

def my_get_Gaussian1D_mask(msize, sigma=1):
    #########################################
    # ToDo
    # 1D gaussian filter 만들기
    #########################################
    x = np.full((1, msize), np.mgrid[-(msize//2):msize//2+1])
    '''
    x = np.full((1, 3), [-1, 0, 1])
    x = [[ -1, 0, 1]]
    
    x = np.array([[-1, 0, 1]])
    x = [[ -1, 0, 1]]
    '''
    gaus1D = np.e**(-x**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

    #mask의 총 합 = 1
    gaus1D = gaus1D/np.sum(gaus1D)
    return gaus1D

if __name__ == '__main__':
    #src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    src = cv2.imread('gaussian_test.jpg', cv2.IMREAD_GRAYSCALE)
    mask_size = 13
    gaus2D = my_get_Gaussian2D_mask(mask_size, sigma = 10)
    gaus1D = my_get_Gaussian1D_mask(mask_size, sigma = 10)
    #C:/Users/hyunseop/Desktop/TA/Homework/imgs/Lena.png
    print('mask size : ', mask_size)
    print('1D gaussian filter')
    start = time.perf_counter()  # 시간 측정 시작
    # dst_gaus1D= my_filtering(src, gaus1D.T)
    # dst_gaus1D= my_filtering(dst_gaus1D, gaus1D)
    dst_gaus1D = cv2.filter2D(src, -1,gaus1D.T)
    dst_gaus1D = cv2.filter2D(dst_gaus1D,-1, gaus1D)
    end = time.perf_counter()  # 시간 측정 끝
    print('1D time : ', end-start)

    print('2D gaussian filter')
    start = time.perf_counter()  # 시간 측정 시작
    # dst_gaus2D= my_filtering(src, gaus2D, pad_type='repetition')
    dst_gaus2D= cv2.filter2D(src, -1,gaus2D)
    end = time.perf_counter()  # 시간 측정 끝
    print('2D time : ', end-start)

    dst_gaus1D = np.clip(dst_gaus1D+0.5, 0, 255)
    dst_gaus1D = dst_gaus1D.astype(np.uint8)
    dst_gaus2D = np.clip(dst_gaus2D+0.5, 0, 255)
    dst_gaus2D = dst_gaus2D.astype(np.uint8)

    cv2.imshow('original', cv2.resize(src,(600,400)))
    cv2.imshow('1D gaussian img', cv2.resize(dst_gaus1D,(600,400)))
    cv2.imshow('2D gaussian img', cv2.resize(dst_gaus2D,(600,400)))
    cv2.waitKey()
    cv2.destroyAllWindows()