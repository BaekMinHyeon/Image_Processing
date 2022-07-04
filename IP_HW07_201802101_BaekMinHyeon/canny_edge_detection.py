import cv2
import numpy as np

def my_padding(src, pad_shape, pad_type='zero'):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w))
    pad_img[p_h:p_h + h, p_w:p_w + w] = src

    if pad_type == 'repetition':
        print('repetition padding')

        # up
        pad_img[:p_h, p_w:p_w + w] = src[0, :]
        # down
        pad_img[p_h + h:, p_w:p_w + w] = src[h - 1, :]
        # left
        pad_img[:, :p_w] = pad_img[:, p_w:p_w + 1]
        # right
        pad_img[:, p_w + w:] = pad_img[:, p_w + w - 1:p_w + w]

    else:
        print('zero padding')

    return pad_img


def my_filtering(src, mask, pad_type='zero'):
    (h, w) = src.shape
    src_pad = my_padding(src, (mask.shape[0] // 2, mask.shape[1] // 2), pad_type)
    dst = np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            val = np.sum(src_pad[row:row + mask.shape[0], col:col + mask.shape[1]] * mask)
            dst[row, col] = val

    return dst


def get_Gaussian_mask(fsize, sigma=1):
    y, x = np.mgrid[-(fsize // 2):(fsize // 2) + 1, -(fsize // 2):(fsize // 2) + 1]

    # 2차 gaussian mask 생성
    gaus2D = 1 / (2 * np.pi * sigma ** 2) * np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    # mask의 총 합 = 1
    gaus2D /= np.sum(gaus2D)

    return gaus2D


def get_sobel_mask():
    derivative = np.array([[-1, 0, 1]])
    blur = np.array([[1], [2], [1]])

    x = np.dot(blur, derivative)
    y = np.dot(derivative.T, blur.T)

    return x, y


def apply_gaussian_filter(src, fsize=3, sigma=1):
    #####################################################
    # TODO                                              #
    # src에 gaussian filter 적용                         #
    #####################################################
    mask = get_Gaussian_mask(fsize, sigma)
    dst = my_filtering(src, mask, 'repetition')

    return dst


def apply_sobel_filter(src):
    #####################################################
    # TODO                                              #
    # src에 sobel filter 적용                            #
    #####################################################
    mask_x, mask_y = get_sobel_mask()
    Ix = my_filtering(src, mask_x, 'repetition')
    Iy = my_filtering(src, mask_y, 'repetition')

    return Ix, Iy


def calc_magnitude(Ix, Iy):
    #####################################################
    # TODO                                              #
    # Ix, Iy 로부터 magnitude 계산                        #
    #####################################################
    magnitude = np.sqrt(Ix**2+Iy**2)

    return magnitude


def calc_angle(Ix, Iy, eps=1e-6):
    #####################################################
    # TODO                                              #
    # Ix, Iy 로부터 angle 계산                            #
    # numpy의 arctan 사용 O, arctan2 사용 X               #
    # 호도법이나 육십분법이나 상관 X                         #
    # eps     : Divide by zero 방지용                    #
    #####################################################
    angle = np.arctan(Iy/(Ix+eps))

    angle = np.rad2deg(angle)
    return angle


def non_maximum_supression(magnitude, angle):
    ####################################################################################
    # TODO                                                                             #
    # Non-maximum-supression 수행                                                       #
    # 스켈레톤 코드는 angle이 육십분법으로 나타나져 있을 것으로 가정                             #
    ####################################################################################
    (h, w) = magnitude.shape
    # angle의 범위 : -90 ~ 90
    largest_magnitude = np.zeros((h, w))
    for row in range(1, h - 1):
        for col in range(1, w - 1):
            degree = angle[row, col]
            # 각도가 d일 때
            # d 각도의 픽셀과 동시에 180 + d 각도 방향의 픽셀과도 비교 해야함.
            # ex) 10도와 190도 -> 대략 우측과 좌측 픽셀
            # interpolation 방법은 linear로 구현
            if 0 <= degree and degree < 45:
                radian = np.deg2rad(degree)
                right_magnitude = (1-np.tan(radian))*magnitude[row, col+1]+np.tan(radian)*magnitude[row+1, col+1]
                left_magnitude = (1-np.tan(radian))*magnitude[row, col-1]+np.tan(radian)*magnitude[row-1, col-1]
                if magnitude[row, col] > right_magnitude and magnitude[row, col] > left_magnitude:
                    largest_magnitude[row, col] = magnitude[row, col]
            elif 45 <= degree and degree <= 90:
                radian = np.deg2rad(90-degree)
                right_magnitude = (1-np.tan(radian))*magnitude[row+1, col]+np.tan(radian)*magnitude[row+1, col+1]
                left_magnitude = (1-np.tan(radian))*magnitude[row-1, col]+np.tan(radian)*magnitude[row-1, col-1]
                if magnitude[row, col] > right_magnitude and magnitude[row, col] > left_magnitude:
                    largest_magnitude[row, col] = magnitude[row, col]
            elif -45 <= degree and degree < 0:
                radian = np.deg2rad(degree)
                right_magnitude = (1-np.abs(np.tan(radian)))*magnitude[row, col-1]+np.abs(np.tan(radian))*magnitude[row+1, col-1]
                left_magnitude = (1-np.abs(np.tan(radian)))*magnitude[row, col+1]+np.abs(np.tan(radian))*magnitude[row-1, col+1]
                if magnitude[row, col] > right_magnitude and magnitude[row, col] > left_magnitude:
                    largest_magnitude[row, col] = magnitude[row, col]
            elif -90 <= degree and degree < -45:
                radian = np.deg2rad(-90-degree)
                right_magnitude = (1-np.abs(np.tan(radian)))*magnitude[row+1, col]+np.abs(np.tan(radian))*magnitude[row+1, col-1]
                left_magnitude = (1-np.abs(np.tan(radian)))*magnitude[row-1, col]+np.abs(np.tan(radian))*magnitude[row-1, col+1]
                if magnitude[row, col] > right_magnitude and magnitude[row, col] > left_magnitude:
                    largest_magnitude[row, col] = magnitude[row, col]
            else:
                print(row, col, 'error!  degree :', degree)

    return largest_magnitude


def double_thresholding(src):
    dst = src.copy()

    # dst 범위 조정 0 ~ 255
    dst = (dst - np.min(dst)) / (np.max(dst) - np.min(dst))
    dst *= 255
    dst = dst.astype('uint8')

    # threshold는 정해진 값을 사용
    high_threshold_value = 40
    low_threshold_value = 5

    print(high_threshold_value, low_threshold_value)

    #####################################################
    # TODO                                              #
    # Double thresholding 수행                           #
    #####################################################
    (h, w) = dst.shape

    for row in range(h):
        for col in range(w):
            if dst[row, col] > high_threshold_value:
                dst[row, col] = 255
            if dst[row, col] < low_threshold_value:
                dst[row, col] = 0
    visit = []
    queue = []
    pad = np.zeros((h+2, w+2))
    pad[1:1 + h, 1:1 + w] = dst
    for row in range(1, 1+h):
        for col in range(1, 1+w):
            if pad[row, col] != 0 and pad[row, col] != 255:
                visit.append([row, col])
                queue.append([row, col])
                while len(queue) != 0:
                    (r, c) = queue.pop(0)
                    if pad[r - 1, c - 1] == 255 or pad[r, c - 1] == 255 or pad[r + 1, c - 1] == 255 or pad[
                        r - 1, c] == 255 or pad[r + 1, c] == 255 or pad[r - 1, c + 1] == 255 or pad[r, c + 1] == 255 or \
                            pad[r + 1, c + 1] == 255:
                        for i in range(len(visit)):
                            (r1, c1) = visit.pop(0)
                            pad[r1, c1] = 255
                        for i in range(len(queue)):
                            queue.pop(0)
                        break
                    if pad[r - 1, c - 1] == 0 and pad[r, c - 1] == 0 and pad[r + 1, c - 1] == 0 and pad[
                        r - 1, c] == 0 and pad[r + 1, c] == 0 and pad[r - 1, c + 1] == 0 and pad[r, c + 1] == 0 and pad[
                        r + 1, c + 1] == 0:
                        for i in range(len(visit)):
                            (r1, c1) = visit.pop(0)
                            pad[r1, c1] = 0
                        for i in range(len(queue)):
                            queue.pop(0)
                        break
                    if pad[r - 1, c - 1] != 0 and pad[r - 1, c - 1] != 255 and [r - 1, c - 1] not in visit:
                        visit.append([r - 1, c - 1])
                        queue.append([r - 1, c - 1])
                    if pad[r, c - 1] != 0 and pad[r, c - 1] != 255 and [r, c - 1] not in visit:
                        visit.append([r, c - 1])
                        queue.append([r, c - 1])
                    if pad[r + 1, c - 1] != 0 and pad[r + 1, c - 1] != 255 and [r + 1, c - 1] not in visit:
                        visit.append([r + 1, c - 1])
                        queue.append([r + 1, c - 1])
                    if pad[r - 1, c] != 0 and pad[r - 1, c] != 255 and [r - 1, c] not in visit:
                        visit.append([r - 1, c])
                        queue.append([r - 1, c])
                    if pad[r + 1, c] != 0 and pad[r + 1, c] != 255 and [r + 1, c] not in visit:
                        visit.append([r + 1, c])
                        queue.append([r + 1, c])
                    if pad[r - 1, c + 1] != 0 and pad[r - 1, c + 1] != 255 and [r - 1, c + 1] not in visit:
                        visit.append([r - 1, c + 1])
                        queue.append([r - 1, c + 1])
                    if pad[r, c + 1] != 0 and pad[r, c + 1] != 255 and [r, c + 1] not in visit:
                        visit.append([r, c + 1])
                        queue.append([r, c + 1])
                    if pad[r + 1, c + 1] != 0 and pad[r + 1, c + 1] != 255 and [r + 1, c + 1] not in visit:
                        visit.append([r + 1, c + 1])
                        queue.append([r + 1, c + 1])
                for i in range(len(visit)):
                    (r1, c1) = visit.pop(0)
                    pad[r1, c1] = 0

    dst = pad[1:1 + h, 1:1 + w]
    # return dst
    dst = dst.astype('float32') / 255.0
    return dst

def canny_edge_detection(src):
    # Apply low pass filter
    I = apply_gaussian_filter(src, fsize=3, sigma=1)

    # Apply high pass filter
    Ix, Iy = apply_sobel_filter(I)

    # Get magnitude and angle
    magnitude = calc_magnitude(Ix, Iy)
    angle = calc_angle(Ix, Iy)

    # Apply non-maximum-supression
    after_nms = non_maximum_supression(magnitude, angle)

    # Apply double thresholding
    dst = double_thresholding(after_nms)

    return dst, after_nms, magnitude


if __name__ == '__main__':
    img = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255.0

    canny, after_nms, magnitude = canny_edge_detection(img)

    # 시각화 하기 위해 0~1로 normalize (min-max scaling)
    magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))
    after_nms = (after_nms - np.min(after_nms)) / (np.max(after_nms) - np.min(after_nms))

    cv2.imshow('original', img)
    cv2.imshow('magnitude', magnitude)
    cv2.imshow('after_nms', after_nms)
    cv2.imshow('canny_edge', canny)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

