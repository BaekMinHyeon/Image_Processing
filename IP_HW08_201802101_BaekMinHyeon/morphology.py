import cv2
import numpy as np

def dilation(B, S):
    ###############################################
    # TODO                                        #
    # dilation 함수 완성                           #
    ###############################################
    (h, w) = B.shape
    (h1, w1) = S.shape
    h2 = h1//2
    w2 = w1//2
    pad_img = np.zeros((h+2*h2, w+2*w2))
    pad_img[h2:h2+h, w2:w2+w] = B
    pad_img2 = pad_img.copy()
    for row in range(pad_img.shape[0]):
        for col in range(pad_img.shape[1]):
            if pad_img2[row, col] == 1:
                pad_img[row - h2:row + h2 + 1, col - w2:col + w2 + 1] = S[0:h2 * 2 + 1, 0:w2 * 2 + 1]
    dst = pad_img[h2:h2+h, w2:w2+w]
    return dst

def erosion(B, S):
    ###############################################
    # TODO                                        #
    # erosion 함수 완성                            #
    ###############################################
    (h, w) = B.shape
    (h1, w1) = S.shape
    h2 = h1 // 2
    w2 = w1 // 2
    pad_img = np.zeros((h + 2 * h2, w + 2 * w2))
    pad_img[h2:h2 + h, w2:w2 + w] = B
    pad_img2 = pad_img.copy()
    for row in range(pad_img.shape[0]):
        for col in range(pad_img.shape[1]):
            if pad_img2[row, col] == 1:
                if np.array_equal(pad_img2[row - h2:row + h2 + 1, col - w2:col + w2 + 1], S[0:h2 * 2 + 1, 0:w2 * 2 + 1]):
                    pad_img[row, col] = 1
                else:
                    pad_img[row, col] = 0
    dst = pad_img[h2:h2 + h, w2:w2 + w]
    return dst

def opening(B, S):
    ###############################################
    # TODO                                        #
    # opening 함수 완성                            #
    ###############################################
    dst = erosion(B, S)
    dst = dilation(dst, S)
    return dst

def closing(B, S):
    ###############################################
    # TODO                                        #
    # closing 함수 완성                            #
    ###############################################
    dst = dilation(B, S)
    dst = erosion(dst, S)
    return dst


if __name__ == '__main__':
    B = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]])

    S = np.array(
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]])


    cv2.imwrite('test_binary_image.png', (B * 255).astype(np.uint8))

    img_dilation = dilation(B, S)
    img_dilation = (img_dilation*255).astype(np.uint8)
    print(img_dilation)
    cv2.imwrite('test_dilation.png', img_dilation)

    img_erosion = erosion(B, S)
    img_erosion = (img_erosion * 255).astype(np.uint8)
    print(img_erosion)
    cv2.imwrite('test_erosion.png', img_erosion)

    img_opening = opening(B, S)
    img_opening = (img_opening * 255).astype(np.uint8)
    print(img_opening)
    cv2.imwrite('test_opening.png', img_opening)

    img_closing = closing(B, S)
    img_closing = (img_closing * 255).astype(np.uint8)
    print(img_closing)
    cv2.imwrite('test_closing.png', img_closing)

    test_img = cv2.imread("morphology_img.png", cv2.IMREAD_GRAYSCALE)
    test_img = test_img / 255.
    test_img = np.round(test_img)
    test_img = test_img.astype(np.uint8)

    img_dilation = dilation(test_img, S)
    img_dilation = (img_dilation*255).astype(np.uint8)
    print(img_dilation)
    cv2.imwrite('morphology_dilation.png', img_dilation)

    img_erosion = erosion(test_img, S)
    img_erosion = (img_erosion * 255).astype(np.uint8)
    print(img_erosion)
    cv2.imwrite('morphology_erosion.png', img_erosion)

    img_opening = opening(test_img, S)
    img_opening = (img_opening * 255).astype(np.uint8)
    print(img_opening)
    cv2.imwrite('morphology_opening.png', img_opening)

    img_closing = closing(test_img, S)
    img_closing = (img_closing * 255).astype(np.uint8)
    print(img_closing)
    cv2.imwrite('morphology_closing.png', img_closing)




