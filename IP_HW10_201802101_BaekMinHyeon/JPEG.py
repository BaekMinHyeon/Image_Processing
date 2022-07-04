import numpy as np
import cv2
import time

def Quantization_Luminance(scale_factor):
    luminance = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]])
    return luminance * scale_factor

def img2block(src, n=8):
    ######################################
    # TODO                               #
    # img2block 완성                      #
    # img를 block으로 변환하기              #
    # blocks : 각 block을 추가한 list       #
    # return value : np.array(blocks)     #
    ######################################
    h, w = src.shape
    blocks = list()
    for row in range(h//n):
        for col in range(w//n):
            blocks.append(src[row*n:row*n+n, col*n:col*n+n])
    return np.array(blocks)


def DCT(block, n=8):
    ######################################
    # TODO                               #
    # DCT 완성                            #
    ######################################
    dst = np.zeros(block.shape)

    v, u = dst.shape
    y, x = np.mgrid[0:u, 0:v]

    for v_ in range(v):
        for u_ in range(u):
            mask = np.cos((2 * x + 1) * u_ * np.pi / (2 * n)) * np.cos((2 * y + 1) * v_ * np.pi / (2 * n))
            Cv = np.sqrt(2 / n)
            Cu = np.sqrt(2 / n)
            if (v_ == 0):
                Cv = np.sqrt(1 / n)
            if (u_ == 0):
                Cu = np.sqrt(1 / n)
            dst[v_, u_] = Cu * Cv * np.sum(block * mask)
    return np.round(dst)

def my_zigzag_encoding(block,block_size=8):
    ######################################
    # TODO                               #
    # my_zigzag_encoding 완성             #
    ######################################
    zigzag = list()
    zero = list()
    n = block_size
    x = 0
    y = 0
    direction = -1
    for i in range(2*n):
        while y>=0 and x>=0 and y<=n-1 and x<=n-1:
            if block[y, x] == 0:
                zero.append(0)
            else:
                zigzag.extend(zero)
                zero = list()
                zigzag.append(block[y, x])
            y += direction
            x -= direction
        if x == n:
            y += 1
        elif y == n:
            x += 1
        elif y == -1:
            x += 1
        elif x == -1:
            y += 1
        y -= direction
        x += direction
        direction = -direction
    if len(zero) != 0:
        zigzag.append('EOB')
    return zigzag

def my_zigzag_decoding(block, block_size=8):
    ######################################
    # TODO                               #
    # my_zigzag_decoding 완성             #
    ######################################
    n = block_size
    re_block = np.zeros((n, n))
    x = 0
    y = 0
    direction = -1
    z = 0
    count = 0
    while True:
        while y>=0 and x>=0 and y<=n-1 and x<=n-1:
            if block[z] != 'EOB':
                re_block[y, x] = block[z]
                z += 1
            else:
                re_block[y, x] = 0
            y += direction
            x -= direction
            count += 1
        if count == n*n:
            break
        if x == n:
            y += 1
        elif y == n:
            x += 1
        elif y == -1:
            x += 1
        elif x == -1:
            y += 1
        y -= direction
        x += direction
        direction = -direction
    return re_block

def DCT_inv(block, n = 8):
    ###################################################
    # TODO                                            #
    # DCT_inv 완성                                     #
    # DCT_inv 는 DCT와 다름.                            #
    ###################################################
    dst = np.zeros(block.shape)

    v, u = dst.shape
    u_, v_ = np.mgrid[0:u, 0:v]
    C = np.zeros((n, n))
    for i in range(v):
        for j in range(u):
            if i == 0 and j == 0:
                C[i, j] = np.sqrt(1/n)*np.sqrt(1/n)
            elif i == 0 or j == 0:
                C[i, j] = np.sqrt(1/n)*np.sqrt(2/n)
            else:
                C[i, j] = np.sqrt(2/n)*np.sqrt(2/n)
    for x in range(v):
        for y in range(u):
            mask = np.cos((2 * x + 1) * u_ * np.pi / (2 * n)) * np.cos((2 * y + 1) * v_ * np.pi / (2 * n))
            dst[x, y] = np.sum(C* block * mask)
    return np.round(dst)

def block2img(blocks, src_shape, n = 8):
    ###################################################
    # TODO                                            #
    # block2img 완성                                   #
    # 복구한 block들을 image로 만들기                     #
    ###################################################
    h, w = src_shape
    dst = np.zeros((h, w))
    i = 0
    rblocks = (blocks-np.min(blocks))/(np.max(blocks)-np.min(blocks))
    for row in range(0, h, n):
        for col in range(0, w, n):
            dst[row:row+n, col:col+n] = rblocks[i]
            i += 1
    return dst

def Encoding(src, n=8,scale_factor=1):
    #################################################################################################
    # TODO                                                                                          #
    # Encoding 완성                                                                                  #
    # Encoding 함수를 참고용으로 첨부하긴 했는데 수정해서 사용하실 분은 수정하셔도 전혀 상관 없습니다.              #
    #################################################################################################
    print('<start Encoding>')
    # img -> blocks
    blocks = img2block(src, n=n)
    print("block = \n",src[150:158,89:97])

    #subtract 128
    blocks = np.double(blocks)-128
    b = np.double(src[150:158,89:97])-128
    print("b = \n",b)

    #DCT
    blocks_dct = []
    for block in blocks:
        blocks_dct.append(DCT(block, n=n))
    blocks_dct = np.array(blocks_dct)

    # print DCT results
    bd = DCT(b,n=8)
    print("bd = \n",bd)


    #Quantization + thresholding
    Q = Quantization_Luminance(scale_factor)
    QnT = np.round(blocks_dct / Q)
    #print Quantization results
    bq = np.round(bd  / Q)
    print("bq = \n",bq)

    # zigzag scanning
    zz = []
    for i in range(len(QnT)):
        zz.append(my_zigzag_encoding(QnT[i],block_size=n))

    return zz, src.shape, bq

def Decoding(zigzag, src_shape,bq, n=8,scale_factor=1):
    #################################################################################################
    # TODO                                                                                          #
    # Decoding 완성                                                                                  #
    # Decoding 함수를 참고용으로 첨부하긴 했는데 수정해서 사용하실 분은 수정하셔도 전혀 상관 없습니다.              #
    #################################################################################################
    print('<start Decoding>')

    # zigzag scanning
    blocks = []
    for i in range(len(zigzag)):
        blocks.append(my_zigzag_decoding(zigzag[i], block_size=n))
    blocks = np.array(blocks)


    # Denormalizing
    Q = Quantization_Luminance(scale_factor=scale_factor)
    blocks = blocks * Q
    # print results Block * Q
    bq2 = bq * Q
    print("bq2 = \n",bq2)

    # inverse DCT
    blocks_idct = []
    for block in blocks:
        blocks_idct.append(DCT_inv(block, n=n))
    blocks_idct = np.array(blocks_idct)

    #print IDCT results
    bd2 = DCT_inv(bq2,n=8)
    print("bd2 = \n",bd2)

    # add 128
    blocks_idct += 128

    # print block value
    b2 = bd2 + 128
    print("b2 = \n",b2)

    # block -> img
    dst = block2img(blocks_idct, src_shape=src_shape, n=n)

    return dst, b2



def main():
    scale_factor = 1
    start = time.time()
    # src = cv2.imread('../imgs/Lenna.png', cv2.IMREAD_GRAYSCALE)
    src = cv2.imread('caribou.tif', cv2.IMREAD_GRAYSCALE)
    comp, src_shape,bq = Encoding(src, n=8,scale_factor=scale_factor)
    np.save('comp.npy', comp)
    np.save('src_shape.npy', src_shape)
    # print(comp)
    comp = np.load('comp.npy', allow_pickle=True)
    src_shape = np.load('src_shape.npy')
    recover_img, b2 = Decoding(comp, src_shape, bq,n=8,scale_factor=scale_factor)
    print("scale_factor : ",scale_factor,"differences between original and reconstructed = \n",src[150:158,89:97]-b2)
    # print(recover_img)
    total_time = time.time() - start
    #
    print('time : ', total_time)
    if total_time > 12:
        print('감점 예정입니다.')
    print(recover_img.shape)
    cv2.imshow('original img', src)
    cv2.imshow('recover img', recover_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
