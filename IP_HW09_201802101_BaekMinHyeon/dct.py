import numpy as np
import cv2


def C(w, n):
    if w == 0:
        return (1/n) ** 0.5
    else:
        return (2/n) ** 0.5


def dct_block(block, n=8):
    dst = np.zeros(block.shape)

    v, u = dst.shape
    y, x = np.mgrid[0:u, 0:v]

    # dct_mask를 시각화 하기 위한 리스트
    # 본 과제에서만 사용되며, 다음 주 jpeg 과제에서는 사용되지 않을 예정
    # 각 단계의 mask를 list에 추가해서 리턴
    dct_mask_list = list()

    for v_ in range(v):
        for u_ in range(u):
            mask = np.cos((2*x+1)*u_*np.pi/(2*n))*np.cos((2*y+1)*v_*np.pi/(2*n))

            dct_mask_list.append(mask)

            Cv = np.sqrt(2/n)
            Cu = np.sqrt(2/n)
            if(v_ == 0):
                Cv = np.sqrt(1/n)
            if(u_ == 0):
                Cu = np.sqrt(1/n)
            dst[v_, u_] = Cu*Cv*np.sum(block*mask)

    return dst, dct_mask_list

if __name__ == '__main__':
    block_size = 4

    np.random.seed(2022)
    src = np.random.randn(4, 4)
    src = (src - src.min()) / (src.max() - src.min())
    print(src)

    dst, dct_mask_list = dct_block(src, n=block_size)
    print(np.round(dst, 4))

    # dct_mask 전체 한번에 시각화
    tmp_list = list()
    for i in range(block_size):
        row = np.concatenate(dct_mask_list[i*block_size:(i+1)*block_size], axis=1)
        tmp_list.append(row)
    dct_mask_show = np.concatenate(tmp_list, axis=0)

    dct_mask_show = (dct_mask_show - dct_mask_show.min()) / (dct_mask_show.max() - dct_mask_show.min())
    dct_mask_show = (dct_mask_show * 255).astype('uint8')

    print(dct_mask_show)

    cv2.imshow('entire_dct_mask', cv2.resize(dct_mask_show, dsize=(512, 512), interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

