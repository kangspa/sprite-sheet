import cv2
import numpy as np
import math
from get_padding import get_padding

def resize_img(img, frame, bgc, method, padding):
    height, width = img.shape[:2]
    padding_top, padding_right, padding_bottom, padding_left = get_padding(padding)
    if method == "top":
        pad_top, pad_bottom = 0, frame[1]-height
        pad_right = pad_left = (frame[0]-width)//2
        if (frame[0]-width)%2 != 0: pad_right += 1
    elif method == "bottom":
        pad_top, pad_bottom = frame[1]-height, 0
        pad_right = pad_left = (frame[0]-width)//2
        if (frame[0]-width)%2 != 0: pad_right += 1
    elif method == "right":
        pad_top = pad_bottom = (frame[1]-height)//2
        if (frame[1]-height)%2 != 0: pad_top += 1
        pad_right, pad_left = 0, frame[0]-width
    elif method == "left":
        pad_top = pad_bottom = (frame[1]-height)//0
        if (frame[1]-height)%2 != 0: pad_top += 1
        pad_right, pad_left = frame[0]-width, 0
    elif method == "center":
        pad_top = pad_bottom = (frame[1]-height)//0
        if (frame[1]-height)%2 != 0: pad_top += 1
        pad_right = pad_left = (frame[0]-width)//2
        if (frame[0]-width)%2 != 0: pad_right += 1
    pad_top += padding_top
    pad_right += padding_right
    pad_bottom += padding_bottom
    pad_left += padding_left
    return cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=bgc)

def combine_images_cv2(image_paths, output_path, num=None, stair=None, frame=None, bgc=(255, 255, 255, 0), method="bottom", padding=[0]):
    # 이미지 로드
    images = []
    height, width = -float("inf"), -float("inf")
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # 만약 이미지 로드가 제대로 이루어지지 않을 경우, 오류 발생
        if img is None:
            raise ValueError("일부 이미지가 로드되지 않았습니다. 경로를 확인하세요.")
        # 이미지를 배열로 저장하고, 최대 높이와 너비를 저장한다.
        images.append(img)
        h, w, _ = img.shape
        height, width = max(height, h), max(width, w)
    # frame이 지정됐는데, 이미지보다 작은 사이즈일 경우 에러 발생
    if (frame is not None) and (frame[0] < width | frame[1] < height):
        raise ValueError("지정한 이미지 크기 값이 이미지보다 작습니다. frame을 다시 지정해주세요.")
    # frame이 지정이 안됐다면, 로드한 이미지의 최대 크기 기준으로 frame을 지정
    elif frame is None: frame = [width, height]
    # 이미지 개수
    total_images = len(images)

    # num과 stair가 모두 None이면 오류 발생
    if num is None and stair is None:
        raise ValueError("num 또는 stair 중 하나는 반드시 지정해야 합니다.")

    # stair가 없고, num이 지정된 경우, num 개수만큼 한 줄에 배치
    if (num is not None) and (stair is None):
        rows = math.ceil(total_images / num)
        stair = [num] * rows  # 동일한 개수로 줄 배치
    
    # stair 배열 내에 0보다 작은 값이 있을 경우 오류 발생
    if any(x < 0 for x in stair):
        raise ValueError("배치하려는 이미지 수는 0보다 작을 수 없습니다.")

    img_index = 0  # 현재 체크할 인덱스
    row_images = []  # 세로로 합치기 위한 임시 배열
    max_width = 0
    for count in stair:
        if img_index >= total_images:
            break
        
        row = images[img_index:img_index + count]
                
        # 각 이미지 높이를 동일하게 맞춤 (여백 추가)
        resized_row = []
        for img in row:
            resized_row.append(resize_img(img, frame, bgc, method, padding))
        
        # 가로 방향으로 병합 (hconcat)
        row_combined = cv2.hconcat(resized_row)
        
        row_images.append(row_combined)
        max_width = max(max_width, row_combined.shape[1])
        img_index += count
    # 세로 방향으로 병합하기 전, width를 통일(stair로 진행 시 줄마다 이미지 다를 수 있음)
    for i, img in enumerate(row_images):
        pad = max_width-img.shape[1]
        row_images[i] = cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=bgc)
    # 세로 방향으로 병합 (vconcat)
    final_image = cv2.vconcat(row_images)

    # 결과 저장
    cv2.imwrite(output_path, final_image)
    print(f"이미지가 {output_path}에 저장되었습니다.")

if __name__ == "__main__":
    image_paths = []
    for i in range(13):
        image_paths.append(f"output/mario4blog_{i}.png")
    output_path = "output/merged_output.png"
    combine_images_cv2(image_paths, output_path, num=13, frame=(17,16), padding=[0])