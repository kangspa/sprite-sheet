import cv2
import numpy as np

# min_area에 따른 동일 물체 검출 함수
def merge_close_objects(mask, min_area):
    # 거리 변환 적용 (255에서 0까지의 거리 계산)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # 일정 거리(min_area) 내의 픽셀을 같은 오브젝트로 취급
    merged_mask = (dist_transform <= min_area).astype(np.uint8) * 255

    return merged_mask

# 원래 물체에 딱 맞춰서 테두리 검출하기 위한 함수
def adjust_bbox_to_original(merged_bbox, object_mask):
    # 현재 체크하는 합쳐진 물체 위치 (merged_mask에서의 위치)
    origin_x, origin_y, origin_w, origin_h = merged_bbox
    
    # 원본 이미지(object_mask)에서의 검출된 위치들을 확인 후, 테두리 위치 검출
    object_region = object_mask[origin_y:origin_y+origin_h, origin_x:origin_x+origin_w]
    contours, _ = cv2.findContours(object_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 최종적으로 물체의 bbox를 정확히 그린 후, return 해준다.
    x1, y1, x2, y2 = float("inf"), float("inf"), -float("inf"), -float("inf")
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x1, y1, x2, y2 = min(x1, x), min(y1, y), max(x2, x+w), max(y2, y+h)
    
    return (origin_x+x1, origin_y+y1, origin_x+x2, origin_y+y2)

# 배경색(이미지 내에서 가장 많은 컬러값) 튜플로 추출
def detect_background_color(image, img_channel):
    pixels = image.reshape(-1, img_channel)
    unique, counts = np.unique(pixels, axis=0, return_counts=True)
    background_color = unique[np.argmax(counts)]  # 가장 빈도가 높은 색상 선택
    return background_color

def detect_objects(image_path, color_type, background_color=None, min_area=0):
    # 이미지 로드
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("이미지를 불러올 수 없습니다.")
    img_height, img_width, img_channel = image.shape
    print(img_height, img_width, img_channel)
    # 배경색 마스크 생성
    if color_type == "alpha":
        if img_channel != 4:
            raise ValueError("투명도 값이 정의되지 않은 이미지입니다.")
        alpha_value = 0  # 알파 채널 값만 사용
        mask = cv2.inRange(image[:, :, 3], alpha_value, alpha_value)  # 알파 채널 값만 비교
    elif color_type == "gray":
        # 배경색 입력 없을 경우, 자동 감지 (가장 많이 검출된 색상을 배경색으로 감지)
        if background_color is None: background_color = detect_background_color(image, img_channel)
        # grayscale 이미지에서는, 여러 채널값이 있어도 동일한 값을 갖는다. (0 ~ 255)
        mask = cv2.inRange(image[:, :, 0], background_color[0], background_color[0])
    elif color_type == "rgb":
        # 배경색 입력 없을 경우, 자동 감지 (가장 많이 검출된 색상을 배경색으로 감지)
        if background_color is None: background_color = detect_background_color(image, img_channel)
        # 배경 색상이 있을 경우, rgb 순서로 입력된 값을 bgr 순서로 변경 후 입력해준다. (cv2 이미지가 bgr 순서로 읽어들임 / 4채널일 경우 a값은 255(최대))
        elif img_channel == 3: background_color = (background_color[2], background_color[1], background_color[0])
        elif img_channel == 4: background_color = (background_color[2], background_color[1], background_color[0], 255)
        mask = cv2.inRange(image, background_color, background_color)  # 전체 색상 비교
    elif color_type == "rgba":
        # 배경색 입력 없을 경우, 자동 감지 (가장 많이 검출된 색상을 배경색으로 감지)
        if background_color is None: background_color = detect_background_color(image, img_channel)
        # 배경 색상이 있을 경우, rgb 순서로 입력된 값을 bgr 순서로 변경 후 입력해준다. (cv2 이미지가 bgr 순서로 읽어들임)
        else: background_color = (background_color[2], background_color[1], background_color[0], background_color[3])
        mask = cv2.inRange(image, background_color, background_color)  # 전체 색상 비교
    else:
        # 다른 선택이 없다면, 무조건 전부 auto 로 진행한다.
        background_color = detect_background_color(image, img_channel)
        mask = cv2.inRange(image, background_color, background_color)

    # 배경을 0, 오브젝트를 255로 변환
    object_mask = cv2.bitwise_not(mask)

    # min_area 기준으로 동일 오브젝트 판정한 mask
    merged_mask = merge_close_objects(mask, min_area)

    # 객체들의 바운딩 박스를 저장할 리스트
    bbox_list = []
    # 객체의 테두리(컨투어) 검출
    contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 각 객체에 대해 바운딩 박스 계산 후 리스트에 추가
    for contour in contours:
        # 바운딩 박스 계산
        x, y, w, h = cv2.boundingRect(contour)
        # merged_result의 바운딩 박스를 object_mask의 물체 테두리로 맞추기
        adjusted_bbox = adjust_bbox_to_original((x, y, w, h), object_mask)
        bbox_list.append(adjusted_bbox)
    # 좌측 상단부터 순서대로 정렬
    bbox_list.sort(key=lambda bbox: (bbox[1], bbox[0]))
    return bbox_list, image

def draw_bboxes_on_image(image, bbox_list):
    # 바운딩 박스를 이미지에 그리기
    for (x, y, x2, y2) in bbox_list:
        cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)  # 초록색으로 바운딩 박스 그리기

    # 결과 이미지와 object_mask, merged_mask를 표시
    cv2.imshow("Result Image with Bounding Boxes", image)

    # 사용자가 키를 누를 때까지 대기
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img_path = "images/mario4blog.png"
    bbox_list, image = detect_objects(img_path, color_type="alpha")
    print(bbox_list)
    draw_bboxes_on_image(image.copy(), bbox_list)