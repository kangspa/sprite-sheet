import cv2
from get_padding import get_padding

def split_frame(img_path, width, height, padding=[0]):
    # get_padding 함수로 4방향 패딩 값 각각 입력 진행
    padding_top, padding_right, padding_bottom, padding_left = get_padding(padding)
    # cv2로 이미지 읽어들인 후, 정보값 입력
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img_height, img_width, img_channel = img.shape
    # 각 프레임을 저장하기 위한 bbox 저장 배열
    bbox_list = []
    # 시작 위치는 프레임의 상단, 좌측 공간만큼 비워둔 후 진행
    # 각 프레임 사이 간격은 top+bottom, left+right만큼이다.
    for y in range(padding_top, img_height, height+(padding_top+padding_bottom)):
        for x in range(padding_left, img_width, width+(padding_left+padding_right)):
            # 만약 그려지는 bbox가 이미지를 넘기면 패스한다.
            if ((y + height) <= img_height) & ((x + width) <= img_width):
                bbox = (x, y, x + width, y + height)
                bbox_list.append(bbox) 
    return bbox_list, img

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
    bbox_list, img = split_frame("images/dude.png", 32, 48)
    print(bbox_list)
    draw_bboxes_on_image(img.copy(), bbox_list)