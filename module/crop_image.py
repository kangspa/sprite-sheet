import cv2
import os
from split_frame_fixed import split_frame
from split_frame_background import detect_objects

def crop_and_save(image_path, bbox_list, output_dir):
    """
    이미지에서 지정된 bbox 영역을 잘라 저장하는 함수
    
    :param image_path: 원본 이미지 경로
    :param bbox_list: (x1, y1, x2, y2) 형태의 튜플 리스트
    :param output_dir: 자른 이미지를 저장할 디렉토리
    """
    # 이미지 읽기
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("이미지를 불러올 수 없습니다: " + image_path)
    
    # 출력 디렉토리 확인 및 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 파일 이름 추출 (경로 마지막 파일명에서 확장자 제거 / 파일명에 .이 있을 경우 대비)
    file_name = ".".join(image_path.split("/")[-1].split(".")[:-1])
    # 확장자 추출 (png일 경우 투명 유지를 위해)
    file_ext = image_path.split("/")[-1].split(".")[-1]
    
    # 바운딩 박스 기준으로 이미지 자르기
    for idx, (x1, y1, x2, y2) in enumerate(bbox_list):
        cropped_image = image[y1:y2, x1:x2]
        output_path = os.path.join(output_dir, f"{file_name}_{idx}.{file_ext}")
        cv2.imwrite(output_path, cropped_image)
        print(f"Saved: {output_path}")
    
    print("모든 잘린 이미지가 저장되었습니다.")

# 사용 예시
# bbox_list = [(50, 50, 200, 200), (30, 30, 100, 100)]
# crop_and_save("input.jpg", bbox_list, "output_crops")
if __name__ == "__main__":
    # img_path = "images/dude.png"
    # bbox_list, img = split_frame(img_path, 32, 48)
    
    img_path = "images/mario4blog.png"
    bbox_list, image = detect_objects(img_path, color_type="alpha")
    print(bbox_list)
    crop_and_save(img_path, bbox_list, "output")