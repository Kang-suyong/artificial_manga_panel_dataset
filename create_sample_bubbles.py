#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image, ImageDraw
import os

# 말풍선 저장 디렉토리
SPEECH_BUBBLE_DIR = "datasets/speech_bubbles"

# 디렉토리가 없으면 생성
if not os.path.exists(SPEECH_BUBBLE_DIR):
    os.makedirs(SPEECH_BUBBLE_DIR, exist_ok=True)

def create_speech_bubble(filename, width, height, shape="ellipse", color=(255, 255, 255, 255), bg_color=(0, 0, 0, 0)):
    """
    기본 말풍선 이미지를 생성합니다.
    
    :param filename: 저장할 파일명
    :param width: 말풍선 너비
    :param height: 말풍선 높이
    :param shape: 모양 (ellipse, rectangle)
    :param color: 말풍선 색상 (RGBA)
    :param bg_color: 배경 색상 (RGBA)
    """
    # 투명 배경의 이미지 생성
    image = Image.new("RGBA", (width, height), bg_color)
    draw = ImageDraw.Draw(image)
    
    # 말풍선 모양 그리기
    if shape == "ellipse":
        draw.ellipse((10, 10, width-10, height-10), fill=color, outline=(0, 0, 0, 255), width=2)
    elif shape == "rectangle":
        draw.rounded_rectangle((10, 10, width-10, height-10), radius=20, fill=color, outline=(0, 0, 0, 255), width=2)
    elif shape == "cloud":
        # 구름 모양의 말풍선
        # 중앙 원
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 3
        draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), 
                     fill=color, outline=(0, 0, 0, 255), width=2)
        
        # 주변 작은 원들
        small_radius = radius // 2
        positions = [
            (center_x - radius - small_radius//2, center_y),
            (center_x + radius + small_radius//2, center_y),
            (center_x, center_y - radius - small_radius//2),
            (center_x, center_y + radius + small_radius//2),
            (center_x - radius//1.5, center_y - radius//1.5),
            (center_x + radius//1.5, center_y - radius//1.5),
            (center_x - radius//1.5, center_y + radius//1.5),
            (center_x + radius//1.5, center_y + radius//1.5),
        ]
        
        for pos_x, pos_y in positions:
            draw.ellipse((pos_x - small_radius, pos_y - small_radius, 
                           pos_x + small_radius, pos_y + small_radius), 
                          fill=color, outline=(0, 0, 0, 255), width=2)
    
    # 파일 저장
    file_path = os.path.join(SPEECH_BUBBLE_DIR, filename)
    image.save(file_path, "PNG")
    print(f"말풍선 이미지 생성: {file_path}")
    return file_path

# 다양한 말풍선 이미지 생성
if __name__ == "__main__":
    # 타원형 말풍선
    create_speech_bubble("bubble_ellipse1.png", 300, 200, "ellipse")
    create_speech_bubble("bubble_ellipse2.png", 250, 150, "ellipse")
    
    # 직사각형 말풍선
    create_speech_bubble("bubble_rectangle1.png", 300, 200, "rectangle")
    create_speech_bubble("bubble_rectangle2.png", 250, 150, "rectangle")
    
    # 구름형 말풍선
    create_speech_bubble("bubble_cloud1.png", 300, 300, "cloud")
    create_speech_bubble("bubble_cloud2.png", 400, 200, "cloud")
    
    print("말풍선 이미지 생성 완료!") 