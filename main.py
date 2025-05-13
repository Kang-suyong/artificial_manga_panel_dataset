#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scraping.download_images import download_db_illustrations
from preprocesing.convert_images import convert_images_to_bw
from preprocesing.layout_engine.page_creator import render_pages
from preprocesing.layout_engine.page_dataset_creator import create_page_metadata
from tqdm import tqdm
import os
import pandas as pd
from argparse import ArgumentParser
import pytest
import json
from PIL import Image

import preprocesing.config_file as cfg

# ===== 말풍선·텍스트 비활성화 =====
# preprocesing/config_file.py 에 정의된 값을 직접 덮어씁니다.
cfg.TEXT_SETTINGS["enabled"] = False
# ==================================

# ===== 말풍선 활성화/비활성화 설정 =====
# 기본적으로 말풍선은 활성화되어 있습니다
cfg.SPEECH_BUBBLE_SETTINGS["enabled"] = True
# ====================================



def find_image_dir():
    """
    Identify the correct BW image directory among possible candidates.
    Raises RuntimeError if none found or all are empty.
    """
    candidates = [
        "datasets/image_dataset/filtered_illustrations_bw",
    ]
    for d in candidates:
        if os.path.isdir(d) and os.listdir(d):
            return d + os.sep
    raise RuntimeError(
        "No BW image folder found. Please ensure images are downloaded and converted.\n"
        "Expected one of:\n  " + "\n  ".join(candidates)
    )


def find_speech_bubbles():
    """
    말풍선 이미지와 태그를 찾아 반환합니다.
    
    :return: 말풍선 이미지 경로 목록, 태그 데이터프레임
    """
    # 말풍선 이미지 디렉터리 찾기
    bubble_candidates = [
        "datasets/speech_bubbles/files",  # 사용자가 지정한 우선 경로
        "datasets/speech_bubbles",        # 기존 경로
        "./speech_bubbles",                # 기존 경로
        # "/dataset/speech_bubbles",  # 시스템 전체 절대 경로는 제거하거나 주석 처리합니다.
    ]
    
    bubble_dir = None
    for d in bubble_candidates:
        if os.path.isdir(d) and os.listdir(d):
            bubble_dir = d + os.sep
            print(f"말풍선 디렉터리를 찾았습니다: {bubble_dir}")
            break
    
    # 말풍선 이미지가 없으면 빈 목록 반환
    if not bubble_dir:
        print("말풍선 디렉터리를 찾을 수 없습니다. datasets/speech_bubbles/files 또는 datasets/speech_bubbles 디렉터리를 확인해주세요.")
        return [], pd.DataFrame()
    
    # 말풍선 이미지 파일 찾기
    bubble_files = []
    for file in os.listdir(bubble_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            bubble_files.append(bubble_dir + file)
    
    print(f"말풍선 이미지 {len(bubble_files)}개를 찾았습니다.")
    
    # 말풍선 태그 생성 (create_default_tags 함수 사용)
    if bubble_files:
        bubble_tags = create_default_tags(bubble_files)
    else:
        bubble_tags = pd.DataFrame() # 빈 데이터프레임 반환
    
    return bubble_files, bubble_tags


def create_default_tags(speech_bubble_files):
    """
    말풍선 파일들에 대한 기본 태그를 생성합니다.
    각 말풍선 이미지 중앙에 텍스트 영역을 배치합니다.
    
    :param speech_bubble_files: 말풍선 이미지 파일 경로 목록
    :return: 말풍선 태그 정보가 있는 데이터프레임
    """
    print("기본 말풍선 태그 생성 중...")
    
    # 파일별 기본 쓰기 영역 생성
    labels = []
    for file_path in speech_bubble_files:
        try:
            # 이미지 크기 확인
            with Image.open(file_path) as img:
                width, height = img.size
            
            # 이미지 중앙 60% 영역을 텍스트 영역으로 설정
            x_margin = width * 0.2
            y_margin = height * 0.2
            
            # 말풍선 중앙에 사각형 영역 생성
            points = [
                [x_margin, y_margin],  # 좌상단
                [width - x_margin, y_margin],  # 우상단
                [width - x_margin, height - y_margin],  # 우하단
                [x_margin, height - y_margin]  # 좌하단
            ]
            
            # JSON 형식의 레이블 생성
            label_json = json.dumps([{
                "points": points,
                "shape_type": "polygon"
            }])
            
            labels.append(label_json)
            
        except Exception as e:
            print(f"파일 {file_path} 처리 중 오류: {e}")
            # 기본값 사용
            labels.append('[{"points": [[10, 10], [90, 10], [90, 90], [10, 90]], "shape_type": "polygon"}]')
    
    # 데이터프레임 생성
    tags_df = pd.DataFrame({
        'imagename': speech_bubble_files,
        'label': labels
    })
    
    print(f"기본 태그 {len(tags_df)}개 생성 완료")
    return tags_df


if __name__ == '__main__':

    usage_message = """
    AMP Dataset Creator (Image-Only Mode)
    Generate manga-style panels using only preprocessed images.
    """

    parser = ArgumentParser(usage=usage_message)
    parser.add_argument("--download_jesc", "-dj",
                        action="store_true",
                        help="Download JESC Japanese/English dialogue corpus")
    parser.add_argument("--download_fonts", "-df",
                        action="store_true",
                        help="Scrape font files")
    parser.add_argument("--download_images", "-di",
                        action="store_true",
                        help="Download anime illustrations from Kaggle and convert to BW")
    parser.add_argument("--download_speech_bubbles", "-ds",
                        action="store_true",
                        help="Download speech bubbles from Gcloud")
    parser.add_argument("--verify_fonts", "-vf",
                        action="store_true",
                        help="Verify fonts for minimum coverage")

    parser.add_argument("--convert_images", "-ci",
                        action="store_true",
                        help="Convert existing images to black and white")

    parser.add_argument("--create_page_metadata", "-pm", nargs=1, type=int,
                        help="Generate metadata for N pages (image-only mode)")
    parser.add_argument("--render_pages", "-rp", action="store_true",
                        help="Render pages from existing metadata")
    parser.add_argument("--generate_pages", "-gp", nargs=1, type=int,
                        help="One-shot: create metadata and render N pages")
    parser.add_argument("--images_only", action="store_true",
                        help="Strip out all text and speech bubbles; only frames and images")
    parser.add_argument("--speech_bubbles", action="store_true", default=True,
                        help="Include speech bubbles without text (default: enabled)")
    parser.add_argument("--no_speech_bubbles", action="store_true",
                        help="Disable speech bubbles completely")
    parser.add_argument("--dry", action="store_true", default=False,
                        help="Dry-run mode: do not write output files")
    parser.add_argument("--run_tests", action="store_true",
                        help="Run unit tests before anything else")

    args = parser.parse_args()

    if args.run_tests:
        pytest.main(["tests/unit_tests/", "-q", "-x"])
        exit(0)
        
    # 말풍선 설정 업데이트
    if args.images_only:
        # images_only가 지정되면 말풍선과 텍스트 모두 비활성화
        cfg.TEXT_SETTINGS["enabled"] = False
        cfg.SPEECH_BUBBLE_SETTINGS["enabled"] = False
    elif args.no_speech_bubbles:
        # no_speech_bubbles가 지정되면 말풍선만 비활성화
        cfg.SPEECH_BUBBLE_SETTINGS["enabled"] = False
    else:
        # 기본적으로 말풍선은 활성화, 텍스트는 비활성화
        cfg.TEXT_SETTINGS["enabled"] = False
        cfg.SPEECH_BUBBLE_SETTINGS["enabled"] = True

    if args.download_images:
        download_db_illustrations()
        convert_images_to_bw()

    if args.convert_images:
        convert_images_to_bw()

    def prepare_image_only_inputs():
        """Return empty placeholders for text/bubbles when images_only."""
        if args.images_only:
            return pd.DataFrame(), [], pd.DataFrame(), []
        else:
            # 말풍선 데이터 로드 (텍스트 없이)
            text_dataset = pd.DataFrame()  # 빈 텍스트 데이터셋
            
            # 말풍선 파일 및 태그 로드
            speech_bubble_files, speech_bubble_tags = find_speech_bubbles()
            
            # 폰트 파일은 빈 리스트로 설정 (텍스트 없음)
            viable_font_files = []
            
            return text_dataset, speech_bubble_files, speech_bubble_tags, viable_font_files

    # 3) Metadata generation (image-only)
    if args.create_page_metadata:
        n = args.create_page_metadata[0]
        metadata_folder = "datasets/page_metadata/"
        os.makedirs(metadata_folder, exist_ok=True)

        image_dir_path = find_image_dir()
        image_list = sorted(os.listdir(image_dir_path))

        text_dataset, speech_bubble_files, speech_bubble_tags, viable_font_files = prepare_image_only_inputs()

        print(f"Creating metadata for {n} image-only pages...")
        for _ in tqdm(range(n)):
            page = create_page_metadata(
                image_list,
                image_dir_path,
                viable_font_files,
                text_dataset,
                speech_bubble_files,
                speech_bubble_tags
            )
            page.dump_data(metadata_folder, dry=args.dry)

    # 4) Render existing metadata to images
    if args.render_pages:
        metadata_folder = "datasets/page_metadata/"
        images_folder = "datasets/page_images/"
        os.makedirs(images_folder, exist_ok=True)

        print("Rendering pages from metadata...")
        render_pages(metadata_folder, images_folder, dry=args.dry)

    # 5) One-shot: create metadata + render
    if args.generate_pages:
        n = args.generate_pages[0]
        metadata_folder = "datasets/page_metadata/"
        os.makedirs(metadata_folder, exist_ok=True)

        image_dir_path = find_image_dir()
        image_list = sorted(os.listdir(image_dir_path))

        text_dataset, speech_bubble_files, speech_bubble_tags, viable_font_files = prepare_image_only_inputs()

        print(f"Generating and rendering {n} image-only pages...")
        for _ in tqdm(range(n)):
            page = create_page_metadata(
                image_list,
                image_dir_path,
                viable_font_files,
                text_dataset,
                speech_bubble_files,
                speech_bubble_tags
            )
            page.dump_data(metadata_folder, dry=args.dry)

        images_folder = "datasets/page_images/"
        os.makedirs(images_folder, exist_ok=True)
        render_pages(metadata_folder, images_folder, dry=args.dry)
