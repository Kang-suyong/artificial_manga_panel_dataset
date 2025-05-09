#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import cv2
import easyocr # EasyOCR 임포트
import numpy as np

# --- 사용자 설정 영역 시작 ---

SRC_DIR = r"/home/liont/artificial_manga_panel_dataset/datasets/image_dataset/db_illustrations_bw"
DST_DIR = r"/home/liont/artificial_manga_panel_dataset/datasets/image_dataset/filtered_illustrations_bw"
DEBUG_DIR = r"/home/liont/artificial_manga_panel_dataset/datasets/image_dataset/debug_images"
ENABLE_DEBUG = False

EASYOCR_LANGUAGES = ['ja', 'en']
EASYOCR_GPU = True

PREPROCESSING_PROFILES = {
    "preprocess_aggressive_scale_for_small_text": {
        "scale_factor": 2.5, "gaussian_blur_ksize": 0,
    },
    "preprocess_moderate_scale_sharp": {
        "scale_factor": 2.0, "gaussian_blur_ksize": 0,
    },
    "preprocess_default_no_change": {
        "scale_factor": 1.0, "gaussian_blur_ksize": 0,
    }
}

OCR_CRITERIA_EASYOCR = {
    "ocr_more_sensitive_v3": {
        "min_confidence": 0.38, "min_chars_per_box": 1, "min_valid_boxes": 1
    },
    "ocr_balanced_recall_v3": {
        "min_confidence": 0.48, "min_chars_per_box": 2, "min_valid_boxes": 1
    },
    "ocr_cautious_but_more_permissive_v3": {
        "min_confidence": 0.60, "min_chars_per_box": 2, "min_valid_boxes": 1
    }
}

STRATEGIES_TO_TRY = [
    ("preprocess_aggressive_scale_for_small_text", "ocr_more_sensitive_v3"),
    ("preprocess_moderate_scale_sharp", "ocr_balanced_recall_v3"),
    ("preprocess_default_no_change", "ocr_more_sensitive_v3"),
    ("preprocess_moderate_scale_sharp", "ocr_cautious_but_more_permissive_v3")
]

RENAME_SEQUENTIAL = True
OUTPUT_EXTENSION = "jpg"
MAX_FILES_TO_PROCESS = 0
MIN_CONF_DEBUG_DRAW_FOR_TEXT_FOCUS = 0.30

# --- 사용자 설정 영역 끝 ---

def has_text_easyocr(ocr_results, min_confidence, min_chars_per_box, min_valid_boxes):
    if not ocr_results:
        return False, ""
    valid_box_count = 0
    detected_texts_snippets = []
    for (bbox, text, prob) in ocr_results:
        if prob >= min_confidence and len(text.strip()) >= min_chars_per_box:
            valid_box_count += 1
            detected_texts_snippets.append(text.strip())
    if valid_box_count >= min_valid_boxes:
        return True, " ".join(detected_texts_snippets)
    return False, ""

def preprocess_image_for_easyocr(img_gray: np.ndarray, scale_factor, gaussian_blur_ksize) -> np.ndarray:
    processed_img = img_gray.copy()
    if gaussian_blur_ksize > 0:
        ksize = gaussian_blur_ksize if gaussian_blur_ksize % 2 != 0 else gaussian_blur_ksize + 1
        processed_img = cv2.GaussianBlur(processed_img, (ksize, ksize), 0)
    if scale_factor != 1.0 and scale_factor > 0:
        h, w = processed_img.shape
        processed_img = cv2.resize(processed_img, (int(w * scale_factor), int(h * scale_factor)),
                                   interpolation=cv2.INTER_CUBIC if scale_factor > 1.0 else cv2.INTER_AREA)
    return processed_img

def draw_ocr_easyocr_results(image_path_or_array, ocr_results, save_path, min_confidence_draw=0.1):
    try:
        if isinstance(image_path_or_array, str):
            img = cv2.imread(image_path_or_array)
            if img is None: return
        elif isinstance(image_path_or_array, np.ndarray):
            img = cv2.cvtColor(image_path_or_array, cv2.COLOR_GRAY2BGR) if len(image_path_or_array.shape) == 2 else image_path_or_array.copy()
        else: return
        img_copy = img.copy()
        if ocr_results:
            for (bbox, text, prob) in ocr_results:
                if prob >= min_confidence_draw:
                    (tl, tr, br, bl) = bbox
                    cv2.rectangle(img_copy, (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])), (0, 255, 0), 2)
        cv2.imwrite(save_path, img_copy)
    except Exception as e:
        print(f"[오류] 디버그 이미지 ({save_path}): {e}")

def filter_images_by_easyocr(reader):
    if not os.path.exists(SRC_DIR):
        print(f"[오류] 원본 폴더 없음: {SRC_DIR}")
        return
    os.makedirs(DST_DIR, exist_ok=True)
    if ENABLE_DEBUG: os.makedirs(DEBUG_DIR, exist_ok=True)

    files = sorted([f for f in os.listdir(SRC_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if MAX_FILES_TO_PROCESS > 0: files = files[:MAX_FILES_TO_PROCESS]
    
    kept_count, skipped_count = 0, 0
    for i, fname in enumerate(files):
        src_path = os.path.join(SRC_DIR, fname)
        overall_contains_text, overall_detected_snippet, strategy_that_detected_text = False, "", "N/A"
        img_for_ocr_processed, ocr_results_for_debug = None, None
        last_ocr_criteria_params = None # 마지막 사용된 OCR 기준 저장용

        for prep_profile_name, ocr_criteria_name in STRATEGIES_TO_TRY:
            prep_params = PREPROCESSING_PROFILES.get(prep_profile_name)
            ocr_criteria_params = OCR_CRITERIA_EASYOCR.get(ocr_criteria_name)
            last_ocr_criteria_params = ocr_criteria_params # 현재 사용된 기준 저장
            if not prep_params or not ocr_criteria_params:
                print(f"[경고] 프로파일 없음: {prep_profile_name} 또는 {ocr_criteria_name} ({fname})")
                continue
            
            img_gray = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                print(f"[경고] 이미지 로드 실패 (전처리): {fname}")
                continue
            
            current_processed_img = preprocess_image_for_easyocr(img_gray, prep_params["scale_factor"], prep_params["gaussian_blur_ksize"])
            img_for_ocr_processed = current_processed_img

            try:
                result = reader.readtext(img_for_ocr_processed)
                ocr_results_for_debug = result
            except Exception as e:
                print(f"[오류] EasyOCR ({fname}, 전략: {prep_profile_name}/{ocr_criteria_name}): {e}")
                ocr_results_for_debug = []
                continue 

            contains_text_this_strategy, snippet_this_strategy = has_text_easyocr(result, **ocr_criteria_params)
            if contains_text_this_strategy:
                overall_contains_text, overall_detected_snippet, strategy_that_detected_text = True, snippet_this_strategy, f"{prep_profile_name}/{ocr_criteria_name}"
                break
        
        if ENABLE_DEBUG:
            debug_fname_prefix = "TEXT_" if overall_contains_text else "NOTEXT_"
            detected_by_suffix = f"_by_{strategy_that_detected_text.replace('/','-')}" if overall_contains_text else ""
            safe_fname_part, original_ext = os.path.splitext(fname)[0].replace(":", "_"), os.path.splitext(fname)[1]
            debug_save_path = os.path.join(DEBUG_DIR, f"{debug_fname_prefix}{safe_fname_part}{detected_by_suffix}{original_ext}")
            
            min_conf_val = MIN_CONF_DEBUG_DRAW_FOR_TEXT_FOCUS # 기본 디버그 신뢰도
            if overall_contains_text and last_ocr_criteria_params : # 텍스트 감지 시 해당 전략의 min_confidence 사용
                 min_conf_val = last_ocr_criteria_params.get("min_confidence", MIN_CONF_DEBUG_DRAW_FOR_TEXT_FOCUS)
            
            if img_for_ocr_processed is not None:
                draw_ocr_easyocr_results(img_for_ocr_processed, ocr_results_for_debug, debug_save_path, min_confidence_draw=min_conf_val)
            else: # 전처리가 없었거나 실패한 경우 (거의 발생 안 함)
                 draw_ocr_easyocr_results(src_path, ocr_results_for_debug, debug_save_path, min_confidence_draw=min_conf_val)

        if not overall_contains_text:
            kept_count += 1
            dst_fname = f"{kept_count}.{OUTPUT_EXTENSION}" if RENAME_SEQUENTIAL else fname
            shutil.copy2(src_path, os.path.join(DST_DIR, dst_fname))
            print(f"[KEEP] ({i+1}/{len(files)}) {fname} → {dst_fname}")
        else:
            skipped_count += 1
            # 오류 수정: replace('\n', ' ') 호출 후 strip() 적용
            snippet_to_print = overall_detected_snippet.replace('\n', ' ').strip()[:60]
            print(f"[SKIP] ({i+1}/{len(files)}) {fname} (감지: {strategy_that_detected_text}, 내용: {snippet_to_print}...)")


    print(f"\n완료: 총 {len(files)} 중 {kept_count} 저장 → {DST_DIR}, 스킵: {skipped_count}")

if __name__ == "__main__":
    print("EasyOCR Reader 초기화 중...")
    try:
        reader = easyocr.Reader(EASYOCR_LANGUAGES, gpu=EASYOCR_GPU)
    except Exception as e:
        print(f"[오류] EasyOCR 초기화 실패: {e}")
        sys.exit(1)
    print("EasyOCR Reader 초기화 완료.")
    filter_images_by_easyocr(reader)