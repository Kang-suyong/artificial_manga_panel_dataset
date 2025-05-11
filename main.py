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

import preprocesing.config_file as cfg

# ===== 말풍선·텍스트 비활성화 =====
# preprocesing/config_file.py 에 정의된 값을 직접 덮어씁니다.
cfg.SPEECH_BUBBLE_SETTINGS["min_per_panel"] = 0
cfg.SPEECH_BUBBLE_SETTINGS["max_per_panel"] = 0
cfg.max_speech_bubbles_per_panel = 0
cfg.TEXT_SETTINGS["enabled"] = False
# ==================================



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
    parser.add_argument("--dry", action="store_true", default=False,
                        help="Dry-run mode: do not write output files")
    parser.add_argument("--run_tests", action="store_true",
                        help="Run unit tests before anything else")

    args = parser.parse_args()

    if args.run_tests:
        pytest.main(["tests/unit_tests/", "-q", "-x"])
        exit(0)

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
            return pd.DataFrame(), [], pd.DataFrame(), []  # same structure if you plan to load real data

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
