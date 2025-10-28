import os
import sys
import argparse
from typing import List, Dict, Any

import pandas as pd
from deepface import DeepFace


def find_image_files(data_root: str) -> List[str]:
    """Recursively find image files under data_root."""
    allowed_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_paths: List[str] = []

    for root, _, files in os.walk(data_root):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in allowed_exts:
                full_path = os.path.abspath(os.path.join(root, name))
                image_paths.append(full_path)

    return image_paths


def load_file_list(file_list_path: str) -> List[str]:
    """Load explicit list of image paths from a newline-delimited file."""
    paths: List[str] = []
    allowed_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    with open(file_list_path, 'r') as f:
        for line in f:
            p = line.strip()
            if not p:
                continue
            ext = os.path.splitext(p)[1].lower()
            if ext in allowed_exts and os.path.exists(p):
                paths.append(p)
    return paths


def compute_age_group(age_value: Any) -> str:
    """Map a numeric age to an age group label.

    Groups: Child (0-14), Youth (15-24), Adult (25-44), Middle-age Adult (45-64), Senior (65+)
    """
    try:
        age_int = int(age_value)
    except (TypeError, ValueError):
        return 'unknown'

    if age_int <= 14:
        return 'Child'
    if age_int <= 24:
        return 'Youth'
    if age_int <= 44:
        return 'Adult'
    if age_int <= 64:
        return 'Middle-age Adult'
    return 'Senior'


def analyze_image_with_deepface(image_path: str) -> Dict[str, Any]:
    """Analyze a single image using DeepFace for specified attributes."""
    try:
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['age', 'gender', 'race', 'emotion'],
            enforce_detection=False,
            detector_backend='mediapipe'
        )

        if isinstance(result, list):
            result = result[0]

        age_value = result.get('age', -1)
        try:
            age = int(age_value)
        except (TypeError, ValueError):
            age = -1

        age_group = compute_age_group(age) if age != -1 else 'unknown'

        # Compute gender label from probabilities if available
        gender_label = 'unknown'
        gender_probs = result.get('gender')
        if isinstance(gender_probs, dict) and len(gender_probs) > 0:
            try:
                gender_label = max(gender_probs.items(), key=lambda kv: float(kv[1]))[0]
            except Exception:
                gender_label = result.get('dominant_gender', result.get('gender', 'unknown'))
        else:
            gender_label = result.get('dominant_gender', result.get('gender', 'unknown'))

        output: Dict[str, Any] = {
            'image_path': image_path,
            'image_id': image_path,
            'age': age,
            'age_group': age_group,
            'gender': gender_label,
            'race': result.get('dominant_race', 'unknown'),
            'emotion': result.get('dominant_emotion', 'unknown')
        }

        return output
    except Exception as e:
        return {
            'image_path': image_path,
            'image_id': image_path,
            'age': -1,
            'age_group': 'unknown',
            'gender': 'unknown',
            'race': 'unknown',
            'emotion': 'unknown',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='DeepFace-only image attribute extraction')
    parser.add_argument('--data_root', type=str, required=False, help='Root directory to recursively scan for images')
    parser.add_argument('--file_list', type=str, required=False, help='Text file with one absolute image path per line')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output CSV with attributes')
    args = parser.parse_args()

    image_paths: List[str] = []
    if args.file_list:
        if not os.path.exists(args.file_list):
            print(f"Error: file_list not found: {args.file_list}")
            sys.exit(1)
        image_paths = load_file_list(args.file_list)
        if not image_paths:
            print("Error: No valid image paths found in file_list")
            sys.exit(1)
    else:
        if not args.data_root or not os.path.isdir(args.data_root):
            print("Error: provide --file_list or a valid --data_root")
            sys.exit(1)
        print(f"Scanning for images under {args.data_root} ...")
        image_paths = find_image_files(args.data_root)
        if not image_paths:
            print("Error: No image files found in data_root")
            sys.exit(1)

    image_paths.sort()
    print(f"Found {len(image_paths)} images. Running DeepFace analysis...")

    results = [analyze_image_with_deepface(p) for p in image_paths]

    df = pd.DataFrame(results)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    print(f"Saving results to {args.output_path}")
    df.to_csv(args.output_path, index=False)
    print("Done!")


if __name__ == '__main__':
    main()


