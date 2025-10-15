import os
import sys
import argparse
from typing import List

from additional_attributes import ATTRIBUTE_CONFIG, process_dataframe


def find_image_files(data_root: str) -> List[str]:
    """Recursively find image files under data_root.

    Only files with allowed image extensions are included.
    """
    allowed_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_paths: List[str] = []

    for root, _, files in os.walk(data_root):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in allowed_exts:
                full_path = os.path.abspath(os.path.join(root, name))
                image_paths.append(full_path)

    return image_paths


def main():
    parser = argparse.ArgumentParser(description='Generate additional attributes for images (auto-discover mode)')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory to recursively scan for images')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output CSV with attributes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--debug_vis', action='store_true', help='Generate debug visualizations')
    parser.add_argument('--debug_vis_dir', type=str, default=None, help='Directory to save debug visualizations')
    parser.add_argument('--max_debug_images', type=int, default=20, help='Maximum number of debug images to generate')
    parser.add_argument('--disable_deepface', action='store_true', help='Disable DeepFace analysis for faster processing')
    parser.add_argument('--disable_emotions', action='store_true', help='Disable emotion detection for faster processing')
    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        print(f"Error: data_root does not exist or is not a directory: {args.data_root}")
        sys.exit(1)

    if args.disable_deepface:
        ATTRIBUTE_CONFIG['deepface']['enabled'] = False
    if args.disable_emotions:
        ATTRIBUTE_CONFIG['emotions']['enabled'] = False

    # Create debug visualization directory if needed
    if args.debug_vis and args.debug_vis_dir:
        os.makedirs(args.debug_vis_dir, exist_ok=True)

    print(f"Scanning for images under {args.data_root} ...")
    image_paths = find_image_files(args.data_root)

    if not image_paths:
        print("Error: No image files found in data_root")
        sys.exit(1)

    print(f"Processing {len(image_paths)} images with batch size {args.batch_size}")

    results_df = process_dataframe(
        image_paths=image_paths,
        batch_size=args.batch_size,
        debug_vis=args.debug_vis,
        debug_vis_dir=args.debug_vis_dir,
        max_debug_images=args.max_debug_images
    )

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    print(f"Saving results to {args.output_path}")
    results_df.to_csv(args.output_path)
    print("Done!")


if __name__ == '__main__':
    main()


