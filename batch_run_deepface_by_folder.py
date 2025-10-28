import os
import sys
import csv
import argparse
import tempfile
import shutil
import subprocess
from typing import Dict, List, Tuple


ALLOWED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def is_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTS


def scan_folders_with_images(data_root: str) -> Dict[str, List[str]]:
    """Return mapping of folder_path -> list of image files directly in that folder.

    Only files directly in the folder are included (not recursive) to avoid duplication
    when processing subfolders independently.
    """
    folder_to_files: Dict[str, List[str]] = {}
    for root, _, files in os.walk(data_root):
        images_here = [os.path.abspath(os.path.join(root, f)) for f in files if is_image_file(f)]
        if images_here:
            folder_to_files[root] = images_here
    return folder_to_files


def write_folders_csv(folder_to_files: Dict[str, List[str]], folders_csv_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(folders_csv_path)), exist_ok=True)
    with open(folders_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['folder_path', 'num_images'])
        for folder, files in sorted(folder_to_files.items()):
            writer.writerow([folder, len(files)])


def append_csv(src_csv: str, dst_csv: str) -> None:
    """Append src_csv rows to dst_csv, writing header only if dst doesn't exist."""
    os.makedirs(os.path.dirname(os.path.abspath(dst_csv)), exist_ok=True)
    if not os.path.exists(dst_csv):
        shutil.copyfile(src_csv, dst_csv)
        return

    with open(src_csv, 'r') as s, open(dst_csv, 'a') as d:
        first = True
        for line in s:
            if first:
                first = False
                continue  # skip header
            d.write(line)


def run_deepface_for_folder(
    folder: str,
    images: List[str],
    deepface_script: str,
    work_dir: str
) -> Tuple[bool, str]:
    """Run deepface-only script for a specific list of files via --file_list.

    Returns (ok, tmp_csv_path or error_message).
    """
    # Prepare a per-folder file list without copying/moving images
    tmp_list_fd, tmp_list_path = tempfile.mkstemp(prefix='df_filelist_', suffix='.txt', dir=work_dir)
    try:
        with os.fdopen(tmp_list_fd, 'w') as f:
            for img in images:
                f.write(img + '\n')

        # Write results CSV path
        tmp_fd, tmp_csv = tempfile.mkstemp(prefix='df_results_', suffix='.csv', dir=work_dir)
        os.close(tmp_fd)

        cmd = [sys.executable, deepface_script, '--file_list', tmp_list_path, '--output_path', tmp_csv]
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode != 0:
            return False, f"Subprocess failed for {folder}: {completed.stderr.strip()}"
        if not os.path.exists(tmp_csv):
            return False, f"Expected output CSV missing for {folder}: {tmp_csv}"
        return True, tmp_csv
    finally:
        try:
            os.remove(tmp_list_path)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description='Batch-run DeepFace-only script per image folder and consolidate results')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory to scan for images')
    parser.add_argument('--output_csv', type=str, required=True, help='Central CSV to append all results to')
    parser.add_argument('--folders_csv', type=str, default=None, help='Optional CSV to write the list of folders and counts')
    parser.add_argument('--work_dir', type=str, default=None, help='Working directory for temporary per-folder runs')
    parser.add_argument('--script_path', type=str, default=None, help='Path to additional_attributes_deepface_only.py')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of folders to process')
    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        print(f"Error: data_root does not exist or is not a directory: {args.data_root}")
        sys.exit(1)

    script_path = args.script_path
    if script_path is None:
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'additional_attributes_deepface_only.py')
    if not os.path.exists(script_path):
        print(f"Error: DeepFace-only script not found at {script_path}")
        sys.exit(1)

    work_dir = args.work_dir or os.path.join(os.path.dirname(os.path.abspath(args.output_csv)), '.df_batch_work')
    os.makedirs(work_dir, exist_ok=True)

    print(f"Scanning folders under {args.data_root} ...")
    folder_to_files = scan_folders_with_images(args.data_root)
    print(f"Found {len(folder_to_files)} folders containing images")

    if args.folders_csv:
        write_folders_csv(folder_to_files, args.folders_csv)
        print(f"Wrote folder list to {args.folders_csv}")

    processed = 0
    for folder, images in sorted(folder_to_files.items()):
        if args.limit is not None and processed >= args.limit:
            break
        print(f"Processing folder {folder} with {len(images)} images ...")
        ok, result_or_err = run_deepface_for_folder(
            folder=folder,
            images=images,
            deepface_script=script_path,
            work_dir=work_dir
        )
        if not ok:
            print(result_or_err)
            continue
        try:
            append_csv(result_or_err, args.output_csv)
        except Exception as e:
            print(f"Failed to append results for {folder}: {e}")
            continue
        processed += 1
        print(f"Appended results for {folder}")

    print(f"Done. Processed {processed} folders. Output: {args.output_csv}")


if __name__ == '__main__':
    main()


