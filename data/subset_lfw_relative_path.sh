#!/bin/bash

# Resolve absolute path of script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && readlink -f .)"

# Prompt for LFW dataset path (default value: "lfw_original")
read -p "Enter the relative path to the LFW dataset (default: lfw_original): " rel_src_dir
SRC_DIR="${SCRIPT_DIR}/${rel_src_dir:-lfw_original}"

# Prompt for destination directory path
read -p "Enter the relative path to the destination directory: " rel_dest_dir
DEST_DIR="$SCRIPT_DIR/$rel_dest_dir"

# Prompt for minimum number of images
read -p "Enter the minimum number of images required in each subdirectory: " min_images

# Prompt for number of subdirectories to take (optional)
read -p "Enter the number of subdirectories to take (or leave blank to take all): " num_subdirs

count=0

for dir in "$SRC_DIR"/*; do
    if [ -d "$dir" ]; then
        num_files=$(ls -1q "$dir"/*.jpg 2>/dev/null | wc -l)
        if [ $num_files -ge $min_images ]; then
            count=$((count+1))
            cp -r "$dir" "$DEST_DIR"
            if [ -n "$num_subdirs" ] && [ $count -eq $num_subdirs ]; then
                exit 0
            fi
        fi
    fi
done

