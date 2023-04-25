#!/bin/bash

# Prompt for LFW dataset path
read -p "Enter the path to the LFW dataset: " src_dir

# Prompt for destination directory path
read -p "Enter the path to the destination directory: " dest_dir

# Prompt for minimum number of images
read -p "Enter the minimum number of images required in each subdirectory: " min_images

# Prompt for number of subdirectories to take
read -p "Enter the number of subdirectories to take: " num_subdirs

count=0

for dir in "$src_dir"/*; do
    if [ -d "$dir" ]; then
        num_files=$(ls -1q "$dir"/*.jpg 2>/dev/null | wc -l)
        if [ $num_files -ge $min_images ]; then
            count=$((count+1))
            cp -r "$dir" "$dest_dir"
            if [ $count -eq $num_subdirs ]; then
                exit 0
            fi
        fi
    fi
done
