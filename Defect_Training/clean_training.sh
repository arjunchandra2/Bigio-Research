#!/bin/bash

# Specify the directory containing the folders
directory="/projectnb/npbssmic/ac25/Defect_Training/runs/detect"

# Check if the directory exists
if [ ! -d "$directory" ]; then
    echo "Directory $directory not found."
    exit 1
fi

# Loop through the folders in the directory
for folder in "$directory"/*; do
    # Check if the folder is a directory and starts with "train"
    if [ -d "$folder" ] && [[ "$(basename "$folder")" == train* ]]; then
        # Check if the folder is "train11", if so, skip deletion
        if [ "$(basename "$folder")" == "train11" ]; then
            echo "Skipping folder: $folder"
        else
            # Ask for confirmation before deleting the folder
            read -p "Do you want to delete folder $folder? (yes/no): " choice
            case "$choice" in 
                yes|YES|y|Y)
                    # Delete the folder
                    echo "Deleting folder: $folder"
                    rm -r "$folder"
                    ;;
                no|NO|n|N)
                    echo "Skipping deletion of folder: $folder"
                    ;;
                *)
                    echo "Invalid choice, skipping deletion of folder: $folder"
                    ;;
            esac
        fi
    fi
done

echo "Deletion complete."
