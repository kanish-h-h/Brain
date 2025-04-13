#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOLDER_NAME="$(basename "$SCRIPT_DIR")"
ZIP_FILE="$SCRIPT_DIR/$FOLDER_NAME.zip"

# Create a zip archive of the current folder
zip -r "$ZIP_FILE" "$SCRIPT_DIR"

echo "Folder '$FOLDER_NAME' has been compressed into '$ZIP_FILE'."

#gh release create v2.0 Brain.zip --title "v2.0" --notes "MLE Roadmap Brain.zip"
