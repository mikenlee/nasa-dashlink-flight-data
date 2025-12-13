#!/bin/bash

# Re-download corrupted/incomplete zip files from NASA Dashlink
# These files failed to unzip due to incomplete downloads

cd "$(dirname "$0")"

BASE_URL="https://c3.ndc.nasa.gov/dashlink/static/media/dataset"
FAILED_FILES_LIST="failed_files.txt"

if [ ! -f "$FAILED_FILES_LIST" ]; then
    echo "Error: $FAILED_FILES_LIST not found"
    exit 1
fi

# Read file names from failed_files.txt
FAILED_FILES=()
while IFS= read -r line || [ -n "$line" ]; do
    [ -n "$line" ] && FAILED_FILES+=("$line")
done < "$FAILED_FILES_LIST"

echo "Re-downloading ${#FAILED_FILES[@]} failed zip files from $FAILED_FILES_LIST..."
echo ""

for file in "${FAILED_FILES[@]}"; do
    echo "----------------------------------------"
    echo "Downloading: $file"

    # Remove corrupted file first
    rm -f "$file"

    # Remove empty extraction directory if it exists
    dir_name="${file%.zip}"
    if [ -d "$dir_name" ] && [ -z "$(ls -A "$dir_name" 2>/dev/null)" ]; then
        rmdir "$dir_name"
    fi

    # Download with retry (--retry 5) and connection timeout
    curl --retry 5 --retry-delay 10 --connect-timeout 30 --max-time 1800 \
         -o "$file" "$BASE_URL/$file"

    if [ $? -eq 0 ]; then
        echo "✓ Downloaded: $file"
        # Verify the zip file
        if unzip -t "$file" > /dev/null 2>&1; then
            echo "✓ Verified: $file is valid"
        else
            echo "✗ Warning: $file may still be corrupted"
        fi
    else
        echo "✗ Failed to download: $file"
    fi
    echo ""
done

echo "Done! Re-run unzip_flight_data() in your notebook to extract the new files."
