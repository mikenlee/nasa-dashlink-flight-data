#!/bin/bash

# Verify all zip files, extract valid ones, delete after successful extraction
# Lists any corrupted files that need re-downloading

cd "$(dirname "$0")"

echo "=========================================="
echo "ZIP FILE VERIFICATION AND EXTRACTION"
echo "=========================================="
echo ""

FAILED_FILES=()
SUCCESS_COUNT=0
ALREADY_EXTRACTED=0

for zip_file in Tail_*.zip; do
    # Skip if no zip files found
    [[ -e "$zip_file" ]] || continue

    dir_name="${zip_file%.zip}"

    echo -n "Checking: $zip_file ... "

    # Test zip file integrity
    if unzip -t "$zip_file" > /dev/null 2>&1; then
        # Check if already extracted (directory exists and has files)
        if [ -d "$dir_name" ] && [ "$(ls -A "$dir_name" 2>/dev/null)" ]; then
            echo "already extracted, deleting zip"
            rm -f "$zip_file"
            ((ALREADY_EXTRACTED++))
        else
            # Extract the zip file
            echo -n "extracting... "
            mkdir -p "$dir_name"
            if unzip -q -o "$zip_file" -d "$dir_name"; then
                echo "success, deleting zip"
                rm -f "$zip_file"
                ((SUCCESS_COUNT++))
            else
                echo "extraction failed"
                FAILED_FILES+=("$zip_file")
            fi
        fi
    else
        echo "CORRUPTED"
        FAILED_FILES+=("$zip_file")
    fi
done

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Already extracted (zips deleted): $ALREADY_EXTRACTED"
echo "Newly extracted (zips deleted):   $SUCCESS_COUNT"
echo "Failed/Corrupted:                 ${#FAILED_FILES[@]}"
echo ""

if [ ${#FAILED_FILES[@]} -gt 0 ]; then
    echo "=========================================="
    echo "CORRUPTED FILES (need re-download):"
    echo "=========================================="
    for file in "${FAILED_FILES[@]}"; do
        echo "  $file"
    done

    echo ""
    echo "=========================================="
    echo "CURL COMMANDS TO RE-DOWNLOAD:"
    echo "=========================================="
    BASE_URL="https://c3.ndc.nasa.gov/dashlink/static/media/dataset"
    for file in "${FAILED_FILES[@]}"; do
        echo "curl --retry 5 --retry-delay 10 -o \"$file\" \"$BASE_URL/$file\""
    done

    # Also save failed files to a text file
    echo ""
    printf '%s\n' "${FAILED_FILES[@]}" > failed_files.txt
    echo "Failed files list saved to: failed_files.txt"
else
    echo "All zip files verified and extracted successfully!"
fi
