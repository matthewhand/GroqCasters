#!/bin/bash

# Directory for processed audio files
INPUT_DIR="f5_tts/tests/ref_audio/"
OUTPUT_DIR="f5_tts/tests/ref_audio/normalized"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Target parameters for normalization
#TARGET_SAMPLE_RATE=24000
TARGET_SAMPLE_RATE=16000
TARGET_CHANNELS=1
#TARGET_DURATION="00:00:20.00"
TARGET_DURATION="00:00:22.00"  # Adjust this as needed

# Function to normalize a WAV file
normalize_audio() {
    local input_file="$1"
    local output_file="$2"
    
    echo "Processing $input_file ..."

    # Step 1: Trim the audio to match the target duration
    ffmpeg -y -i "$input_file" -t "$TARGET_DURATION" -acodec copy "$output_file".trimmed.wav

    # Step 2: Resample to the target sample rate and mono channel
    ffmpeg -y -i "$output_file".trimmed.wav -ar $TARGET_SAMPLE_RATE -ac $TARGET_CHANNELS "$output_file".resampled.wav

    # Step 3: Convert to raw PCM to check for hidden data
    ffmpeg -y -i "$output_file".resampled.wav -f s16le -acodec pcm_s16le "$output_file".pcm

    # Step 4: Convert back to WAV format from PCM
    ffmpeg -y -f s16le -ar $TARGET_SAMPLE_RATE -ac $TARGET_CHANNELS -i "$output_file".pcm "$output_file"

    # Clean up intermediate files
    rm "$output_file".trimmed.wav "$output_file".resampled.wav "$output_file".pcm

    echo "Normalized audio saved to $output_file"
}

# Loop over all the files in the input directory
for input_file in "$INPUT_DIR"/*.wav; do
    # Define output file based on the input filename
    base_filename=$(basename "$input_file")
    output_file="$OUTPUT_DIR/$base_filename"

    # Call the normalization function
    normalize_audio "$input_file" "$output_file"
done

echo "All files have been normalized and saved in $OUTPUT_DIR"
