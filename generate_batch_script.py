import os
from pathlib import Path
from datetime import datetime

# Define directories using pathlib
input_dir = Path('input/')
output_dir = Path('output/')
tmp_dir = Path('tmp/')

# Ensure directories exist
for directory in [input_dir, output_dir, tmp_dir]:
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Verified directory: {directory}")

# Configuration variables
vocab_map = '../F5-TTS/data/Emilia_ZH_EN_pinyin/vocab.txt'
ref_audios = 'f5_tts/tests/ref_audio/normalized/2024-10-15-neets-kermit.wav,f5_tts/tests/ref_audio/normalized/2024-10-15-neets-patrick-stewart.wav'

# Get today's date dynamically
today_date = datetime.now().strftime('%Y-%m-%d')

# Create the batch processing script
script_lines = [
    '#!/bin/bash',  # Shebang line to specify the interpreter
    '# Batch processing script for converting log files to mp3',
    'set -e',  # Exit immediately if a command exits with a non-zero status
    '',
    '# Configuration Variables',
    f"VOCAB_MAP='{vocab_map}'",
    f"REF_AUDIOS='{ref_audios}'",
    '',
]

try:
    # Iterate through each .log file in the input directory
    for log_file in input_dir.glob('*.log'):
        story_name = log_file.stem  # Get the story name without extension
        
        # Define paths
        input_path = log_file.resolve()
        tmp_wav = tmp_dir / f"{story_name}.wav"
        output_mp3 = output_dir / f"{today_date}-{story_name}.mp3"
        processed_log = log_file.with_suffix('.log.processed')
        
        # Construct the processing command using configuration variables
        processing_command = (
            f"echo 'Processing {log_file.name}...' && "
            f"python groqcasters.py --voice-model f5_tts '{input_path}' '{tmp_dir}' "
            f"--vocab-char-map \"$VOCAB_MAP\" --ref-audio-paths \"$REF_AUDIOS\" && "
            f"mv '{tmp_dir}/full_podcast.wav' '{tmp_wav}' && "
            f"echo 'Converting {tmp_wav} to MP3...' && "
            f"ffmpeg -i '{tmp_wav}' -vn -ar 44100 -ac 1 -b:a 192k '{output_mp3}' && "
            f"echo 'Generated {output_mp3}' && "
            f"mv '{input_path}' '{processed_log}' && "
            f"echo 'Renamed {log_file.name} to {processed_log.name}'"
        )
        
        # Add the processing command to the script
        script_lines.append(processing_command)
        script_lines.append('')  # Add an empty line for readability

except Exception as e:
    script_lines.append(f'echo "An unexpected error occurred: {e}"')

# Write the polished script to a file
script_path = Path('batch_process_script.sh')
script_path.write_text('\n'.join(script_lines))

# Make the script executable
os.chmod(script_path, 0o755)

print(f"Batch processing script created at: {script_path.resolve()}")
