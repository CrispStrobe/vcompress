import os
import subprocess
import json
import argparse
import sys

def get_file_info(filepath):
    """Returns codec_name and bitrate (Mbps) using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', 
            '-show_entries', 'stream=codec_name,bit_rate,duration', 
            '-of', 'json', filepath
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        if not data.get('streams'): return None
        stream = data['streams'][0]
        
        codec = stream.get('codec_name', 'unknown')
        duration = float(stream.get('duration', 0))
        
        if 'bit_rate' in stream:
            bitrate_mbps = int(stream['bit_rate']) / 1_000_000
        else:
            size_bytes = os.path.getsize(filepath)
            bitrate_mbps = (size_bytes * 8) / (duration * 1_000_000) if duration > 0 else 0

        return {'codec': codec, 'bitrate': bitrate_mbps}
    except:
        return None

def extract_audio(input_file, args):
    """Extracts audio to MP3, explicitly remapping any channel layout to stereo."""
    directory, filename = os.path.split(input_file)
    name, _ = os.path.splitext(filename)
    output_audio = os.path.join(directory, f"{name}_audio.mp3")
    
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'warning',
        '-i', input_file,
        '-vn',
        # Explicitly remap up to 4 channels down to stereo L/R
        '-af', 'pan=stereo|c0=c0+c2|c1=c1+c3',
        '-ar', '44100', '-ac', '2', '-b:a', '192k',
        '-y', output_audio
    ]
    
    if args.dry_run:
        print(f"  [DRY RUN] Command: {' '.join(cmd)}")
        return
    
    print(f"  [>>>] Extracting Audio: {output_audio}")
    try:
        subprocess.check_call(cmd)
        print("      ✅ Success.")
    except subprocess.CalledProcessError:
        print("      ❌ Failed.")

def compress_video(input_file, args):
    """Compresses video using macOS Hardware Acceleration."""
    directory, filename = os.path.split(input_file)
    name, _ = os.path.splitext(filename)
    output_file = os.path.join(directory, f"{name}_compressed.mp4")
    
    # Map CRF-like input to VideoToolbox 0-100 scale
    quality_val = str(int(100 - (args.crf * 1.5)))
    
    # Selection of hardware codec
    codec = f"{args.codec}_videotoolbox"

    cmd = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'warning',
        '-hwaccel', 'auto',
        '-i', input_file,
        '-c:v', codec,
        '-q:v', quality_val,
        '-realtime', 'true',
        # Same stereo downmix for the audio track in the output video
        '-af', 'pan=stereo|c0=c0+c2|c1=c1+c3',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-movflags', '+faststart',
        '-y', output_file
    ]
    
    if args.dry_run:
        print(f"  [DRY RUN] Command: {' '.join(cmd)}")
        return
    
    print(f"  [>>>] Compressing Video ({codec}): {output_file}")
    try:
        subprocess.check_call(cmd) 
        print("      ✅ Success.")
    except subprocess.CalledProcessError:
        print("      ❌ Failed.")

def main():
    parser = argparse.ArgumentParser(description="Apple Silicon Video & Audio Processor")
    parser.add_argument("path", nargs="?", default=".")
    parser.add_argument("--threshold", type=float, default=25.0, help="Bitrate Mbps threshold to trigger compression")
    parser.add_argument("--crf", type=int, default=23, help="Quality (lower is better, default 23)")
    parser.add_argument("--codec", type=str, default="h264", choices=["h264", "hevc"], help="Output codec")
    parser.add_argument("--onlyaudio", action="store_true", help="Only extract audio, skip video compression")
    parser.add_argument("--noaudio", action="store_true", help="Skip audio extraction step")
    parser.add_argument("--dry-run", action="store_true")
    
    args = parser.parse_args()
    scan_path = os.path.expanduser(args.path)
    
    files = sorted([f for f in os.listdir(scan_path) if f.lower().endswith(('.mov', '.mp4'))])

    for f in files:
        if f.startswith("._") or "_compressed" in f or "_audio" in f: continue
        
        filepath = os.path.join(scan_path, f)
        info = get_file_info(filepath)
        if not info: continue

        should_process = (info['bitrate'] > args.threshold) or ('prores' in info['codec'])
        
        if should_process:
            print(f"\nTarget: {f} ({info['bitrate']:.1f} Mbps)")
            
            if not args.noaudio:
                extract_audio(filepath, args)
            
            if not args.onlyaudio:
                compress_video(filepath, args)

if __name__ == "__main__":
    main()