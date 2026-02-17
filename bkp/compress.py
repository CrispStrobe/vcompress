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

def compress_file(input_file, args):
    directory, filename = os.path.split(input_file)
    name, ext = os.path.splitext(filename)
    output_file = os.path.join(directory, f"{name}_compressed.mp4")
    
    quality_val = str(int(100 - (args.crf * 1.7)))
    
    cmd = [
        'ffmpeg',
        '-hide_banner',            # ADD: Suppress banner
        '-loglevel', 'warning',    # CHANGE: Only show warnings/errors
        '-stats',                  # Keep progress stats
        '-n',
        '-i', input_file,
        '-c:v', 'h264_videotoolbox',
        '-q:v', quality_val,
        '-c:a', 'aac',
        '-b:a', '192k',
        '-movflags', '+faststart',
        output_file
    ]
    
    print(f"\n[>>>] Compressing (GPU): {filename}")
    print(f"      To: {output_file}")
    
    if args.dry_run:
        print(f"      [DRY RUN] Command: {' '.join(cmd)}")
        return
    
    try:
        subprocess.check_call(cmd) 
        print("      ✅ Success.")
    except subprocess.CalledProcessError:
        print("      ❌ Failed.")

def main():
    parser = argparse.ArgumentParser(description="Apple Silicon Video Compressor")
    parser.add_argument("path", nargs="?", default=".")
    parser.add_argument("--threshold", type=float, default=25.0)
    parser.add_argument("--crf", type=int, default=23, help="Rough quality target (Standard=23)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scan_path = os.path.expanduser(args.path)
    files = sorted([f for f in os.listdir(scan_path) if f.lower().endswith(('.mov', '.mp4'))])

    for f in files:
        if f.startswith("._") or "_compressed" in f: continue
        
        filepath = os.path.join(scan_path, f)
        info = get_file_info(filepath)
        if not info: continue

        should_compress = (info['bitrate'] > args.threshold) or ('prores' in info['codec'])
        
        if should_compress:
            print(f"Target: {f[:30]}... ({info['bitrate']:.1f} Mbps) -> ACTION: COMPRESS")
            compress_file(filepath, args)

if __name__ == "__main__":
    main()