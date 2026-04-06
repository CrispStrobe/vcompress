#!/usr/bin/env python3
"""Convert video/audio files with optional parallel hardware acceleration, multiplexing, and live per-file progress."""

import argparse
import logging
import os
import signal
import sys
import subprocess
import concurrent.futures
import threading
import time
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Tuple, Dict

# Enable ANSI escape sequences for Windows 10+
if sys.platform == "win32":
    os.system("")

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Global registry and concurrency locks
active_processes: Set[subprocess.Popen] = set()
process_lock = threading.Lock()
shutdown_event = threading.Event()

# Regex for parsing FFmpeg output
duration_regex = re.compile(r"Duration:\s*(\d{2,}):(\d{2}):(\d+(?:\.\d+)?)")
time_regex = re.compile(r"time=\s*(\d{2,}):(\d{2}):(\d+(?:\.\d+)?)")

def parse_ffmpeg_time(h: str, m: str, s: str) -> float:
    """Convert hours, minutes, and seconds from ffmpeg log to total seconds."""
    return float(h) * 3600 + float(m) * 60 + float(s)

class ProgressTracker:
    """Manages a dynamic terminal UI to show real-time, per-file progress percentages."""
    def __init__(self, total: int, verbose: bool):
        self.total = total
        self.completed = 0
        self.verbose = verbose
        self.active_jobs = {}
        self.lock = threading.Lock()
        self.last_lines = 0
        self.stop_event = threading.Event()

        if not self.verbose:
            self.ui_thread = threading.Thread(target=self._ui_loop, daemon=True)
            self.ui_thread.start()

    def set_job_status(self, filename: str, status: str):
        if self.verbose:
            return
        with self.lock:
            self.active_jobs[filename] = status

    def complete_job(self, filename: str, success: bool, msg: str = ""):
        with self.lock:
            self.completed += 1
            if filename in self.active_jobs:
                del self.active_jobs[filename]

            if not self.verbose:
                self._clear_ui()
                status_str = "\033[32mSUCCESS\033[0m" if success else "\033[31mFAILED\033[0m"
                print(f"[{self.completed}/{self.total}] {filename}: {status_str} {msg}")

    def log_message(self, message: str):
        """Safely print a log message above the live UI."""
        if self.verbose:
            logger.info(message)
            return
        with self.lock:
            self._clear_ui()
            print(message)

    def _clear_ui(self):
        """Clear the currently printed UI lines from the terminal."""
        if self.last_lines > 0:
            sys.stdout.write(f"\033[{self.last_lines}A")  # Move up
            sys.stdout.write("\033[J")                    # Clear from cursor to bottom
            sys.stdout.flush()
            self.last_lines = 0

    def _ui_loop(self):
        """Background thread that continually redraws the active jobs list."""
        while not self.stop_event.is_set():
            with self.lock:
                if not self.active_jobs:
                    time.sleep(0.1)
                    continue

                self._clear_ui()

                output = []
                for f, stat in list(self.active_jobs.items()):
                    fname = f if len(f) <= 35 else f[:32] + "..."
                    output.append(f" \033[36m->\033[0m {fname:<35} | {stat}")

                sys.stdout.write("\n".join(output) + "\n")
                sys.stdout.flush()
                self.last_lines = len(output)
            time.sleep(0.25)

    def stop(self):
        self.stop_event.set()
        if not self.verbose:
            self.ui_thread.join()
            with self.lock:
                self._clear_ui()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert video/audio files with parallel HW acceleration.")
    parser.add_argument("path", nargs="?", default=".", help="Path to process.")
    parser.add_argument("--globs", nargs="+", default=["*.mp4", "*.mkv", "*.m4a", "*.dashVideo", "*.dashAudio", "*.webm"])
    parser.add_argument("--date", default="today", help="Filter by date (YYYY-MM-DD, 'today', or 'all'). Default: today.")
    parser.add_argument("--suffix", default="_conv", help="Suffix for output filename (default: _conv).")
    parser.add_argument("-r", action="store_true", default=True, help="Recursive search.")
    parser.add_argument("--output-format", default="mp4", help="Output format (default: mp4).")
    parser.add_argument("--resolution", default="720p", help="Output resolution height (e.g., 720p).")
    parser.add_argument("--audio-bitrate", default="192k", help="Audio bitrate (default: 192k).")
    parser.add_argument("-j", "--jobs", type=int, default=min(4, os.cpu_count() or 1), help="Parallel jobs.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done.")
    parser.add_argument("--verbose", action="store_true", help="Enable direct raw ffmpeg output and disable UI.")
    return parser.parse_args()

def discover_files(path: str, globs: List[str], recursive: bool, date_filter: str, suffix: str, output_format: str) -> List[Tuple[Path, Optional[Path]]]:
    """Discover files, filtering by date, and pair .dashVideo with .dashAudio if they exist."""
    path_obj = Path(path).resolve()
    if not path_obj.exists():
        logger.error("Path does not exist: %s", path)
        return []

    target_date: Optional[datetime.date] = None
    if date_filter == "today":
        target_date = datetime.now().date()
    elif date_filter != "all":
        try:
            target_date = datetime.strptime(date_filter, "%Y-%m-%d").date()
        except ValueError:
            logger.warning("Invalid date format: %s. Using 'all'.", date_filter)

    seen = set()
    raw_files = []

    for pattern in globs:
        iterator = path_obj.rglob(pattern) if recursive else path_obj.glob(pattern)
        for f in iterator:
            if not f.is_file() or f in seen:
                continue
            seen.add(f)

            # Skip already converted files
            if f.stem.endswith(suffix) and f.suffix[1:].lower() == output_format.lower():
                continue

            # Date filter
            if target_date:
                f_mtime = datetime.fromtimestamp(f.stat().st_mtime).date()
                if f_mtime != target_date:
                    continue

            raw_files.append(f)

    # Group files by their base name (without extension)
    grouped_files: Dict[Path, Dict[str, Path]] = {}
    for f in raw_files:
        base = f.with_suffix('')
        ext = f.suffix.lower()
        if base not in grouped_files:
            grouped_files[base] = {}
        grouped_files[base][ext] = f

    tasks: List[Tuple[Path, Optional[Path]]] = []

    for base, exts in grouped_files.items():
        # Multiplexing logic for DASH formats
        if '.dashvideo' in exts:
            vid = exts.pop('.dashvideo')
            aud = exts.pop('.dashaudio', None)
            tasks.append((vid, aud))
        elif '.dashaudio' in exts:
            # Standalone dashAudio without a video pair
            tasks.append((exts.pop('.dashaudio'), None))

        # Add remaining standalone files
        for ext, f in exts.items():
            tasks.append((f, None))

    # Sort alphabetically by the primary input path
    return sorted(tasks, key=lambda t: t[0])

def get_video_codecs() -> List[str]:
    """Cascading fallback list of codecs based on OS."""
    if sys.platform == "darwin":
        return ["h264_videotoolbox", "libx264"]
    elif sys.platform == "win32":
        return ["h264_nvenc", "h264_amf", "h264_qsv", "libx264"]
    else:
        return ["h264_nvenc", "h264_vaapi", "h264_qsv", "libx264"]

def terminate_process_group(p: subprocess.Popen):
    """Safely terminate a subprocess and all its children across OS platforms."""
    try:
        if sys.platform == "win32":
            p.send_signal(signal.CTRL_BREAK_EVENT)
            p.kill()
        else:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        pass

def convert_one(
    input_video: Path, input_audio: Optional[Path], output_path: Path, resolution: str,
    audio_bitrate: str, output_format: str, verbose: bool, tracker: ProgressTracker
) -> bool:
    """Worker function that handles individual/multiplexed file conversions and streams progress."""
    if shutdown_event.is_set():
        return False

    height = resolution.lower().rstrip('p')
    height = height if height.isdigit() else "720"

    # Check if primary input is actually just an audio file (e.g., standard .m4a or standalone .dashaudio)
    is_audio_only = (output_format.lower() == "m4a") or (input_video.suffix.lower() in [".m4a", ".mp3", ".aac", ".dashaudio"])
    codecs_to_try = [None] if is_audio_only else get_video_codecs()

    display_name = f"{input_video.name} (+audio)" if input_audio else input_video.name

    success = False
    for codec in codecs_to_try:
        if shutdown_event.is_set():
            break

        tracker.set_job_status(display_name, f"Initializing ({codec or 'audio'})...")

        # Build FFmpeg command with multiplexing capability
        cmd = [
            "ffmpeg", "-y", "-i", str(input_video)
        ]

        # Inject secondary audio input if paired
        if input_audio:
            cmd.extend(["-i", str(input_audio)])

        cmd.extend([
            "-c:a", "aac", "-b:a", audio_bitrate,
            "-threads", "1",
            "-loglevel", "debug" if verbose else "info"
        ])

        # Explicitly map streams if we have two inputs to prevent FFmpeg dropping them
        if input_audio:
            cmd.extend(["-map", "0:v:0", "-map", "1:a:0"])

        if is_audio_only:
            cmd.append("-vn")
        else:
            cmd.extend(["-vf", f"scale=-2:{height}", "-c:v", codec, "-b:v", "2M"])

        cmd.append(str(output_path))

        kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,  # Route stderr to stdout so we can stream it
            "text": True,
            "bufsize": 1,
            "errors": "replace"
        }

        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True

        try:
            process = subprocess.Popen(cmd, **kwargs)
            with process_lock:
                active_processes.add(process)

            duration = 0.0
            last_lines = []

            # Stream stdout line-by-line
            for line in process.stdout:
                if shutdown_event.is_set():
                    break

                if verbose:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    continue

                if len(last_lines) >= 20:
                    last_lines.pop(0)
                last_lines.append(line.strip())

                # Grab the first valid duration (usually Input 0 / the video length)
                if duration == 0.0:
                    m = duration_regex.search(line)
                    if m:
                        duration = parse_ffmpeg_time(m.group(1), m.group(2), m.group(3))

                m = time_regex.search(line)
                if m and duration > 0:
                    current_time = parse_ffmpeg_time(m.group(1), m.group(2), m.group(3))
                    pct = min(100.0, (current_time / duration) * 100)
                    tracker.set_job_status(display_name, f"[\033[33m{pct:>5.1f}%\033[0m] using {codec or 'audio'}")

            process.wait()
            retcode = process.returncode

            with process_lock:
                active_processes.discard(process)

            if retcode == 0 and not shutdown_event.is_set():
                success = True
                tracker.complete_job(display_name, True)
                break
            elif not shutdown_event.is_set():
                if codec == "libx264" or is_audio_only:
                    tracker.complete_job(display_name, False, msg="\n" + "\n".join(last_lines[-5:]))
                    break
                else:
                    tracker.log_message(f"Codec '{codec}' failed for {display_name}. Falling back...")

        except Exception as e:
            tracker.complete_job(display_name, False, msg=str(e))
            break

    return success

def main() -> None:
    args = parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("ffmpeg must be installed and in your PATH.")
        sys.exit(1)

    # Note: files is now a list of Tuples containing (video_path, optional_audio_path)
    files = discover_files(args.path, args.globs, args.r, args.date, args.suffix, args.output_format)

    if not files:
        logger.info("No files found matching the criteria.")
        return

    logger.info("Found %d tasks to process. Using up to %d concurrent jobs.", len(files), args.jobs)

    tasks = []
    for vid_path, aud_path in files:
        output_name = f"{vid_path.stem}{args.suffix}.{args.output_format}"
        output_path = vid_path.with_name(output_name)
        tasks.append((vid_path, aud_path, output_path))

    if args.dry_run:
        for vid_p, aud_p, out_p in tasks:
            if aud_p:
                logger.info("[Dry Run] %s + %s -> %s", vid_p.name, aud_p.name, out_p.name)
            else:
                logger.info("[Dry Run] %s -> %s", vid_p.name, out_p.name)
        return

    if not args.verbose:
        logger.setLevel(logging.WARNING)

    tracker = ProgressTracker(len(files), args.verbose)
    success_count = 0
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs)

    try:
        futures = [
            executor.submit(
                convert_one, vid, aud, outp, args.resolution, args.audio_bitrate,
                args.output_format, args.verbose, tracker
            ) for vid, aud, outp in tasks
        ]

        for future in concurrent.futures.as_completed(futures):
            if future.result():
                success_count += 1

    except KeyboardInterrupt:
        shutdown_event.set()
        tracker.stop()

        print("\n\033[31m[!] Interrupted by user! Initiating graceful shutdown...\033[0m")

        if sys.version_info >= (3, 9):
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=False)

        with process_lock:
            for p in active_processes:
                if p.poll() is None:
                    terminate_process_group(p)

        print(f"Killed {len(active_processes)} active conversion processes.")
        sys.exit(130)

    finally:
        tracker.stop()

    print(f"\n✅ Operation Complete. Successfully converted {success_count}/{len(files)} tasks.")

if __name__ == "__main__":
    main()
