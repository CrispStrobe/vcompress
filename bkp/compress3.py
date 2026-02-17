#!/usr/bin/env python3
"""
compress3.py — Cross-Platform Chunked Video Compressor v3
Hardware acceleration on Apple Silicon, NVIDIA, Intel, AMD.
Chunked encoding with resume, live progress, audio extraction.
"""

import os
import sys
import subprocess
import json
import argparse
import time
import re
import platform
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# ─── ANSI Colors ───────────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    RED    = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
    BLUE   = "\033[94m"; CYAN  = "\033[96m"; WHITE  = "\033[97m"
    GREY   = "\033[90m"

def cprint(color, msg): print(f"{color}{msg}{C.RESET}")
def log(msg, verbose):
    if verbose: print(f"{C.GREY}  [DBG] {msg}{C.RESET}")

# ─── Platform / Encoder Detection ─────────────────────────────────────────────
def detect_encoders(preferred_codec: str, verbose: bool) -> dict:
    """
    Probe ffmpeg for available hardware encoders.
    Returns a dict with 'video_h264', 'video_hevc', 'label'.
    Falls back gracefully: hw-accel → libx264/libx265 → error.
    """
    system = platform.system()
    log(f"Platform: {system} ({platform.machine()})", verbose)

    # Check ffmpeg availability first
    if not shutil.which('ffmpeg'):
        return None

    # Fetch all available encoders from ffmpeg
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=10
        )
        available = result.stdout + result.stderr
    except Exception:
        available = ""

    def has_enc(name): return f" {name} " in available or f"\n {name} " in available

    # Priority chains per platform
    if system == "Darwin":
        h264_chain = ['h264_videotoolbox', 'libx264']
        hevc_chain = ['hevc_videotoolbox', 'libx265']
        label      = "Apple VideoToolbox"
    elif system == "Windows":
        h264_chain = ['h264_nvenc', 'h264_amf', 'h264_qsv', 'libx264']
        hevc_chain = ['hevc_nvenc', 'hevc_amf', 'hevc_qsv', 'libx265']
        label      = "Windows HW"
    else:  # Linux
        h264_chain = ['h264_nvenc', 'h264_vaapi', 'h264_qsv', 'libx264']
        hevc_chain = ['hevc_nvenc', 'hevc_vaapi', 'hevc_qsv', 'libx265']
        label      = "Linux HW"

    def pick(chain):
        for enc in chain:
            if has_enc(enc):
                log(f"  Selected encoder: {enc}", verbose)
                return enc
        return None

    h264 = pick(h264_chain)
    hevc = pick(hevc_chain)

    if not h264 and not hevc:
        return None

    chosen_video = h264 if preferred_codec == 'h264' else (hevc or h264)

    return {
        'video':       chosen_video,
        'video_h264':  h264,
        'video_hevc':  hevc,
        'label':       label,
        'is_hw':       not (chosen_video or '').startswith('lib'),
        'is_vt':       'videotoolbox' in (chosen_video or ''),
        'is_nvenc':    'nvenc' in (chosen_video or ''),
        'is_software': (chosen_video or '').startswith('lib'),
    }

def build_quality_flags(enc: str, crf: int) -> list:
    """Returns encoder-specific quality flags."""
    if 'videotoolbox' in enc:
        q = str(int(100 - (crf * 1.5)))
        return ['-q:v', q]
    elif 'nvenc' in enc or 'amf' in enc:
        return ['-rc', 'vbr', '-cq', str(crf)]
    elif 'qsv' in enc:
        return ['-global_quality', str(crf)]
    elif 'vaapi' in enc:
        return ['-compression_level', str(crf)]
    else:  # libx264 / libx265
        return ['-crf', str(crf)]

def build_hwaccel_flags(enc: str) -> list:
    """Returns input-side hardware decode flags where applicable."""
    if 'videotoolbox' in enc:
        return ['-hwaccel', 'videotoolbox']
    elif 'nvenc' in enc:
        return ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda']
    elif 'qsv' in enc:
        return ['-hwaccel', 'qsv']
    elif 'vaapi' in enc:
        return ['-hwaccel', 'vaapi', '-vaapi_device', '/dev/dri/renderD128']
    return []

# ─── File Info ─────────────────────────────────────────────────────────────────
@dataclass
class FileInfo:
    path: str
    codec: str
    bitrate_mbps: float
    duration_sec: float
    width: int
    height: int
    fps: float
    size_mb: float
    has_audio: bool
    audio_codec: str
    audio_bitrate_kbps: float
    container: str
    color_space: str
    is_hdr: bool

    @property
    def duration_str(self):
        h, rem = divmod(int(self.duration_sec), 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @property
    def is_prores(self): return 'prores' in self.codec.lower()
    @property
    def is_raw(self): return self.codec.lower() in ('dnxhd', 'dnxhr', 'cineform', 'v210')

    @property
    def compression_label(self):
        if self.bitrate_mbps > 100 or self.is_prores or self.is_raw:
            return f"{C.RED}UNCOMPRESSED/RAW{C.RESET}"
        elif self.bitrate_mbps > 25:
            return f"{C.YELLOW}HIGH BITRATE{C.RESET}"
        elif self.bitrate_mbps > 8:
            return f"{C.CYAN}MEDIUM{C.RESET}"
        else:
            return f"{C.GREEN}COMPRESSED{C.RESET}"

def probe_file(filepath: str, verbose: bool) -> Optional[FileInfo]:
    log(f"Probing: {filepath}", verbose)
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', filepath],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            log(f"ffprobe stderr: {result.stderr.strip()}", verbose)
            return None

        data    = json.loads(result.stdout)
        streams = data.get('streams', [])
        fmt     = data.get('format', {})

        vstream = next((s for s in streams if s.get('codec_type') == 'video'), None)
        astream = next((s for s in streams if s.get('codec_type') == 'audio'), None)
        if not vstream:
            log("No video stream.", verbose); return None

        codec       = vstream.get('codec_name', 'unknown').lower()
        width       = int(vstream.get('width', 0))
        height      = int(vstream.get('height', 0))
        color_space = vstream.get('color_space', 'unknown')
        color_trc   = vstream.get('color_transfer', '')
        is_hdr      = any(x in color_trc.lower() for x in ('smpte2084', 'arib-std-b67', 'bt2020'))

        fps = 0.0
        try:
            n, d = map(int, vstream.get('r_frame_rate', '0/1').split('/'))
            fps = n / d if d else 0.0
        except Exception: pass

        duration = 0.0
        for src in [vstream, astream or {}, fmt]:
            if src and 'duration' in src:
                try: duration = float(src['duration']); break
                except (ValueError, TypeError): pass

        bitrate_mbps = 0.0
        for src, key in [(vstream, 'bit_rate'), (fmt, 'bit_rate')]:
            if src and src.get(key) not in (None, 'N/A', ''):
                try: bitrate_mbps = int(src[key]) / 1_000_000; break
                except (ValueError, TypeError): pass
        if bitrate_mbps == 0 and duration > 0:
            bitrate_mbps = (os.path.getsize(filepath) * 8) / (duration * 1_000_000)

        log(f"codec={codec} res={width}x{height} fps={fps:.2f} br={bitrate_mbps:.2f}Mbps dur={duration:.1f}s hdr={is_hdr}", verbose)

        has_audio = astream is not None
        audio_codec = (astream or {}).get('codec_name', 'none')
        try:
            audio_br_raw = (astream or {}).get('bit_rate', '0')
            audio_bitrate_kbps = int(audio_br_raw) / 1000 if audio_br_raw not in ('N/A', '', None) else 0.0
        except (ValueError, TypeError):
            audio_bitrate_kbps = 0.0

        return FileInfo(
            path=filepath, codec=codec, bitrate_mbps=bitrate_mbps,
            duration_sec=duration, width=width, height=height, fps=fps,
            size_mb=os.path.getsize(filepath) / (1024 * 1024),
            has_audio=has_audio, audio_codec=audio_codec,
            audio_bitrate_kbps=audio_bitrate_kbps,
            container=Path(filepath).suffix.lower().lstrip('.'),
            color_space=color_space, is_hdr=is_hdr
        )
    except subprocess.TimeoutExpired:
        print(f"{C.RED}  [!] ffprobe timed out{C.RESET}"); return None
    except Exception as e:
        print(f"{C.RED}  [!] probe error: {e}{C.RESET}"); return None

def should_compress(info: FileInfo, threshold: float, verbose: bool) -> tuple:
    reasons = []
    if info.is_prores:            reasons.append(f"ProRes codec ({info.codec})")
    if info.is_raw:               reasons.append(f"Raw codec ({info.codec})")
    if info.bitrate_mbps > threshold: reasons.append(f"Bitrate {info.bitrate_mbps:.1f} Mbps > {threshold} Mbps")
    if info.size_mb > 500 and info.bitrate_mbps > 10: reasons.append(f"Large file {info.size_mb:.0f} MB")
    if reasons:
        log(f"Compress: {'; '.join(reasons)}", verbose)
        return True, "; ".join(reasons)
    log(f"Skip: {info.bitrate_mbps:.1f} Mbps, codec={info.codec}", verbose)
    return False, "within acceptable range"

# ─── Progress Bar ──────────────────────────────────────────────────────────────
# ffmpeg writes stats to stderr as a single line refreshed with \r (not \n).
# We must read character-by-character or use a line buffer that splits on \r too.

_STATS_RE = re.compile(
    r'frame=\s*(\d+).*?fps=\s*([\d.]+).*?time=\s*([\d:.]+).*?bitrate=\s*(\S+).*?speed=\s*([\d.]+)x',
    re.DOTALL
)
_TIME_RE = re.compile(r'time=\s*([\d:.]+)')

def _hms_to_sec(t: str) -> float:
    t = t.strip()
    try:
        parts = t.split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return float(t)
    except Exception:
        return 0.0

def _render_bar(current_sec: float, total_sec: float, fps: float, speed: float,
                bitrate: str, chunk_label: str):
    bar_width = 38
    pct    = min(current_sec / total_sec, 1.0) if total_sec > 0 else 0.0
    filled = int(bar_width * pct)
    bar    = "█" * filled + "░" * (bar_width - filled)

    eta_str = ""
    if speed > 0.01 and 0 < pct < 1.0:
        remaining = (total_sec - current_sec) / speed
        m, s = divmod(int(remaining), 60)
        eta_str = f" ETA {m:02d}:{s:02d}"

    parts = [
        f"\r  {C.CYAN}{chunk_label}[{bar}]{C.RESET}",
        f" {C.BOLD}{pct*100:5.1f}%{C.RESET}",
        f"{C.GREY}",
        f" {fps:.0f}fps"   if fps > 0     else "",
        f" {speed:.2f}x"   if speed > 0   else "",
        f" {bitrate}"      if bitrate      else "",
        eta_str,
        f"{C.RESET}  ",
    ]
    sys.stdout.write("".join(parts))
    sys.stdout.flush()

def run_ffmpeg_progress(cmd: list, total_sec: float,
                        chunk_label: str = "", verbose: bool = False) -> bool:
    """
    Run ffmpeg and parse its stderr for live progress.

    ffmpeg writes stats as \r-terminated lines (overwriting the same terminal
    line). We read stderr in binary mode, splitting on both \r and \n so we
    capture every update without waiting for a newline.
    """
    if verbose:
        print(f"\n  {C.GREY}[DBG] {' '.join(cmd)}{C.RESET}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,  # binary; we decode manually
        )
    except FileNotFoundError:
        print(f"{C.RED}  [!] ffmpeg not found{C.RESET}")
        return False

    current_sec = 0.0
    fps         = 0.0
    speed       = 0.0
    bitrate     = ""
    buf         = b""

    # Read stderr byte-by-byte accumulating into buf, flushing on \r or \n.
    # This is the only reliable way to get ffmpeg's \r-refreshed stat lines.
    while True:
        chunk = proc.stderr.read(256)
        if not chunk:
            break
        buf += chunk

        # Split on both \r and \n
        while b'\r' in buf or b'\n' in buf:
            for sep in (b'\r', b'\n'):
                idx = buf.find(sep)
                if idx == -1:
                    continue
                line_b = buf[:idx]
                buf    = buf[idx + 1:]
                line   = line_b.decode('utf-8', errors='replace').strip()
                if not line:
                    continue

                # Classify line
                is_stats   = ('time=' in line and 'frame=' in line)
                is_problem = any(x in line.lower() for x in (
                    'error', 'warning', 'invalid', 'failed', 'overread',
                    'guessed', 'no filtered', 'nothing was encoded'))

                # Always show warnings/errors; show everything in verbose
                if is_problem or (verbose and not is_stats):
                    color = C.RED if 'error' in line.lower() or 'failed' in line.lower() \
                            else C.YELLOW if is_problem else C.GREY
                    sys.stdout.write(f"\n  {color}[ffmpeg] {line}{C.RESET}")
                    sys.stdout.flush()
                    if not is_stats:
                        continue

                # Parse stats line
                m = _STATS_RE.search(line)
                if m:
                    try: fps   = float(m.group(2))
                    except ValueError: pass
                    try: current_sec = _hms_to_sec(m.group(3))
                    except Exception: pass
                    bitrate = m.group(4)
                    try: speed = float(m.group(5))
                    except ValueError: pass
                else:
                    # Partial match — at least grab time=
                    tm = _TIME_RE.search(line)
                    if tm:
                        try: current_sec = _hms_to_sec(tm.group(1))
                        except Exception: pass

                if current_sec > 0:
                    _render_bar(current_sec, total_sec, fps, speed, bitrate, chunk_label)
                break  # restart the while loop with updated buf

    proc.wait()
    # Draw 100% on completion
    if total_sec > 0:
        _render_bar(total_sec, total_sec, fps, speed, bitrate, chunk_label)
    sys.stdout.write("\n")

    if proc.returncode not in (0, None):
        print(f"  {C.RED}[✗] ffmpeg exited {proc.returncode}{C.RESET}")
        return False
    return True

# ─── Audio Extraction ──────────────────────────────────────────────────────────
def probe_audio_stream(filepath: str, verbose: bool) -> dict:
    """Full audio stream probe including codec_tag which reveals in24/in32 Atomos issues."""
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
             '-show_entries', 'stream=channels,sample_rate,codec_name,channel_layout,start_time,codec_tag_string',
             '-of', 'json', filepath],
            capture_output=True, text=True, timeout=15
        )
        streams = json.loads(r.stdout).get('streams', [])
        if not streams:
            return {}
        s = streams[0]
        layout     = s.get('channel_layout', '')
        codec_tag  = s.get('codec_tag_string', '').lower().strip()
        import re
        is_guessed = bool(re.search(r'\d+[.]\d+', layout)) or layout in ('', 'unknown')
        # in24/in32 = Atomos/AJA packed PCM — ffmpeg 8.x cannot decode these
        is_in24    = codec_tag in ('in24', 'in32', 'in16')
        result = {
            'channels':    int(s.get('channels', 2)),
            'sample_rate': int(s.get('sample_rate', 48000)),
            'codec_name':  s.get('codec_name', 'unknown'),
            'codec_tag':   codec_tag,
            'layout':      layout,
            'is_guessed':  is_guessed,
            'is_in24':     is_in24,
            'start_time':  s.get('start_time', '0'),
        }
        log(f"Audio stream: {result}", verbose)
        return result
    except Exception as e:
        log(f"Audio probe failed: {e}", verbose)
        return {}


def afconvert_available() -> bool:
    return shutil.which('afconvert') is not None


def extract_audio_afconvert(filepath: str, out: str, verbose: bool, dry_run: bool) -> bool:
    """
    macOS CoreAudio path via afconvert.

    afconvert uses Apple's native CoreAudio stack and handles every MOV audio
    codec that QuickTime can play — including Atomos in24/in32 packed PCM that
    ffmpeg 8.x cannot decode.

    Two steps:
      1. afconvert MOV -> temp WAV (CoreAudio decodes in24 natively)
      2. ffmpeg WAV -> MP3  (ffmpeg handles plain WAV perfectly)

    The temp WAV is deleted on success or failure.
    """
    tmp_wav = out.replace('.mp3', '_tmp_af.wav')

    # Step 1: afconvert
    cmd1 = [
        'afconvert',
        '-f', 'WAVE',        # output format: WAV
        '-d', 'LEI16@44100', # 16-bit little-endian, 44100 Hz
        '-c', '2',           # 2 channels (stereo downmix)
        filepath,
        tmp_wav
    ]
    log(f"afconvert cmd: {' '.join(cmd1)}", verbose)

    if dry_run:
        print(f"  {C.DIM}[DRY/afconvert] {' '.join(cmd1)}{C.RESET}")
        return True

    r1 = subprocess.run(cmd1, capture_output=True, text=True)
    if r1.stderr.strip():
        for line in r1.stderr.strip().splitlines():
            print(f"  {C.GREY}[afconvert] {line}{C.RESET}")

    if r1.returncode != 0 or not os.path.exists(tmp_wav) or os.path.getsize(tmp_wav) < 1000:
        log(f"afconvert failed (exit={r1.returncode})", verbose)
        if os.path.exists(tmp_wav): os.remove(tmp_wav)
        return False

    wav_mb = os.path.getsize(tmp_wav) / 1024 / 1024
    log(f"afconvert WAV: {wav_mb:.0f} MB", verbose)

    # Step 2: ffmpeg WAV -> MP3
    cmd2 = [
        'ffmpeg', '-hide_banner', '-loglevel', 'warning',
        '-i', tmp_wav,
        '-b:a', '192k',
        '-y', out
    ]
    log(f"WAV->MP3 cmd: {' '.join(cmd2)}", verbose)

    ok2 = run_ffmpeg_progress(cmd2, 0.0, chunk_label="mp3   ", verbose=verbose)

    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)
        log("Temp WAV deleted.", verbose)

    return ok2 and os.path.exists(out) and os.path.getsize(out) > 10_000


def extract_audio_ffmpeg(filepath: str, out: str, duration: float,
                         verbose: bool, dry_run: bool) -> bool:
    """
    Standard ffmpeg audio extraction with fallback strategy chain.
    Works for all files except Atomos in24/in32 on ffmpeg 8.x.
    """
    strategies = [
        {
            'label': 'standard',
            'cmd': ['ffmpeg', '-hide_banner', '-loglevel', 'warning',
                    '-i', filepath, '-vn', '-ac', '2', '-ar', '44100',
                    '-b:a', '192k', '-y', out],
        },
        {
            'label': 'no-guess',
            'cmd': ['ffmpeg', '-hide_banner', '-loglevel', 'warning',
                    '-guess_layout_max', '0',
                    '-i', filepath, '-vn', '-ac', '2', '-ar', '44100',
                    '-b:a', '192k', '-y', out],
        },
        {
            'label': 'explicit-map',
            'cmd': ['ffmpeg', '-hide_banner', '-loglevel', 'warning',
                    '-guess_layout_max', '0',
                    '-i', filepath, '-map', '0:a:0',
                    '-ac', '2', '-ar', '44100', '-b:a', '192k', '-y', out],
        },
    ]

    for strat in strategies:
        label = strat['label']
        if os.path.exists(out) and os.path.getsize(out) < 10_000:
            os.remove(out)

        if dry_run:
            print(f"  {C.DIM}[DRY:{label}] {' '.join(strat['cmd'])}{C.RESET}")
            return True

        log(f"Trying ffmpeg strategy: {label}", verbose)
        log(f"cmd: {' '.join(strat['cmd'])}", verbose)

        ok = run_ffmpeg_progress(strat['cmd'], duration,
                                 chunk_label=f"{label[:5]:5} ", verbose=verbose)
        sz = os.path.getsize(out) if os.path.exists(out) else 0
        if ok and sz > 10_000:
            log(f"Strategy '{label}' succeeded ({sz} bytes)", verbose)
            return True
        log(f"Strategy '{label}' produced {sz}b, trying next", verbose)

    return False


def extract_audio(info: FileInfo, args):
    """
    Extract audio to stereo MP3.

    Routing logic:
      - macOS + in24/in32 codec tag (Atomos/AJA recorders) + afconvert available
        → use afconvert (CoreAudio path, handles in24 natively)
      - everything else
        → ffmpeg strategy chain (standard → no-guess → explicit-map)
    """
    out = str(Path(info.path).with_name(Path(info.path).stem + "_audio.mp3"))

    if not info.has_audio:
        print(f"  {C.YELLOW}[!] No audio stream, skipping.{C.RESET}"); return
    if os.path.exists(out) and not args.overwrite:
        print(f"  {C.GREY}[~] Audio exists: {Path(out).name} (--overwrite to redo){C.RESET}"); return

    if os.path.exists(out):
        if os.path.getsize(out) < 10_000:
            os.remove(out)
        elif args.overwrite:
            os.remove(out)

    ainfo    = probe_audio_stream(info.path, args.verbose)
    ch       = ainfo.get('channels', 2)
    is_in24  = ainfo.get('is_in24', False)
    ch_note  = f"  ({ch}ch -> stereo)" if ch > 2 else ""

    # Routing decision
    use_afconvert = (
        platform.system() == 'Darwin'
        and is_in24
        and afconvert_available()
    )

    if is_in24:
        log(f"Atomos in24/in32 codec detected — ffmpeg 8.x cannot decode this", args.verbose)
    if use_afconvert:
        log(f"Routing to afconvert (CoreAudio) path", args.verbose)
        print(f"  {C.BLUE}[♫] Extracting audio (CoreAudio) -> {Path(out).name}{ch_note}{C.RESET}")
    else:
        print(f"  {C.BLUE}[♫] Extracting audio -> {Path(out).name}{ch_note}{C.RESET}")

    if use_afconvert:
        ok = extract_audio_afconvert(info.path, out, args.verbose, args.dry_run)
    else:
        ok = extract_audio_ffmpeg(info.path, out, info.duration_sec,
                                  args.verbose, args.dry_run)

    # If primary path failed and we're on macOS, try afconvert as last resort
    if not ok and platform.system() == 'Darwin' and afconvert_available() and not use_afconvert:
        log("ffmpeg failed, falling back to afconvert", args.verbose)
        print(f"  {C.YELLOW}[!] ffmpeg failed, trying CoreAudio fallback...{C.RESET}")
        ok = extract_audio_afconvert(info.path, out, args.verbose, args.dry_run)

    out_size = os.path.getsize(out) if os.path.exists(out) else 0
    if ok and out_size > 10_000:
        print(f"  {C.GREEN}[✓] Audio saved ({out_size / 1024 / 1024:.1f} MB){C.RESET}")
    else:
        print(f"  {C.RED}[✗] All audio strategies failed (size={out_size}b).{C.RESET}")
        if not args.verbose:
            print(f"  {C.YELLOW}    Re-run with --verbose for details.{C.RESET}")
        if platform.system() == 'Darwin':
            print(f"  {C.YELLOW}    Manual: afconvert -f WAVE -d LEI16@44100 -c 2 \"{info.path}\" /tmp/audio.wav{C.RESET}")


# ─── Chunked Video Compression ─────────────────────────────────────────────────
def compress_video_chunked(info: FileInfo, args, encoders: dict):
    stem       = Path(info.path).stem
    directory  = Path(info.path).parent
    final_out  = directory / f"{stem}_compressed.mp4"
    chunks_dir = directory / f"{stem}_chunks"

    if final_out.exists() and not args.overwrite:
        print(f"  {C.GREY}[~] Output exists: {final_out.name} (--overwrite to redo){C.RESET}")
        return

    chunks_dir.mkdir(exist_ok=True)

    enc        = encoders['video']
    chunk_sec  = args.chunk_minutes * 60
    total_sec  = info.duration_sec
    n_chunks   = max(1, -(-int(total_sec) // int(chunk_sec)))  # ceiling div

    quality_flags = build_quality_flags(enc, args.crf)
    hwaccel_flags = build_hwaccel_flags(enc)

    hdr_flags = []
    if info.is_hdr:
        print(f"  {C.YELLOW}[!] HDR source — preserving BT.2020/SMPTE2084.{C.RESET}")
        hdr_flags = ['-color_primaries', 'bt2020',
                     '-color_trc',       'smpte2084',
                     '-colorspace',      'bt2020nc']

    q_display = quality_flags[1] if len(quality_flags) >= 2 else '?'
    print(f"  {C.CYAN}[▶] {n_chunks} chunk(s) × {args.chunk_minutes} min  "
          f"encoder={enc}  quality={quality_flags[0]}={q_display}{C.RESET}")

    chunk_paths = []
    total_start = time.time()

    for i in range(n_chunks):
        start_sec  = i * chunk_sec
        duration   = min(chunk_sec, total_sec - start_sec)
        chunk_file = chunks_dir / f"chunk_{i+1:04d}.mp4"
        partial    = chunks_dir / f"chunk_{i+1:04d}.partial.mp4"
        chunk_paths.append(chunk_file)
        chunk_label = f"[{i+1}/{n_chunks}] "

        # Resume: skip completed chunks
        if chunk_file.exists() and chunk_file.stat().st_size > 10_000:
            size_mb = chunk_file.stat().st_size / 1024 / 1024
            print(f"  {C.GREY}{chunk_label}✓ Already done: {chunk_file.name} ({size_mb:.0f} MB){C.RESET}")
            continue

        # Clean up any leftover partial
        if partial.exists():
            partial.unlink()
            log(f"Removed partial: {partial}", args.verbose)

        ts_start = f"{int(start_sec//3600):02d}:{int((start_sec%3600)//60):02d}:{start_sec%60:06.3f}"
        print(f"  {C.BOLD}{chunk_label}Encoding {start_sec/60:.1f}–{(start_sec+duration)/60:.1f} min  ({duration:.0f}s){C.RESET}")

        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'warning',
            *hwaccel_flags,
            '-ss', ts_start,   # fast seek before -i
            '-t',  str(duration),
            '-i',  str(info.path),
            '-c:v', enc,
            *quality_flags,
            *hdr_flags,
            '-c:a', 'aac', '-b:a', '192k',
            '-movflags', '+faststart',
            '-y', str(partial)
        ]

        if args.dry_run:
            print(f"  {C.DIM}[DRY] {' '.join(cmd)}{C.RESET}")
            chunk_file.touch()
            continue

        ok = run_ffmpeg_progress(cmd, duration, chunk_label=chunk_label, verbose=args.verbose)

        if ok and partial.exists() and partial.stat().st_size > 10_000:
            partial.rename(chunk_file)
            size_mb = chunk_file.stat().st_size / 1024 / 1024
            print(f"  {C.GREEN}  └─ ✓ Saved {chunk_file.name} ({size_mb:.0f} MB){C.RESET}")
        else:
            msg = "ffmpeg error" if not ok else f"output too small ({partial.stat().st_size if partial.exists() else 0} bytes)"
            print(f"  {C.RED}  └─ ✗ Chunk {i+1} failed ({msg}) — stopping.{C.RESET}")
            if partial.exists(): partial.unlink()
            sys.exit(1)

    # ── Concatenation ──────────────────────────────────────────────────────────
    concat_list = chunks_dir / "concat.txt"
    with open(concat_list, 'w') as f:
        for cp in chunk_paths:
            escaped = str(cp.resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    print(f"\n  {C.CYAN}[⧉] Concatenating {n_chunks} chunk(s) → {final_out.name}{C.RESET}")

    concat_cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'warning',
        '-f', 'concat', '-safe', '0',
        '-i', str(concat_list),
        '-c', 'copy',
        '-movflags', '+faststart',
        '-y', str(final_out)
    ]

    if args.dry_run:
        print(f"  {C.DIM}[DRY] {' '.join(concat_cmd)}{C.RESET}"); return

    ok = run_ffmpeg_progress(concat_cmd, total_sec,
                             chunk_label="concat ", verbose=args.verbose)

    if ok and final_out.exists():
        elapsed  = time.time() - total_start
        out_mb   = final_out.stat().st_size / 1024 / 1024
        savings  = info.size_mb - out_mb
        ratio    = out_mb / info.size_mb * 100 if info.size_mb > 0 else 0
        print(f"\n  {C.GREEN}[✓] Complete in {elapsed/60:.1f} min{C.RESET}")
        print(f"  {C.GREEN}    {info.size_mb:.0f} MB → {out_mb:.0f} MB  "
              f"({ratio:.0f}% of original, -{savings:.0f} MB){C.RESET}")

        if not args.keep_chunks:
            for cp in chunk_paths:
                if cp.exists(): cp.unlink()
            if concat_list.exists(): concat_list.unlink()
            try: chunks_dir.rmdir()
            except OSError: pass
            print(f"  {C.GREY}[~] Chunks cleaned up.{C.RESET}")
        else:
            print(f"  {C.GREY}[~] Chunks kept: {chunks_dir}{C.RESET}")
    else:
        print(f"  {C.RED}[✗] Concatenation failed.{C.RESET}")

# ─── Scanner ───────────────────────────────────────────────────────────────────
SUPPORTED_EXTS = ('.mov', '.mp4', '.m4v', '.mxf', '.avi', '.mkv')

def scan_directory(path: str, recursive: bool, verbose: bool):
    p = Path(path)
    for f in sorted(p.glob('**/*' if recursive else '*')):
        if not f.is_file(): continue
        if f.suffix.lower() not in SUPPORTED_EXTS: continue
        if f.name.startswith('._'): continue
        if any(x in f.name for x in ('_compressed', '_audio', '_chunks')): continue
        log(f"Found: {f}", verbose)
        yield str(f)

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="compress3.py — Cross-Platform Chunked Video Compressor",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("path", nargs="?", default=".",
                        help="Directory or single file (default: .)")
    parser.add_argument("--threshold",     type=float, default=25.0,
                        help="Bitrate Mbps threshold to trigger compression (default: 25)")
    parser.add_argument("--crf",           type=int,   default=23,
                        help="Quality 0–51, lower=better (default: 23)")
    parser.add_argument("--codec",         choices=["h264", "hevc"], default="h264",
                        help="Preferred output codec (default: h264)")
    parser.add_argument("--chunk-minutes", type=int,   default=10,
                        help="Chunk length in minutes (default: 10)")
    parser.add_argument("--onlyaudio",     action="store_true", help="Only extract audio")
    parser.add_argument("--noaudio",       action="store_true", help="Skip audio extraction")
    parser.add_argument("--overwrite",     action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--keep-chunks",   action="store_true", help="Keep chunks after concat")
    parser.add_argument("--recursive","-r",action="store_true", help="Recurse subdirectories")
    parser.add_argument("--scan-only",     action="store_true", help="Probe only, no encoding")
    parser.add_argument("--dry-run",       action="store_true", help="Print commands, no execution")
    parser.add_argument("--verbose","-v",  action="store_true", help="Debug verbosity")
    args = parser.parse_args()

    # ── Check for ffmpeg / ffprobe ─────────────────────────────────────────────
    missing = [t for t in ('ffmpeg', 'ffprobe') if not shutil.which(t)]
    if missing:
        cprint(C.RED, f"\n[!] Missing tools: {', '.join(missing)}")
        system = platform.system()
        if system == "Darwin":
            print("    Install via Homebrew:  brew install ffmpeg")
        elif system == "Linux":
            print("    Install via apt:       sudo apt install ffmpeg")
            print("    Install via dnf:       sudo dnf install ffmpeg")
        elif system == "Windows":
            print("    Download from:         https://ffmpeg.org/download.html")
            print("    Or via winget:         winget install Gyan.FFmpeg")
        print()
        sys.exit(1)

    # ── Detect encoders ───────────────────────────────────────────────────────
    encoders = detect_encoders(args.codec, args.verbose)
    if not encoders:
        cprint(C.RED, "\n[!] No usable video encoders found. Install ffmpeg with codec support.")
        sys.exit(1)

    # ── Find files ────────────────────────────────────────────────────────────
    scan_path = os.path.expanduser(args.path)
    files = ([scan_path] if os.path.isfile(scan_path)
             else list(scan_directory(scan_path, args.recursive, args.verbose)))

    if not files:
        print(f"{C.YELLOW}[!] No supported video files found.{C.RESET}"); sys.exit(0)

    # ── Header ────────────────────────────────────────────────────────────────
    enc_display = encoders['video'] or "none"
    sw_note     = f"  {C.YELLOW}(software fallback){C.RESET}" if encoders['is_software'] else ""
    cprint(C.BOLD + C.WHITE, f"\n{'─'*62}")
    cprint(C.BOLD + C.WHITE,  "  compress3.py — Cross-Platform Video Compressor v3")
    cprint(C.BOLD + C.WHITE, f"{'─'*62}")
    print(f"  Platform     : {platform.system()} {platform.machine()}  [{encoders['label']}]")
    print(f"  Encoder      : {C.CYAN}{enc_display}{C.RESET}{sw_note}")
    print(f"  Path         : {scan_path}")
    print(f"  Files        : {len(files)} candidate(s)")
    print(f"  Threshold    : {args.threshold} Mbps")
    print(f"  CRF          : {args.crf}")
    print(f"  Chunk size   : {args.chunk_minutes} min")
    flags = [f for f, on in [("DRY-RUN", args.dry_run), ("SCAN-ONLY", args.scan_only),
             ("RECURSIVE", args.recursive), ("VERBOSE", args.verbose),
             ("KEEP-CHUNKS", args.keep_chunks), ("OVERWRITE", args.overwrite)] if on]
    if flags: print(f"  Flags        : {', '.join(flags)}")
    cprint(C.BOLD + C.WHITE, f"{'─'*62}\n")

    # ── Scan & decide ─────────────────────────────────────────────────────────
    to_process = []
    for filepath in files:
        fname = os.path.basename(filepath)
        print(f"{C.BOLD}[→] {fname}{C.RESET}")
        info = probe_file(filepath, args.verbose)
        if not info:
            print(f"  {C.RED}[!] Could not probe, skipping.{C.RESET}\n"); continue

        print(f"  Codec    : {C.CYAN}{info.codec}{C.RESET}"
              + (f"  {C.YELLOW}[ProRes]{C.RESET}"  if info.is_prores else "")
              + (f"  {C.YELLOW}[HDR]{C.RESET}"     if info.is_hdr    else ""))
        print(f"  Video    : {info.width}x{info.height} @ {info.fps:.2f}fps  |  {info.color_space}")
        print(f"  Bitrate  : {info.bitrate_mbps:.2f} Mbps  [{info.compression_label}]")
        print(f"  Duration : {info.duration_str}  |  {info.size_mb:.1f} MB")
        if info.has_audio:
            print(f"  Audio    : {info.audio_codec} @ {info.audio_bitrate_kbps:.0f} kbps")
        else:
            print(f"  Audio    : {C.GREY}none{C.RESET}")

        do_it, reason = should_compress(info, args.threshold, args.verbose)
        if do_it:
            print(f"  Decision : {C.YELLOW}COMPRESS{C.RESET}  — {reason}")
            to_process.append(info)
        else:
            print(f"  Decision : {C.GREEN}SKIP{C.RESET}  — {reason}")
        print()

    # ── Gate ──────────────────────────────────────────────────────────────────
    cprint(C.BOLD + C.WHITE, f"{'─'*62}")
    print(f"  {len(to_process)} of {len(files)} file(s) queued for processing.")
    if not to_process or args.scan_only:
        if args.scan_only: cprint(C.GREY, "  [--scan-only] No encoding performed.")
        print(); sys.exit(0)

    # ── Process ───────────────────────────────────────────────────────────────
    for i, info in enumerate(to_process, 1):
        fname = os.path.basename(info.path)
        cprint(C.BOLD + C.WHITE, f"\n[{i}/{len(to_process)}] {fname}")
        cprint(C.GREY, f"  {info.size_mb:.1f} MB  |  {info.duration_str}  |  {info.bitrate_mbps:.1f} Mbps")
        try:
            if not args.noaudio and not args.onlyaudio:
                extract_audio(info, args)
            if not args.onlyaudio:
                compress_video_chunked(info, args, encoders)
            elif args.onlyaudio:
                extract_audio(info, args)
        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}[!] Interrupted — chunks saved, re-run to resume.{C.RESET}")
            sys.exit(130)

    cprint(C.BOLD + C.GREEN, "\n  ✓ All done.\n")

if __name__ == "__main__":
    main()