#!/usr/bin/env python3
"""
vcompress.py — Cross-Platform Chunked Video Compressor

Hardware acceleration: Apple VideoToolbox, NVIDIA NVENC, AMD AMF, Intel QSV, VAAPI
Audio backends:  ffmpeg · afconvert (macOS CoreAudio) · VLC · sox · pydub · lame
Chunked encoding with resume, live progress bar, flexible trim/range.

Usage examples:
  python compress3.py .                          # compress all high-bitrate files
  python compress3.py video.mov --onlyaudio      # extract audio only
  python compress3.py . --audio-format flac      # extract to FLAC
  python compress3.py . --output-format mkv      # output MKV instead of MP4
  python compress3.py . --from 1:30 --to 5:00    # trim to range
  python compress3.py . --trimfront 30 --trimback 10  # remove 30s front, 10s back
  python compress3.py . --audio-backend vlc      # force VLC for audio
  python compress3.py . --pipe                   # force pipe mode for audio
  python compress3.py . --direct-mp3             # direct MOV→MP3 via lame
  python compress3.py . --list-backends          # show detected tools
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

# Enable ANSI on Windows if colorama is available
if platform.system() == "Windows":
    try:
        import colorama; colorama.init()
    except ImportError:
        for attr in list(vars(C).keys()):
            if not attr.startswith('_'):
                setattr(C, attr, "")

def cprint(color, msg): print(f"{color}{msg}{C.RESET}")
def log(msg, verbose):
    if verbose: print(f"  {C.GREY}[DBG] {msg}{C.RESET}")

# ─── Time Parsing ──────────────────────────────────────────────────────────────
def parse_time(s: str) -> float:
    """
    Parse flexible time specification into float seconds.
    Accepts: 30  90.5  1:30  1:30:00  1h30m  1h30m20s  90m  2h  30s
    """
    s = s.strip()
    if re.match(r'^[\d:.]+$', s):
        parts = s.split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    h   = re.search(r'(\d+(?:\.\d+)?)h',     s)
    m   = re.search(r'(\d+(?:\.\d+)?)m(?!s)', s)
    sec = re.search(r'(\d+(?:\.\d+)?)s',      s)
    result = 0.0
    if h:   result += float(h.group(1))   * 3600
    if m:   result += float(m.group(1))   * 60
    if sec: result += float(sec.group(1))
    if result > 0:
        return result
    raise ValueError(f"Cannot parse time: {s!r}  (try: 30 | 1:30 | 1:30:00 | 1h30m | 90m)")

def sec_to_ts(sec: float) -> str:
    """Seconds → HH:MM:SS.mmm (ffmpeg -ss / -to format)."""
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def _resolve_trim(args, duration: float) -> tuple:
    """Resolve --from/--to/--trimfront/--trimback into (start_sec, end_sec)."""
    start = 0.0
    end   = duration
    if getattr(args, 'from_time',  None): start = max(start, parse_time(args.from_time))
    if getattr(args, 'trimfront',  None): start = max(start, parse_time(args.trimfront))
    if getattr(args, 'to_time',    None): end   = parse_time(args.to_time)
    if getattr(args, 'trimback',   None): end   = min(end, duration - parse_time(args.trimback))
    start = max(0.0, start)
    end   = min(end, duration)
    if end <= start:
        raise ValueError(f"Trim gives zero-length output (start={start:.1f}s end={end:.1f}s)")
    return start, end

def _trim_flags(args, duration: float) -> list:
    """Return [-ss start -to end] flags (empty list if no trim)."""
    start, end = _resolve_trim(args, duration)
    flags = []
    if start > 0:        flags += ['-ss', sec_to_ts(start)]
    if end < duration:   flags += ['-to', sec_to_ts(end)]
    return flags

# ─── Format Tables ─────────────────────────────────────────────────────────────
AUDIO_FORMATS = {
    # key: (ext, ffmpeg_codec, quality_flags)
    'mp3':  ('.mp3',  'libmp3lame', ['-b:a', '192k']),
    'aac':  ('.m4a',  'aac',        ['-b:a', '192k']),
    'm4a':  ('.m4a',  'aac',        ['-b:a', '192k']),
    'flac': ('.flac', 'flac',       ['-compression_level', '8']),
    'wav':  ('.wav',  'pcm_s16le',  []),
    'opus': ('.opus', 'libopus',    ['-b:a', '128k']),
    'ogg':  ('.ogg',  'libvorbis',  ['-q:a', '5']),
    'aiff': ('.aiff', 'pcm_s16be',  []),
}

VIDEO_FORMATS = {
    # key: (ext, extra_mux_flags)
    'mp4':  ('.mp4', ['-movflags', '+faststart']),
    'mkv':  ('.mkv', []),
    'mov':  ('.mov', []),
    'avi':  ('.avi', []),
}

# ─── Tool Detection ────────────────────────────────────────────────────────────
def detect_tools(verbose: bool) -> dict:
    """
    Detect available CLI tools and Python libraries for audio/video processing.
    Returns dict of name → path/bool.
    """
    tools = {}

    # CLI tools
    cli_tools = {
        'ffmpeg':     'ffmpeg',
        'ffprobe':    'ffprobe',
        'afconvert':  'afconvert',         # macOS CoreAudio converter
        'sox':        'sox',               # SoX: Sound eXchange
        'lame':       'lame',              # direct MP3 encoder
        'mpv':        'mpv',               # mpv player (audio extract via pipe)
        'mplayer':    'mplayer',
        'gstreamer':  'gst-launch-1.0',    # GStreamer pipeline
        'avconv':     'avconv',            # libav ffmpeg fork
    }
    for key, name in cli_tools.items():
        tools[key] = shutil.which(name)

    # VLC: check platform-specific paths first, then PATH
    vlc_candidates = [
        shutil.which('cvlc'),
        shutil.which('vlc'),
        '/Applications/VLC.app/Contents/MacOS/VLC',              # macOS
        r'C:\Program Files\VideoLAN\VLC\vlc.exe',                # Windows
        r'C:\Program Files (x86)\VideoLAN\VLC\vlc.exe',
    ]
    tools['vlc'] = next((p for p in vlc_candidates if p and os.path.isfile(p)), None)

    # Python libraries
    for lib in ('pydub', 'soundfile', 'audioread', 'mutagen', 'moviepy'):
        try:
            __import__(lib)
            tools[f'py_{lib}'] = True
        except ImportError:
            tools[f'py_{lib}'] = False

    if verbose:
        found   = [k for k, v in tools.items() if v]
        missing = [k for k, v in tools.items() if not v]
        log(f"Tools found:   {', '.join(found)}",   True)
        log(f"Tools missing: {', '.join(missing)}", True)

    return tools

# ─── Encoder Detection ─────────────────────────────────────────────────────────
def detect_encoders(preferred_codec: str, verbose: bool) -> Optional[dict]:
    system = platform.system()
    log(f"Platform: {system} ({platform.machine()})", verbose)

    if not shutil.which('ffmpeg'):
        return None

    try:
        r = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'],
                           capture_output=True, text=True, timeout=10)
        available = r.stdout + r.stderr
    except Exception:
        available = ""

    def has(name): return f" {name} " in available or f"\n {name} " in available

    if system == "Darwin":
        h264_chain = ['h264_videotoolbox', 'libx264']
        hevc_chain = ['hevc_videotoolbox', 'libx265']
        label      = "Apple VideoToolbox"
    elif system == "Windows":
        h264_chain = ['h264_nvenc', 'h264_amf', 'h264_qsv', 'libx264']
        hevc_chain = ['hevc_nvenc', 'hevc_amf', 'hevc_qsv', 'libx265']
        label      = "Windows HW"
    else:
        h264_chain = ['h264_nvenc', 'h264_vaapi', 'h264_qsv', 'libx264']
        hevc_chain = ['hevc_nvenc', 'hevc_vaapi', 'hevc_qsv', 'libx265']
        label      = "Linux HW"

    def pick(chain):
        for enc in chain:
            if has(enc):
                log(f"Selected encoder: {enc}", verbose)
                return enc
        return None

    h264   = pick(h264_chain)
    hevc   = pick(hevc_chain)
    if not h264 and not hevc:
        return None

    chosen = h264 if preferred_codec == 'h264' else (hevc or h264)
    return {
        'video':       chosen,
        'video_h264':  h264,
        'video_hevc':  hevc,
        'label':       label,
        'is_hw':       not (chosen or '').startswith('lib'),
        'is_vt':       'videotoolbox' in (chosen or ''),
        'is_nvenc':    'nvenc' in (chosen or ''),
        'is_software': (chosen or '').startswith('lib'),
    }

def build_quality_flags(enc: str, crf: int) -> list:
    if 'videotoolbox' in enc:  return ['-q:v', str(int(100 - crf * 1.5))]
    if 'nvenc' in enc or 'amf' in enc: return ['-rc', 'vbr', '-cq', str(crf)]
    if 'qsv'  in enc:          return ['-global_quality', str(crf)]
    if 'vaapi' in enc:         return ['-compression_level', str(crf)]
    return ['-crf', str(crf)]

def build_hwaccel_flags(enc: str) -> list:
    if 'videotoolbox' in enc: return ['-hwaccel', 'videotoolbox']
    if 'nvenc' in enc:        return ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda']
    if 'qsv'  in enc:         return ['-hwaccel', 'qsv']
    if 'vaapi' in enc:        return ['-hwaccel', 'vaapi', '-vaapi_device', '/dev/dri/renderD128']
    return []

# ─── File Info ─────────────────────────────────────────────────────────────────
@dataclass
class FileInfo:
    path:               str
    codec:              str
    bitrate_mbps:       float
    duration_sec:       float
    width:              int
    height:             int
    fps:                float
    size_mb:            float
    has_audio:          bool
    audio_codec:        str
    audio_codec_tag:    str
    audio_bitrate_kbps: float
    audio_channels:     int
    audio_sample_rate:  int
    container:          str
    color_space:        str
    is_hdr:             bool

    @property
    def duration_str(self):
        h, rem = divmod(int(self.duration_sec), 3600)
        m, s   = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @property
    def is_prores(self): return 'prores' in self.codec.lower()
    @property
    def is_raw(self): return self.codec.lower() in ('dnxhd', 'dnxhr', 'cineform', 'v210')
    @property
    def is_in24(self): return self.audio_codec_tag.lower() in ('in24', 'in32', 'in16')

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
        r = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', filepath],
            capture_output=True, text=True, timeout=30
        )
        if r.returncode != 0:
            return None

        data    = json.loads(r.stdout)
        streams = data.get('streams', [])
        fmt     = data.get('format', {})

        vstream = next((s for s in streams if s.get('codec_type') == 'video'), None)
        astream = next((s for s in streams if s.get('codec_type') == 'audio'), None)
        if not vstream:
            return None

        codec       = vstream.get('codec_name', 'unknown').lower()
        width       = int(vstream.get('width', 0))
        height      = int(vstream.get('height', 0))
        color_space = vstream.get('color_space', 'unknown')
        color_trc   = vstream.get('color_transfer', '')
        is_hdr      = any(x in color_trc.lower() for x in ('smpte2084', 'arib-std-b67', 'bt2020'))

        fps = 0.0
        try:
            n, d = map(int, vstream.get('r_frame_rate', '0/1').split('/'))
            fps  = n / d if d else 0.0
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
            bitrate_mbps = os.path.getsize(filepath) * 8 / (duration * 1_000_000)

        log(f"codec={codec} res={width}x{height} fps={fps:.2f} "
            f"br={bitrate_mbps:.2f}Mbps dur={duration:.1f}s hdr={is_hdr}", verbose)

        a = astream or {}
        has_audio         = astream is not None
        audio_codec       = a.get('codec_name', 'none')
        audio_codec_tag   = a.get('codec_tag_string', '').strip()
        audio_channels    = int(a.get('channels', 0))
        audio_sample_rate = int(a.get('sample_rate', 0)) if a.get('sample_rate') else 0
        try:
            raw = a.get('bit_rate', '0')
            audio_br_kbps = int(raw) / 1000 if raw not in ('N/A', '', None) else 0.0
        except (ValueError, TypeError):
            audio_br_kbps = 0.0

        return FileInfo(
            path=filepath, codec=codec, bitrate_mbps=bitrate_mbps,
            duration_sec=duration, width=width, height=height, fps=fps,
            size_mb=os.path.getsize(filepath) / (1024 * 1024),
            has_audio=has_audio, audio_codec=audio_codec,
            audio_codec_tag=audio_codec_tag, audio_bitrate_kbps=audio_br_kbps,
            audio_channels=audio_channels, audio_sample_rate=audio_sample_rate,
            container=Path(filepath).suffix.lower().lstrip('.'),
            color_space=color_space, is_hdr=is_hdr
        )
    except subprocess.TimeoutExpired:
        print(f"{C.RED}  [!] ffprobe timed out{C.RESET}"); return None
    except Exception as e:
        print(f"{C.RED}  [!] probe error: {e}{C.RESET}"); return None

def should_compress(info: FileInfo, threshold: float, verbose: bool) -> tuple:
    reasons = []
    if info.is_prores:                reasons.append(f"ProRes codec ({info.codec})")
    if info.is_raw:                   reasons.append(f"Raw codec ({info.codec})")
    if info.bitrate_mbps > threshold: reasons.append(f"Bitrate {info.bitrate_mbps:.1f} Mbps > {threshold} Mbps")
    if info.size_mb > 500 and info.bitrate_mbps > 10: reasons.append(f"Large file {info.size_mb:.0f} MB")
    if reasons:
        log(f"Compress: {'; '.join(reasons)}", verbose)
        return True, "; ".join(reasons)
    return False, f"within acceptable range ({info.bitrate_mbps:.1f} Mbps)"

# ─── Progress Bar ──────────────────────────────────────────────────────────────
_STATS_RE = re.compile(
    r'frame=\s*(\d+).*?fps=\s*([\d.]+).*?time=\s*([\d:.]+)'
    r'.*?bitrate=\s*(\S+).*?speed=\s*([\d.]+)x',
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

def _render_bar(cur: float, total: float, fps: float, speed: float,
                bitrate: str, label: str):
    W   = 36
    pct = min(cur / total, 1.0) if total > 0 else 0.0
    bar = "█" * int(W * pct) + "░" * (W - int(W * pct))
    eta = ""
    if speed > 0.01 and 0 < pct < 1.0:
        rem = (total - cur) / speed
        eta = f" ETA {int(rem//60):02d}:{int(rem%60):02d}"
    line = (f"\r  {C.CYAN}{label}[{bar}]{C.RESET}"
            f" {C.BOLD}{pct*100:5.1f}%{C.RESET}{C.GREY}"
            + (f" {fps:.0f}fps"    if fps > 0   else "")
            + (f" {speed:.2f}x"    if speed > 0 else "")
            + (f" {bitrate}"       if bitrate    else "")
            + eta + f"{C.RESET}  ")
    sys.stdout.write(line)
    sys.stdout.flush()

def run_ffmpeg_progress(cmd: list, total_sec: float,
                        label: str = "", verbose: bool = False) -> bool:
    """Run ffmpeg, render live progress bar from its stderr. Returns True on success."""
    if verbose:
        print(f"\n  {C.GREY}[DBG] {' '.join(str(x) for x in cmd)}{C.RESET}")

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print(f"{C.RED}  [!] ffmpeg not found{C.RESET}"); return False

    cur = fps = speed = 0.0
    bitrate = ""
    buf     = b""

    while True:
        chunk = proc.stderr.read(256)
        if not chunk: break
        buf += chunk
        while b'\r' in buf or b'\n' in buf:
            for sep in (b'\r', b'\n'):
                idx = buf.find(sep)
                if idx == -1: continue
                line_b = buf[:idx]; buf = buf[idx+1:]
                line   = line_b.decode('utf-8', errors='replace').strip()
                if not line: continue

                is_stats = 'time=' in line and 'frame=' in line
                is_bad   = any(x in line.lower() for x in (
                    'error', 'invalid', 'failed', 'overread',
                    'guessed', 'no filtered', 'nothing was encoded'))
                is_warn  = 'warning' in line.lower()

                if is_bad or is_warn or (verbose and not is_stats):
                    color = (C.RED if 'error' in line.lower() or 'failed' in line.lower()
                             else C.YELLOW if is_bad or is_warn else C.GREY)
                    sys.stdout.write(f"\n  {color}[ffmpeg] {line}{C.RESET}")
                    sys.stdout.flush()
                    if not is_stats: continue

                m = _STATS_RE.search(line)
                if m:
                    try: fps  = float(m.group(2))
                    except ValueError: pass
                    try: cur  = _hms_to_sec(m.group(3))
                    except Exception: pass
                    bitrate = m.group(4)
                    try: speed = float(m.group(5))
                    except ValueError: pass
                else:
                    tm = _TIME_RE.search(line)
                    if tm:
                        try: cur = _hms_to_sec(tm.group(1))
                        except Exception: pass

                if cur > 0: _render_bar(cur, total_sec, fps, speed, bitrate, label)
                break

    proc.wait()
    if total_sec > 0: _render_bar(total_sec, total_sec, fps, speed, bitrate, label)
    sys.stdout.write("\n")
    if proc.returncode not in (0, None):
        print(f"  {C.RED}[✗] ffmpeg exited {proc.returncode}{C.RESET}")
        return False
    return True

# ─── Audio Extraction Backends ─────────────────────────────────────────────────

def _afmt(args) -> tuple:
    """Return (ext, ffmpeg_codec, quality_flags) for requested audio format."""
    key = (getattr(args, 'audio_format', None) or 'mp3').lower()
    return AUDIO_FORMATS.get(key, AUDIO_FORMATS['mp3'])

def _audio_out(info: FileInfo, args) -> str:
    ext = _afmt(args)[0]
    return str(Path(info.path).with_name(Path(info.path).stem + f"_audio{ext}"))

# PCM codec_name → raw sample format string for pipe mode
_PCM_FMT = {
    'pcm_s8':    's8',    'pcm_u8':    'u8',
    'pcm_s16le': 's16le', 'pcm_s16be': 's16be',
    'pcm_s24le': 's24le', 'pcm_s24be': 's24be',
    'pcm_s32le': 's32le', 'pcm_s32be': 's32be',
    'pcm_f32le': 'f32le', 'pcm_f32be': 'f32be',
}


def _backend_ffmpeg(filepath, out, duration, args, ainfo) -> bool:
    """
    Standard ffmpeg extraction with three fallback strategies:
      1. plain -vn -ac 2
      2. -guess_layout_max 0 (prevents macOS filter-graph failure on guessed layouts)
      3. -guess_layout_max 0 + -map 0:a:0 (explicit stream selection)
    """
    ext, codec, quality = _afmt(args)
    trim  = _trim_flags(args, duration)
    ac    = ['-ac', '2']     if ext not in ('.wav', '.aiff', '.flac') else []
    rate  = ['-ar', '44100'] if ext not in ('.wav', '.aiff', '.flac') else []
    enc   = ['-c:a', codec] + quality

    base = ['-vn', *trim, *ac, *rate, *enc, '-y', out]

    strats = [
        ('ffmpeg',         ['ffmpeg', '-hide_banner', '-loglevel', 'warning',
                            '-i', filepath, *base]),
        ('ffmpeg/no-guess',['ffmpeg', '-hide_banner', '-loglevel', 'warning',
                            '-guess_layout_max', '0', '-i', filepath, *base]),
        ('ffmpeg/map',     ['ffmpeg', '-hide_banner', '-loglevel', 'warning',
                            '-guess_layout_max', '0', '-i', filepath,
                            '-map', '0:a:0', *trim, *ac, *rate, *enc, '-y', out]),
    ]

    for label, cmd in strats:
        _clean(out)
        if args.dry_run:
            print(f"  {C.DIM}[DRY:{label}] {' '.join(cmd)}{C.RESET}")
            return True
        log(f"Strategy: {label}", args.verbose)
        eff = _trim_dur(args, duration)
        ok  = run_ffmpeg_progress(cmd, eff, label=f"{label:14}", verbose=args.verbose)
        sz  = _sz(out)
        if ok and sz > 10_000: return True
        log(f"  → {sz}b, trying next", args.verbose)
    return False


def _backend_pipe(filepath, out, duration, args, ainfo) -> bool:
    """
    Two-process pipe: proc1 demuxes raw PCM bytes (no filter graph),
    proc2 encodes with explicit format params.
    Bypasses the filter-graph channel-layout issue entirely.
    """
    ext, codec, quality = _afmt(args)
    channels    = ainfo.get('channels', 2)
    sample_rate = ainfo.get('sample_rate', 48000)
    codec_name  = ainfo.get('codec_name', 'pcm_s16le')
    raw_fmt     = _PCM_FMT.get(codec_name, 's16le')

    trim  = _trim_flags(args, duration)
    ac    = ['-ac', '2']     if ext not in ('.wav', '.aiff', '.flac') else []
    rate  = ['-ar', '44100'] if ext not in ('.wav', '.aiff', '.flac') else []

    cmd1 = ['ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-i', filepath, '-vn', *trim, '-c:a', 'copy', '-f', raw_fmt, '-']
    cmd2 = ['ffmpeg', '-hide_banner', '-loglevel', 'warning',
            '-f', raw_fmt, '-ar', str(sample_rate), '-ac', str(channels), '-i', '-',
            *ac, *rate, '-c:a', codec, *quality, '-y', out]

    if args.dry_run:
        print(f"  {C.DIM}[DRY:pipe] {' '.join(cmd1)} |{C.RESET}")
        print(f"  {C.DIM}          {' '.join(cmd2)}{C.RESET}")
        return True

    log(f"Pipe cmd1: {' '.join(cmd1)}", args.verbose)
    log(f"Pipe cmd2: {' '.join(cmd2)}", args.verbose)

    try:
        p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p2 = subprocess.Popen(cmd2, stdin=p1.stdout,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        p1.stdout.close()
        eff = _trim_dur(args, duration)
        buf = b""; cur = fps = speed = 0.0; bitrate = ""
        while True:
            chunk = p2.stderr.read(256)
            if not chunk: break
            buf += chunk
            while b'\r' in buf or b'\n' in buf:
                for sep in (b'\r', b'\n'):
                    idx = buf.find(sep)
                    if idx == -1: continue
                    line = buf[:idx].decode('utf-8', errors='replace').strip()
                    buf  = buf[idx+1:]
                    if not line: continue
                    is_bad = any(x in line.lower() for x in ('error','failed','no filtered','nothing was encoded'))
                    if is_bad or (args.verbose and 'time=' not in line):
                        color = C.RED if 'error' in line.lower() else C.YELLOW if is_bad else C.GREY
                        sys.stdout.write(f"\n  {color}[pipe] {line}{C.RESET}"); sys.stdout.flush()
                    m = _STATS_RE.search(line)
                    if m:
                        try: fps=float(m.group(2))
                        except: pass
                        try: cur=_hms_to_sec(m.group(3))
                        except: pass
                        bitrate=m.group(4)
                        try: speed=float(m.group(5))
                        except: pass
                    if cur > 0: _render_bar(cur, eff, fps, speed, bitrate, "pipe          ")
                    break
        p2.wait(); p1.wait()
        if eff > 0: _render_bar(eff, eff, fps, speed, bitrate, "pipe          ")
        sys.stdout.write("\n")
    except Exception as e:
        log(f"Pipe error: {e}", args.verbose); return False

    return p2.returncode == 0 and _sz(out) > 10_000


def _backend_afconvert(filepath, out, duration, args, ainfo) -> bool:
    """
    macOS CoreAudio via afconvert.
    The ONLY tool that correctly decodes Atomos in24/in32 packed PCM.
    Converts to temp WAV first (CoreAudio path), then ffmpeg encodes to target format.
    """
    ext, codec, quality = _afmt(args)
    tmp = out.rsplit('.', 1)[0] + '_tmp_af.wav'

    cmd1 = ['afconvert', '-f', 'WAVE', '-d', 'LEI16@44100', '-c', '2', filepath, tmp]
    if args.dry_run:
        print(f"  {C.DIM}[DRY:afconvert] {' '.join(cmd1)}{C.RESET}"); return True

    log(f"afconvert: {' '.join(cmd1)}", args.verbose)
    r1 = subprocess.run(cmd1, capture_output=True, text=True)
    if args.verbose and r1.stderr.strip():
        for line in r1.stderr.strip().splitlines():
            print(f"  {C.GREY}[afconvert] {line}{C.RESET}")

    if r1.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) < 1000:
        log(f"afconvert failed (exit={r1.returncode})", args.verbose)
        if os.path.exists(tmp): os.remove(tmp)
        return False

    log(f"afconvert WAV: {os.path.getsize(tmp)/1024/1024:.0f} MB", args.verbose)

    # Apply trim and encode
    trim = _trim_flags(args, duration)
    eff  = _trim_dur(args, duration)
    cmd2 = ['ffmpeg', '-hide_banner', '-loglevel', 'warning',
            '-i', tmp, *trim, '-c:a', codec, *quality, '-y', out]
    log(f"WAV→{ext}: {' '.join(cmd2)}", args.verbose)
    ok2 = run_ffmpeg_progress(cmd2, eff, label="afconv        ", verbose=args.verbose)

    if os.path.exists(tmp): os.remove(tmp)
    return ok2 and _sz(out) > 10_000


def _backend_vlc(filepath, out, duration, args, tools) -> bool:
    """
    VLC audio extraction via sout transcode chain.
    Works on all platforms where VLC is installed.
    Handles codecs that ffmpeg cannot, including some proprietary formats.
    Trim is applied post-extraction via ffmpeg if needed.
    """
    vlc_bin = tools.get('vlc')
    if not vlc_bin: return False

    ext, _, _ = _afmt(args)
    VLC_CODEC = {'.mp3':'mp3', '.m4a':'mp4a', '.ogg':'vorb',
                 '.flac':'flac', '.wav':'s16l', '.opus':'opus', '.aiff':'s16b'}
    VLC_MUX   = {'mp3':'mp3', 'mp4a':'mp4', 'vorb':'ogg',
                 'flac':'raw', 's16l':'wav', 'opus':'ogg', 's16b':'aiff'}
    vlc_acodec = VLC_CODEC.get(ext, 'mp3')
    vlc_mux    = VLC_MUX.get(vlc_acodec, 'mp3')
    out_vlc    = out.replace('\\', '/')

    sout = (f"#transcode{{acodec={vlc_acodec},ab=192,channels=2,samplerate=44100}}"
            f":std{{access=file,mux={vlc_mux},dst={out_vlc}}}")

    cmd = [vlc_bin, '--intf', 'dummy', '--no-video', '--no-sout-video',
           filepath, '--sout', sout, 'vlc://quit']

    if args.dry_run:
        print(f"  {C.DIM}[DRY:vlc] {' '.join(cmd)}{C.RESET}"); return True

    log(f"VLC: {' '.join(cmd)}", args.verbose)
    print(f"  {C.GREY}[vlc] Converting... (no live progress){C.RESET}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 120)
    except subprocess.TimeoutExpired:
        log("VLC timed out", args.verbose); return False

    if args.verbose and r.stderr.strip():
        for line in r.stderr.strip().splitlines()[-5:]:
            print(f"  {C.GREY}[vlc] {line}{C.RESET}")

    if _sz(out) < 10_000: return False

    # Post-trim via ffmpeg
    try:
        start, end = _resolve_trim(args, duration)
        if start > 0 or end < duration:
            tmp = out + '.trim_tmp'
            trim_cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'warning',
                        '-i', out, '-ss', sec_to_ts(start), '-to', sec_to_ts(end),
                        '-c:a', 'copy', '-y', tmp]
            r2 = subprocess.run(trim_cmd, capture_output=True, text=True)
            if r2.returncode == 0 and _sz(tmp) > 10_000:
                os.replace(tmp, out)
    except Exception: pass

    return _sz(out) > 10_000


def _backend_sox(filepath, out, duration, args, tools) -> bool:
    """
    SoX audio extraction: ffmpeg demuxes to WAV pipe → sox encodes.
    SoX cannot read video containers directly, so ffmpeg handles demux.
    """
    sox_bin = tools.get('sox')
    if not sox_bin: return False

    ext, _, _ = _afmt(args)
    trim = _trim_flags(args, duration)
    eff  = _trim_dur(args, duration)

    cmd1 = ['ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-i', filepath, '-vn', *trim, '-f', 'wav', '-']

    sox_fmt = {'.mp3':'mp3', '.wav':'wav', '.flac':'flac',
               '.ogg':'ogg', '.aiff':'aiff'}.get(ext, 'mp3')
    cmd2 = [sox_bin, '-t', 'wav', '-', '-t', sox_fmt, '-r', '44100', '-c', '2', out]

    if args.dry_run:
        print(f"  {C.DIM}[DRY:sox] {' '.join(cmd1)} | {' '.join(cmd2)}{C.RESET}"); return True

    log(f"SoX: ffmpeg | sox", args.verbose)
    try:
        p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p2 = subprocess.Popen(cmd2, stdin=p1.stdout,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        p1.stdout.close()
        p2.wait(); p1.wait()
    except Exception as e:
        log(f"SoX error: {e}", args.verbose); return False

    return p2.returncode == 0 and _sz(out) > 10_000


def _backend_pydub(filepath, out, duration, args, tools) -> bool:
    """
    pydub audio extraction. Pure Python, wraps ffmpeg internally.
    Handles trim natively (millisecond precision) and stereo downmix.
    Requires: pip install pydub
    """
    if not tools.get('py_pydub'): return False
    try:
        from pydub import AudioSegment
        ext, _, _ = _afmt(args)
        FMT = {'.mp3':'mp3', '.wav':'wav', '.ogg':'ogg', '.flac':'flac',
               '.m4a':'mp4', '.aiff':'aiff', '.opus':'opus'}
        out_fmt = FMT.get(ext, 'mp3')

        if args.dry_run:
            print(f"  {C.DIM}[DRY:pydub] AudioSegment → {out!r}{C.RESET}"); return True

        log(f"pydub: loading {filepath}", args.verbose)
        audio = AudioSegment.from_file(filepath)
        try:
            start, end = _resolve_trim(args, duration)
            audio = audio[int(start * 1000):int(end * 1000)]
        except Exception: pass

        if audio.channels > 2:
            audio = audio.set_channels(2)
        audio = audio.set_frame_rate(44100)
        audio.export(out, format=out_fmt, bitrate='192k')
        return _sz(out) > 10_000
    except Exception as e:
        log(f"pydub error: {e}", args.verbose); return False


def _backend_lame(filepath, out, duration, args, tools) -> bool:
    """
    Direct MP3 via lame: ffmpeg demuxes to WAV pipe → lame encodes MP3.
    Only outputs MP3 regardless of --audio-format.
    Requires: lame installed separately.
    """
    lame_bin = tools.get('lame')
    if not lame_bin: return False

    trim = _trim_flags(args, duration)
    cmd1 = ['ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-i', filepath, '-vn', *trim, '-ac', '2', '-ar', '44100', '-f', 'wav', '-']
    cmd2 = [lame_bin, '-b', '192', '-h', '-', out]

    if args.dry_run:
        print(f"  {C.DIM}[DRY:lame] {' '.join(cmd1)} | {' '.join(cmd2)}{C.RESET}"); return True

    log(f"lame: ffmpeg | lame", args.verbose)
    try:
        p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p2 = subprocess.Popen(cmd2, stdin=p1.stdout,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        p1.stdout.close()
        p2.wait(); p1.wait()
    except Exception as e:
        log(f"lame error: {e}", args.verbose); return False

    return p2.returncode == 0 and _sz(out) > 10_000


# ─── Helpers ───────────────────────────────────────────────────────────────────
def _sz(path: str) -> int:
    try: return os.path.getsize(path)
    except OSError: return 0

def _clean(path: str):
    if os.path.exists(path) and _sz(path) < 10_000:
        try: os.remove(path)
        except OSError: pass

def _trim_dur(args, duration: float) -> float:
    try:
        start, end = _resolve_trim(args, duration)
        return end - start
    except Exception:
        return duration

def _avail_backends(tools: dict) -> str:
    avail = ['ffmpeg']
    if tools.get('afconvert'):  avail.append('afconvert')
    if tools.get('vlc'):        avail.append('vlc')
    if tools.get('sox'):        avail.append('sox')
    if tools.get('py_pydub'):   avail.append('pydub')
    if tools.get('lame'):       avail.append('lame')
    return ', '.join(avail)

# ─── Audio Extraction Dispatcher ──────────────────────────────────────────────
def extract_audio(info: FileInfo, args, tools: dict):
    """
    Dispatch audio extraction to the right backend(s).

    Priority:
      1. --audio-backend X     → run only that backend
      2. --pipe                → force pipe mode
      3. --direct-mp3          → force lame, fallback to ffmpeg
      4. Auto: in24 on macOS   → afconvert first
      5. Auto: ffmpeg chain    → afconvert → vlc → sox → pydub (fallback cascade)
    """
    out = _audio_out(info, args)

    if not info.has_audio:
        print(f"  {C.YELLOW}[!] No audio stream — skipping.{C.RESET}"); return
    if os.path.exists(out) and not args.overwrite:
        print(f"  {C.GREY}[~] Exists: {Path(out).name}  (--overwrite to redo){C.RESET}"); return
    _clean(out)
    if os.path.exists(out) and args.overwrite:
        os.remove(out)

    # Trim validation
    try:
        start, end = _resolve_trim(args, info.duration_sec)
        if start > 0 or end < info.duration_sec:
            log(f"Trim: {sec_to_ts(start)} → {sec_to_ts(end)}  ({end-start:.1f}s)", args.verbose)
    except ValueError as e:
        print(f"  {C.RED}[!] Trim error: {e}{C.RESET}"); return

    ch_note = f"  ({info.audio_channels}ch→stereo)" if info.audio_channels > 2 else ""
    ext     = _afmt(args)[0]
    if info.is_in24:
        log(f"Atomos in24/in32 detected — ffmpeg 8.x cannot decode", args.verbose)

    print(f"  {C.BLUE}[♫] Audio → {Path(out).name}{ch_note}{C.RESET}")

    ainfo = {
        'channels':    info.audio_channels,
        'sample_rate': info.audio_sample_rate,
        'codec_name':  info.audio_codec,
        'codec_tag':   info.audio_codec_tag,
    }

    forced = getattr(args, 'audio_backend', None)
    BACKEND_MAP = {
        'ffmpeg':    lambda: _backend_ffmpeg(info.path, out, info.duration_sec, args, ainfo),
        'pipe':      lambda: _backend_pipe(info.path, out, info.duration_sec, args, ainfo),
        'afconvert': lambda: _backend_afconvert(info.path, out, info.duration_sec, args, ainfo),
        'vlc':       lambda: _backend_vlc(info.path, out, info.duration_sec, args, tools),
        'sox':       lambda: _backend_sox(info.path, out, info.duration_sec, args, tools),
        'pydub':     lambda: _backend_pydub(info.path, out, info.duration_sec, args, tools),
        'lame':      lambda: _backend_lame(info.path, out, info.duration_sec, args, tools),
    }

    if forced:
        fn = BACKEND_MAP.get(forced)
        if not fn:
            print(f"  {C.RED}[!] Unknown backend: {forced!r}{C.RESET}"); return
        log(f"Forced backend: {forced}", args.verbose)
        ok = fn()

    elif getattr(args, 'pipe', False):
        log("--pipe: forcing pipe mode", args.verbose)
        ok = _backend_pipe(info.path, out, info.duration_sec, args, ainfo)

    elif getattr(args, 'direct_mp3', False):
        log("--direct-mp3: forcing lame", args.verbose)
        ok = _backend_lame(info.path, out, info.duration_sec, args, tools)
        if not ok:
            print(f"  {C.YELLOW}[!] lame not available → falling back to ffmpeg{C.RESET}")
            ok = _backend_ffmpeg(info.path, out, info.duration_sec, args, ainfo)

    else:
        # Auto routing: in24 on macOS → afconvert; otherwise ffmpeg chain
        if info.is_in24 and platform.system() == 'Darwin' and tools.get('afconvert'):
            log("Auto: afconvert for in24 on macOS", args.verbose)
            ok = _backend_afconvert(info.path, out, info.duration_sec, args, ainfo)
            if not ok:
                ok = _backend_ffmpeg(info.path, out, info.duration_sec, args, ainfo)
        else:
            ok = _backend_ffmpeg(info.path, out, info.duration_sec, args, ainfo)

        # Fallback cascade
        if not ok and platform.system() == 'Darwin' and tools.get('afconvert') and not info.is_in24:
            print(f"  {C.YELLOW}[!] ffmpeg failed → afconvert{C.RESET}")
            ok = _backend_afconvert(info.path, out, info.duration_sec, args, ainfo)
        if not ok and tools.get('vlc'):
            print(f"  {C.YELLOW}[!] → VLC{C.RESET}")
            ok = _backend_vlc(info.path, out, info.duration_sec, args, tools)
        if not ok and tools.get('sox'):
            print(f"  {C.YELLOW}[!] → SoX{C.RESET}")
            ok = _backend_sox(info.path, out, info.duration_sec, args, tools)
        if not ok and tools.get('py_pydub'):
            print(f"  {C.YELLOW}[!] → pydub{C.RESET}")
            ok = _backend_pydub(info.path, out, info.duration_sec, args, tools)

    sz = _sz(out)
    if ok and sz > 10_000:
        print(f"  {C.GREEN}[✓] Saved {sz/1024/1024:.1f} MB → {Path(out).name}{C.RESET}")
    else:
        print(f"  {C.RED}[✗] All backends failed (size={sz}b).{C.RESET}")
        if not args.verbose:
            print(f"  {C.YELLOW}    Re-run with --verbose for details.{C.RESET}")
        print(f"  {C.YELLOW}    Available: {_avail_backends(tools)}{C.RESET}")
        if platform.system() == 'Darwin':
            print(f"  {C.YELLOW}    Manual: afconvert -f WAVE -d LEI16@44100 -c 2 \"{info.path}\" /tmp/audio.wav{C.RESET}")

# ─── Chunked Video Compression ─────────────────────────────────────────────────
def compress_video_chunked(info: FileInfo, args, encoders: dict):
    # Resolve trim for video
    try:
        trim_start, trim_end = _resolve_trim(args, info.duration_sec)
    except ValueError as e:
        print(f"  {C.RED}[!] Trim error: {e}{C.RESET}"); return

    trim_dur = trim_end - trim_start

    # Output container
    vfmt = (getattr(args, 'output_format', None) or 'mp4').lower()
    vext, vmux_flags = VIDEO_FORMATS.get(vfmt, VIDEO_FORMATS['mp4'])
    stem       = Path(info.path).stem
    directory  = Path(info.path).parent
    final_out  = directory / f"{stem}_compressed{vext}"
    chunks_dir = directory / f"{stem}_chunks"

    if final_out.exists() and not args.overwrite:
        print(f"  {C.GREY}[~] Exists: {final_out.name}  (--overwrite to redo){C.RESET}"); return

    chunks_dir.mkdir(exist_ok=True)

    enc           = encoders['video']
    chunk_sec     = args.chunk_minutes * 60
    n_chunks      = max(1, -(-int(trim_dur) // int(chunk_sec)))
    quality_flags = build_quality_flags(enc, args.crf)
    hwaccel_flags = build_hwaccel_flags(enc)

    hdr_flags = []
    if info.is_hdr:
        print(f"  {C.YELLOW}[!] HDR — preserving BT.2020/SMPTE2084{C.RESET}")
        hdr_flags = ['-color_primaries', 'bt2020',
                     '-color_trc',       'smpte2084',
                     '-colorspace',      'bt2020nc']

    if trim_start > 0 or trim_end < info.duration_sec:
        print(f"  {C.CYAN}[✂] Trim: {sec_to_ts(trim_start)} → {sec_to_ts(trim_end)}"
              f"  ({trim_dur:.1f}s){C.RESET}")

    q_disp = quality_flags[1] if len(quality_flags) >= 2 else '?'
    print(f"  {C.CYAN}[▶] {n_chunks} chunk(s) × {args.chunk_minutes} min"
          f"  encoder={enc}  quality={quality_flags[0]}={q_disp}{C.RESET}")

    chunk_paths = []
    total_start = time.time()

    for i in range(n_chunks):
        abs_start   = trim_start + i * chunk_sec
        duration    = min(chunk_sec, trim_end - abs_start)
        chunk_file  = chunks_dir / f"chunk_{i+1:04d}{vext}"
        partial     = chunks_dir / f"chunk_{i+1:04d}.partial{vext}"
        chunk_label = f"[{i+1}/{n_chunks}]"
        chunk_paths.append(chunk_file)

        if chunk_file.exists() and chunk_file.stat().st_size > 10_000:
            mb = chunk_file.stat().st_size / 1024 / 1024
            print(f"  {C.GREY}{chunk_label} ✓ {chunk_file.name} ({mb:.0f} MB){C.RESET}")
            continue

        if partial.exists(): partial.unlink()

        print(f"  {C.BOLD}{chunk_label} Encoding "
              f"{abs_start/60:.1f}–{(abs_start+duration)/60:.1f} min"
              f"  ({duration:.0f}s){C.RESET}")

        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'warning',
            *hwaccel_flags,
            '-ss', sec_to_ts(abs_start),
            '-t',  str(duration),
            '-i',  str(info.path),
            '-c:v', enc, *quality_flags, *hdr_flags,
            '-c:a', 'aac', '-b:a', '192k',
            *vmux_flags,
            '-y', str(partial)
        ]

        if args.dry_run:
            print(f"  {C.DIM}[DRY] {' '.join(cmd)}{C.RESET}")
            chunk_file.touch(); continue

        ok = run_ffmpeg_progress(cmd, duration, label=chunk_label, verbose=args.verbose)

        if ok and partial.exists() and partial.stat().st_size > 10_000:
            partial.rename(chunk_file)
            mb = chunk_file.stat().st_size / 1024 / 1024
            print(f"  {C.GREEN}  └─ ✓ {chunk_file.name} ({mb:.0f} MB){C.RESET}")
        else:
            sz  = partial.stat().st_size if partial.exists() else 0
            msg = "ffmpeg error" if not ok else f"too small ({sz}b)"
            print(f"  {C.RED}  └─ ✗ Chunk {i+1} failed ({msg}) — stopping.{C.RESET}")
            if partial.exists(): partial.unlink()
            sys.exit(1)

    # ── Concat ────────────────────────────────────────────────────────────────
    concat_list = chunks_dir / "concat.txt"
    with open(concat_list, 'w', encoding='utf-8') as f:
        for cp in chunk_paths:
            p = str(cp.resolve()).replace('\\', '/').replace("'", "\\'")
            f.write(f"file '{p}'\n")

    print(f"\n  {C.CYAN}[⧉] Concatenating {n_chunks} chunk(s) → {final_out.name}{C.RESET}")

    concat_cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'warning',
        '-f', 'concat', '-safe', '0',
        '-i', str(concat_list), '-c', 'copy',
        *vmux_flags, '-y', str(final_out)
    ]

    if args.dry_run:
        print(f"  {C.DIM}[DRY] {' '.join(concat_cmd)}{C.RESET}"); return

    ok = run_ffmpeg_progress(concat_cmd, trim_dur, label="concat ", verbose=args.verbose)

    if ok and final_out.exists():
        elapsed = time.time() - total_start
        out_mb  = final_out.stat().st_size / 1024 / 1024
        savings = info.size_mb - out_mb
        ratio   = out_mb / info.size_mb * 100 if info.size_mb > 0 else 0
        print(f"\n  {C.GREEN}[✓] Done in {elapsed/60:.1f} min{C.RESET}")
        print(f"  {C.GREEN}    {info.size_mb:.0f} MB → {out_mb:.0f} MB"
              f"  ({ratio:.0f}% of original, -{savings:.0f} MB){C.RESET}")

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
SUPPORTED_EXTS = {'.mov', '.mp4', '.m4v', '.mxf', '.avi', '.mkv', '.ts',
                  '.mts', '.m2ts', '.wmv', '.webm', '.flv', '.3gp', '.dv'}

def scan_directory(path: str, recursive: bool, verbose: bool):
    p = Path(path)
    for f in sorted(p.glob('**/*' if recursive else '*')):
        if not f.is_file(): continue
        if f.suffix.lower() not in SUPPORTED_EXTS: continue
        if f.name.startswith('._'): continue
        if any(x in f.name for x in ('_compressed', '_audio', '_chunks', '.partial')): continue
        log(f"Found: {f.name}", verbose)
        yield str(f)

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="vcompress.py — Cross-Platform Chunked Video Compressor",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Time format (--from / --to / --trimfront / --trimback):
  30   90s   90.5   1:30   01:30:00   1h30m   1h30m20s   90m

Audio backends (--audio-backend):
  ffmpeg     Standard ffmpeg (default, 3 fallback strategies)
  pipe       Two-process raw-PCM pipe (bypasses filter graph)
  afconvert  macOS CoreAudio — required for Atomos in24/in32
  vlc        VLC media player transcode chain
  sox        SoX audio tool (requires ffmpeg for demux)
  pydub      Python pydub library  (pip install pydub)
  lame       Direct lame MP3 encoder pipe

Audio formats (--audio-format):
  mp3  aac  m4a  flac  wav  opus  ogg  aiff

Video output formats (--output-format):
  mp4 (default)  mkv  mov  avi
"""
    )

    parser.add_argument("path",            nargs="?", default=".",
                        help="File or directory (default: .)")
    # Video
    parser.add_argument("--threshold",     type=float, default=25.0,
                        help="Bitrate Mbps to trigger compression (default: 25)")
    parser.add_argument("--crf",           type=int,   default=23,
                        help="Quality 0–51, lower=better (default: 23)")
    parser.add_argument("--codec",         choices=["h264","hevc"], default="h264",
                        help="Preferred video codec (default: h264)")
    parser.add_argument("--chunk-minutes", type=int,   default=10,
                        help="Chunk length in minutes (default: 10)")
    parser.add_argument("--output-format", choices=list(VIDEO_FORMATS), default="mp4",
                        help="Video container format (default: mp4)")
    # Audio
    parser.add_argument("--audio-format",  choices=list(AUDIO_FORMATS), default="mp3",
                        help="Audio output format (default: mp3)")
    parser.add_argument("--audio-backend",
                        choices=['ffmpeg','pipe','afconvert','vlc','sox','pydub','lame'],
                        help="Force specific audio backend")
    parser.add_argument("--pipe",          action="store_true",
                        help="Force two-process pipe mode for audio")
    parser.add_argument("--direct-mp3",    action="store_true",
                        help="Force ffmpeg→lame pipe (requires lame)")
    # Trim
    parser.add_argument("--from",          dest="from_time", metavar="TIME",
                        help="Start time (e.g. 30  1:30  1h30m)")
    parser.add_argument("--to",            dest="to_time",   metavar="TIME",
                        help="End time (e.g. 5:00  2h)")
    parser.add_argument("--trimfront",     metavar="TIME",
                        help="Remove TIME from the beginning")
    parser.add_argument("--trimback",      metavar="TIME",
                        help="Remove TIME from the end")
    # Workflow
    parser.add_argument("--onlyaudio",     action="store_true", help="Audio extraction only")
    parser.add_argument("--noaudio",       action="store_true", help="Skip audio extraction")
    parser.add_argument("--overwrite",     action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--keep-chunks",   action="store_true", help="Keep chunks after concat")
    parser.add_argument("--recursive","-r",action="store_true", help="Recurse subdirectories")
    parser.add_argument("--scan-only",     action="store_true", help="Probe only, no encoding")
    parser.add_argument("--dry-run",       action="store_true", help="Print commands, no execution")
    parser.add_argument("--verbose","-v",  action="store_true", help="Debug output")
    parser.add_argument("--list-backends", action="store_true", help="Show detected tools and exit")

    args = parser.parse_args()

    # ── Tool detection ────────────────────────────────────────────────────────
    tools = detect_tools(args.verbose)

    if args.list_backends:
        cprint(C.BOLD + C.WHITE, "\n  Detected tools and libraries:")
        for k, v in sorted(tools.items()):
            if k.startswith('py_'):
                status = f"{C.GREEN}✓{C.RESET}  installed" if v else f"{C.GREY}✗  not installed{C.RESET}"
            else:
                status = f"{C.GREEN}✓{C.RESET}  {v}" if v else f"{C.GREY}✗  not found{C.RESET}"
            print(f"    {k:20s} {status}")
        print()
        cprint(C.GREY, "  Install hints:")
        print("    pydub      pip install pydub")
        print("    soundfile  pip install soundfile")
        print("    sox        brew install sox  /  apt install sox  /  choco install sox")
        print("    lame       brew install lame  /  apt install lame  /  choco install lame")
        print("    vlc        https://www.videolan.org/vlc/")
        sys.exit(0)

    # ── ffmpeg check ──────────────────────────────────────────────────────────
    missing = [t for t in ('ffmpeg', 'ffprobe') if not shutil.which(t)]
    if missing:
        cprint(C.RED, f"\n[!] Missing: {', '.join(missing)}")
        sys = platform.system()
        if sys == "Darwin":  print("    brew install ffmpeg")
        elif sys == "Linux": print("    sudo apt install ffmpeg")
        else:                print("    winget install Gyan.FFmpeg  or  https://ffmpeg.org")
        import sys as _sys; _sys.exit(1)

    # ── Validate time args ────────────────────────────────────────────────────
    for name in ('from_time', 'to_time', 'trimfront', 'trimback'):
        val = getattr(args, name, None)
        if val:
            try: parse_time(val)
            except ValueError as e:
                cprint(C.RED, f"\n[!] --{name.replace('_','-')}: {e}"); sys.exit(1)

    # ── Encoders ──────────────────────────────────────────────────────────────
    encoders = detect_encoders(args.codec, args.verbose)
    if not encoders and not args.onlyaudio:
        cprint(C.RED, "\n[!] No usable video encoders found."); sys.exit(1)

    # ── Find files ────────────────────────────────────────────────────────────
    scan_path = os.path.expanduser(args.path)
    files = ([scan_path] if os.path.isfile(scan_path)
             else list(scan_directory(scan_path, args.recursive, args.verbose)))
    if not files:
        print(f"{C.YELLOW}[!] No supported video files found.{C.RESET}"); sys.exit(0)

    # ── Banner ────────────────────────────────────────────────────────────────
    enc_disp = (encoders or {}).get('video') or 'n/a'
    sw_note  = f"  {C.YELLOW}(software){C.RESET}" if (encoders or {}).get('is_software') else ""
    cprint(C.BOLD + C.WHITE, f"\n{'─'*62}")
    cprint(C.BOLD + C.WHITE,  "  vcompress.py — Cross-Platform Video Compressor")
    cprint(C.BOLD + C.WHITE, f"{'─'*62}")
    print(f"  Platform     : {platform.system()} {platform.machine()}"
          f"  [{(encoders or {}).get('label','?')}]")
    print(f"  Encoder      : {C.CYAN}{enc_disp}{C.RESET}{sw_note}")
    print(f"  Path         : {scan_path}")
    print(f"  Files        : {len(files)} candidate(s)")
    print(f"  Threshold    : {args.threshold} Mbps")
    print(f"  CRF          : {args.crf}")
    print(f"  Chunk size   : {args.chunk_minutes} min")
    print(f"  Video fmt    : {args.output_format}"
          + (f"  Audio fmt: {args.audio_format}" if not args.noaudio else ""))
    if args.audio_backend: print(f"  Audio backend: {args.audio_backend}")

    trim_parts = []
    if args.from_time:  trim_parts.append(f"from={args.from_time}")
    if args.to_time:    trim_parts.append(f"to={args.to_time}")
    if args.trimfront:  trim_parts.append(f"front={args.trimfront}")
    if args.trimback:   trim_parts.append(f"back={args.trimback}")
    if trim_parts: print(f"  Trim         : {', '.join(trim_parts)}")

    flags = [n for n, v in [
        ("DRY-RUN",    args.dry_run),    ("SCAN-ONLY", args.scan_only),
        ("RECURSIVE",  args.recursive),  ("VERBOSE",   args.verbose),
        ("KEEP-CHUNKS",args.keep_chunks),("OVERWRITE",  args.overwrite),
        ("PIPE",       args.pipe),       ("DIRECT-MP3", args.direct_mp3),
    ] if v]
    if flags: print(f"  Flags        : {', '.join(flags)}")
    cprint(C.BOLD + C.WHITE, f"{'─'*62}\n")

    # ── Scan phase ────────────────────────────────────────────────────────────
    to_process = []
    for filepath in files:
        fname = os.path.basename(filepath)
        print(f"{C.BOLD}[→] {fname}{C.RESET}")
        info = probe_file(filepath, args.verbose)
        if not info:
            print(f"  {C.RED}[!] Could not probe — skipping.{C.RESET}\n"); continue

        tag_str  = f"  [{info.audio_codec_tag}]" if info.audio_codec_tag.strip('0 \x00') else ""
        in24_str = f"  {C.YELLOW}[Atomos in24!]{C.RESET}" if info.is_in24 else ""

        print(f"  Codec    : {C.CYAN}{info.codec}{C.RESET}"
              + (f"  {C.YELLOW}[ProRes]{C.RESET}" if info.is_prores else "")
              + (f"  {C.YELLOW}[HDR]{C.RESET}"    if info.is_hdr    else ""))
        print(f"  Video    : {info.width}x{info.height} @ {info.fps:.2f}fps  |  {info.color_space}")
        print(f"  Bitrate  : {info.bitrate_mbps:.2f} Mbps  [{info.compression_label}]")
        print(f"  Duration : {info.duration_str}  |  {info.size_mb:.1f} MB")
        if info.has_audio:
            print(f"  Audio    : {info.audio_codec}{tag_str}  "
                  f"{info.audio_channels}ch  {info.audio_sample_rate}Hz  "
                  f"{info.audio_bitrate_kbps:.0f} kbps{in24_str}")
        else:
            print(f"  Audio    : {C.GREY}none{C.RESET}")

        if args.onlyaudio:
            print(f"  Decision : {C.YELLOW}AUDIO ONLY{C.RESET}")
            to_process.append(info)
        else:
            do_it, reason = should_compress(info, args.threshold, args.verbose)
            if do_it:
                print(f"  Decision : {C.YELLOW}COMPRESS{C.RESET}  — {reason}")
                to_process.append(info)
            else:
                print(f"  Decision : {C.GREEN}SKIP{C.RESET}  — {reason}")
        print()

    cprint(C.BOLD + C.WHITE, f"{'─'*62}")
    print(f"  {len(to_process)} of {len(files)} file(s) queued.")
    if not to_process or args.scan_only:
        if args.scan_only: cprint(C.GREY, "  [--scan-only] No encoding performed.")
        print(); sys.exit(0)
    cprint(C.BOLD + C.WHITE, f"{'─'*62}\n")

    # ── Process phase ─────────────────────────────────────────────────────────
    for i, info in enumerate(to_process, 1):
        cprint(C.BOLD + C.WHITE, f"[{i}/{len(to_process)}] {os.path.basename(info.path)}")
        cprint(C.GREY, f"  {info.size_mb:.1f} MB  |  {info.duration_str}  |  {info.bitrate_mbps:.1f} Mbps")
        try:
            if not args.noaudio:
                extract_audio(info, args, tools)
            if not args.onlyaudio:
                compress_video_chunked(info, args, encoders)
        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}[!] Interrupted — chunks saved, re-run to resume.{C.RESET}")
            sys.exit(130)

    cprint(C.BOLD + C.GREEN, "\n  ✓ All done.\n")

if __name__ == "__main__":
    main()