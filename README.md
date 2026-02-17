
# vcompress.py — Cross-Platform Video Compressor

`vcompress.py` is a simple video compression and (forensic) audio extraction tool. 

It attempts to recover audio from difficult sources (like Atomos recorders with corrupt headers or `in24`/`in32` PCM formats) and compressing high-bitrate footage using a **chunked encoding strategy** that supports resume capabilities.

## 🚀 Key Features

* **🛡️ Forensic Audio Recovery:** Features a multi-backend cascade (FFmpeg, Pipe, afconvert, VLC, SoX) to recover audio from corrupted containers. Specifically optimized for **Atomos `in24`/`in32`** PCM formats that often fail in standard NLEs.
* **🧩 Chunked Encoding:** Splits long videos into temporary chunks for encoding. If the process crashes, you only lose the current chunk, not the whole encode.
* **⚡ Hardware Acceleration:** Auto-detects and uses GPU acceleration:
    * **macOS:** Apple VideoToolbox (ProRes, H.264, HEVC)
    * **Windows:** NVIDIA NVENC, AMD AMF, Intel QSV
    * **Linux:** NVENC, VAAPI
* **🎛️ Precision Trimming:** Supports flexible time formats (e.g., `1:30`, `90s`, `1h20m`) for start/end points and front/back trimming.
* **📂 Folders Specifications:** Support for specific output directories and temporary folders (e.g. for network drives or NVMe scratch disks).

---

## 🛠️ Installation

### 1. Requirements
* **Python 3.7+**
* **FFmpeg** (Required)

### 2. Install Dependencies
You can run the script with just the standard library, but for full functionality:

**macOS:**
```bash
brew install ffmpeg sox
# VLC is optional but recommended for fallback
brew install --cask vlc

```

**Windows:**

```powershell
winget install Gyan.FFmpeg
# Optional: Install VLC and SoX manually

```

**Linux (Ubuntu/Debian):**

```bash
sudo apt update && sudo apt install ffmpeg sox libsox-fmt-all

```

**Python Libraries (Optional):**

```bash
pip install pydub

```

---

## 📖 Usage Examples

### Basic Video Compression

Compress all videos in the current directory to H.264 MP4 (default ~25Mbps threshold):

```bash
python vcompress.py .

```

### Forensic Audio Extraction (The "Nuclear" Option)

If you have a corrupt file or an Atomos recording that won't open in Premiere/Resolve:

```bash
# Attempt to extract audio using all available backends, including raw stream piping
python vcompress.py input.mov --onlyaudio --forensic

```

### High-Efficiency Encoding (HEVC/H.265)

Use HEVC codec with a specific quality factor (CRF):

```bash
python vcompress.py . --codec hevc --crf 26

```

### Managing Storage Paths (New in v5.1)

Read from a slow network drive, write temporary chunks to a fast local SSD, and save the final result to a different folder:

```bash
python vcompress.py /Volumes/SlowServer/Footage \
  --tmpdir /tmp/fast_nvme \
  --outdir ~/Desktop/Finals

```

### Trimming

Cut from 1 minute 30 seconds to 5 minutes:

```bash
python vcompress.py video.mov --from 1:30 --to 5:00

```

Remove the first 10 seconds and the last 30 seconds:

```bash
python vcompress.py video.mov --trimfront 10s --trimback 30s

```

---

## 🔧 Audio Backends

The script employs a "waterfall" strategy for audio extraction. If one fails, it moves to the next. You can also force a specific backend:

| Backend | Description | Best For |
| --- | --- | --- |
| **`ffmpeg`** | Standard FFmpeg with Atomos patches. | General use. |
| **`afconvert`** | macOS CoreAudio CLI. | **Atomos in24/in32** files on macOS. |
| **`pipe`** | Raw PCM piping (bypasses container). | Corrupt container headers. |
| **`vlc`** | VLC Transcode Engine. | Stubborn formats VLC plays but FFmpeg rejects. |
| **`forensic`** | OS-level binary stream (`cat` / `type`). | severely corrupted files. |

**Force a backend:**

```bash
python vcompress.py video.mov --onlyaudio --audio-backend afconvert

```

---

## ⚙️ Command Line Arguments

| Argument | Description |
| --- | --- |
| `path` | File or directory to scan. |
| **Video Options** |  |
| `--threshold` | Bitrate (Mbps) trigger. Files below this are skipped. Default: 25. |
| `--crf` | Quality (0-51). Lower is better. Default: 23. |
| `--codec` | `h264` or `hevc`. |
| `--chunk-minutes` | Length of encoding chunks. Default: 10 mins. |
| **Audio Options** |  |
| `--onlyaudio` | Extract audio only; skip video compression. |
| `--audio-format` | `mp3`, `aac`, `wav`, `flac`, `m4a`, `opus`, `ogg`. |
| `--forensic` | Enable aggressive recovery strategies. |
| **Paths** |  |
| `--outdir` | Override output directory. |
| `--tmpdir` | Override temporary directory for chunks. |
| **General** |  |
| `--dry-run` | Print commands without executing. |
| `-v`, `-vv`, `-vvv` | Verbosity levels. |

---

## 📄 License

**MIT License**

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
