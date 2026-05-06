# svg2mp4

> Convert Animated SVG to MP4 Video

A Python CLI tool that renders animated SVGs into MP4 video files or PNG frame sequences.

## Features

- Renders SMIL animations (`<animate>`, `<animateTransform>`, `<animateMotion>`)
- Custom canvas size with background color padding
- Replace SVG background color at render time
- Extract individual frames as PNG instead of video
- Embedded font support — auto-converts WOFF2/WOFF to TTF for Cairo

## Requirements

- Python 3.8+
- `ffmpeg` (must be on `$PATH`)
- `fontconfig` + `fc-cache` (for embedded font rendering)

```bash
pip install -r requirements.txt
```

> Also requires `fonttools` for WOFF2 font conversion: `pip install fonttools brotli`

## Usage

```bash
python svg_to_video.py input.svg output.mp4 --duration 5 --fps 30
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--duration` | `2` | Video duration in seconds |
| `--fps` | `25` | Frames per second |
| `--width` | auto | SVG render width in pixels |
| `--height` | auto | SVG render height in pixels |
| `--bg-color` | none | Replace SVG background rect color (e.g. `#ffffff`) |
| `--canvas-w` | auto | Output canvas width — pads if larger than SVG |
| `--canvas-h` | auto | Output canvas height — pads if larger than SVG |
| `--canvas-color` | none | Canvas background color (e.g. `#1a1a2e`, `white`) |
| `--frames` | off | Save PNG frames to a directory instead of encoding video |

### Examples

```bash
# Basic render
python svg_to_video.py animation.svg out.mp4 --duration 3

# Excalidraw export with white canvas padding to 1920×1080
python svg_to_video.py drawing.svg out.mp4 --duration 4 --canvas-w 1920 --canvas-h 1080 --canvas-color white

# Extract frames
python svg_to_video.py animation.svg ./frames/ --duration 2 --fps 25 --frames
```

## How It Works

1. Parses the SVG with `lxml`
2. Walks SMIL animation elements and interpolates attribute values at each frame's timestamp
3. Rasterizes each frame to PNG via `cairosvg`
4. Encodes the frame sequence to MP4 using `moviepy` + `ffmpeg`

## License

MIT
