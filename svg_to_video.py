import os.path
import re
import sys
import io
import math
import logging
import numpy as np
import moviepy.video.io.ImageSequenceClip as ImageSequenceClip
from PIL import Image, ImageColor
from cairosvg import svg2png
from lxml import etree
from tqdm import tqdm
import base64
import subprocess
import tempfile

# Suppress moviepy / imageio / ffmpeg log noise
logging.getLogger("moviepy").setLevel(logging.ERROR)
logging.getLogger("imageio").setLevel(logging.ERROR)
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)

XLINK_NS = "http://www.w3.org/1999/xlink"
NUM_RE = re.compile(r'-?[\d.]+(?:e[+-]?\d+)?')
ANIMATE_XPATH = "//*[local-name()='animate' or local-name()='animateTransform' or local-name()='animateMotion']"


def to_seconds(s):
    s = s.strip()
    if not s or s == "indefinite":
        return 0
    if s.endswith("ms"):   return float(s[:-2]) / 1000
    if s.endswith("min"):  return float(s[:-3]) * 60
    if s.endswith("h"):    return float(s[:-1]) * 3600
    if s.endswith("m"):    return float(s[:-1]) * 60
    if s.endswith("s"):    return float(s[:-1])
    return float(s)


def lerp_path(a, b, t):
    a_nums = NUM_RE.findall(a)
    b_nums = NUM_RE.findall(b)
    if len(a_nums) != len(b_nums):
        return b if t >= 0.5 else a
    blended = [str(float(x) + (float(y) - float(x)) * t) for x, y in zip(a_nums, b_nums)]
    parts, idx = NUM_RE.split(a), 0
    out = []
    for i, part in enumerate(parts):
        out.append(part)
        if idx < len(blended):
            out.append(blended[idx])
            idx += 1
    return ''.join(out)


def resolve_smil(root, t):
    """
    Bake SMIL animation state at time t into the DOM.

    When multiple chained <animate> elements target the same attribute and all
    have completed (elapsed >= dur, fill="freeze"), apply only the one with the
    greatest begin time — that carries the final intended value.
    """
    from collections import defaultdict

    freeze_candidates = defaultdict(list)  # (id(parent), attr) -> [(begin, to, parent, attr)]

    for anim in root.xpath(ANIMATE_XPATH):
        dur_str = anim.get("dur", "").strip()
        if not dur_str or dur_str == "indefinite":
            continue
        dur = to_seconds(dur_str)
        if dur == 0:
            continue

        begin = to_seconds(anim.get("begin", "0"))
        if t < begin:
            continue

        parent = anim.getparent()
        if parent is None:
            continue

        elapsed = t - begin
        attr  = anim.get("attributeName", "")
        from_ = anim.get("from", "")
        to_   = anim.get("to", "")

        if elapsed >= dur:
            if anim.get("fill") == "freeze" and to_:
                freeze_candidates[(id(parent), attr)].append((begin, to_, parent, attr))
            continue

        # In-progress: interpolate
        progress = elapsed / dur

        if attr == "d":
            parent.set("d", lerp_path(from_, to_, progress))
        elif attr == "opacity":
            try:
                f = float(from_) if from_ else float(parent.get("opacity", "1"))
                parent.set("opacity", str(f + (float(to_) - f) * progress))
            except ValueError:
                pass
        elif attr == "transform" and anim.get("type") == "translate":
            values    = anim.get("values", "").split(";")
            key_times = anim.get("keyTimes", "")
            if not key_times:
                continue
            kt = list(map(float, key_times.split(";")))
            for i in range(len(kt) - 1):
                if kt[i] <= progress < kt[i + 1]:
                    sx, sy = map(float, values[i].split())
                    ex, ey = map(float, values[i + 1].split())
                    seg = (progress - kt[i]) / (kt[i + 1] - kt[i])
                    parent.set("transform", f"translate({sx + (ex-sx)*seg}, {sy + (ey-sy)*seg})")
                    break

    # Apply only the latest-begin freeze value per (parent, attr) group
    for candidates in freeze_candidates.values():
        _, to_val, parent, attr = max(candidates, key=lambda x: x[0])
        parent.set(attr, to_val)


def patch_xlink_refs(root):
    for tp in root.xpath("//*[local-name()='textPath']"):
        href = tp.get("href")
        if href and not tp.get(f"{{{XLINK_NS}}}href"):
            tp.set(f"{{{XLINK_NS}}}href", href)


def widen_textpaths(root, scale=2.0):
    refs = set()
    for tp in root.xpath("//*[local-name()='textPath']"):
        for attr in ("href", f"{{{XLINK_NS}}}href"):
            v = tp.get(attr, "")
            if v.startswith("#"):
                refs.add(v[1:])

    for path in root.xpath("//*[local-name()='path']"):
        if path.get("id", "") not in refs:
            continue
        d = path.get("d", "")
        if d:
            path.set("d", re.sub(r'h([\d.]+)', lambda m: f"h{float(m.group(1)) * scale}", d))


def detach_filter(root):
    style = root.get("style", "")
    m = re.search(r'filter\s*:\s*([^;]+)', style)
    if not m:
        return None
    result = m.group(1).strip()
    cleaned = re.sub(r'filter\s*:[^;]+;?\s*', '', style).strip()
    if cleaned:
        root.set("style", cleaned)
    else:
        del root.attrib["style"]
    return result


def shift_hue(rgb, degrees):
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    mat = np.array([
        [0.213 + c*0.787 - s*0.213,  0.715 - c*0.715 - s*0.715,  0.072 - c*0.072 + s*0.928],
        [0.213 - c*0.213 + s*0.143,  0.715 + c*0.285 + s*0.140,  0.072 - c*0.072 - s*0.283],
        [0.213 - c*0.213 - s*0.787,  0.715 - c*0.715 + s*0.715,  0.072 + c*0.928 + s*0.072],
    ], dtype=np.float32)
    h, w = rgb.shape[:2]
    return np.clip((rgb.reshape(-1, 3) @ mat.T).reshape(h, w, 3), 0.0, 1.0)


def apply_filter(arr, filter_str):
    img = arr.astype(np.float32) / 255.0
    for token in re.findall(r'[\w-]+\([^)]+\)', filter_str):
        name = token[:token.index('(')]
        val  = token[token.index('(') + 1:-1].strip()
        if name == 'invert':
            p = float(val.rstrip('%')) / (100 if '%' in val else 1)
            img = np.clip(img * (1 - 2*p) + p, 0.0, 1.0)
        elif name == 'hue-rotate':
            img = shift_hue(img, float(NUM_RE.search(val).group()))
    return (img * 255).astype(np.uint8)


def parse_color_hex(color_str):
    """
    Parse a color string into an (R, G, B) tuple of ints 0-255.
    Accepts:
      - Hex codes with or without '#': '#ff0000', 'ff0000', '#f00', 'f00'
      - CSS named colors: 'red', 'white', 'transparent' (→ black)
    Raises ValueError for unrecognised values.
    """
    s = color_str.strip()
    if not s.startswith('#'):
        # Try as a named colour first; if it looks like a bare hex, prepend '#'
        if re.fullmatch(r'[0-9a-fA-F]{3}|[0-9a-fA-F]{6}', s):
            s = '#' + s
    try:
        return ImageColor.getrgb(s)[:3]   # drop alpha if present
    except (ValueError, AttributeError):
        raise ValueError(
            f"Unrecognised canvas_color '{color_str}'. "
            "Use a hex code (e.g. '#1a2b3c' or '1a2b3c') or a CSS colour name (e.g. 'red')."
        )


def composite_onto_canvas(frame_rgba, canvas_w, canvas_h, canvas_color):
    """
    Alpha-composite *frame_rgba* (H×W×4 uint8 RGBA) onto a canvas_w×canvas_h
    background filled with *canvas_color* (hex string or CSS name).
    Returns a uint8 RGB array of shape (canvas_h, canvas_w, 3).
    """
    r, g, b = parse_color_hex(canvas_color)
    # Build canvas as float32 for blending
    canvas = np.full((canvas_h, canvas_w, 3), [r, g, b], dtype=np.float32)

    fh, fw = frame_rgba.shape[:2]

    src_y0 = max(0, (fh - canvas_h) // 2)
    src_x0 = max(0, (fw - canvas_w) // 2)
    crop_h = min(fh, canvas_h)
    crop_w = min(fw, canvas_w)

    dst_y0 = max(0, (canvas_h - fh) // 2)
    dst_x0 = max(0, (canvas_w - fw) // 2)

    patch = frame_rgba[src_y0:src_y0 + crop_h, src_x0:src_x0 + crop_w]
    rgb   = patch[:, :, :3].astype(np.float32)
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0   # shape (h, w, 1)

    region = canvas[dst_y0:dst_y0 + crop_h, dst_x0:dst_x0 + crop_w]
    canvas[dst_y0:dst_y0 + crop_h, dst_x0:dst_x0 + crop_w] = (
        rgb * alpha + region * (1.0 - alpha)
    )

    return canvas.astype(np.uint8)


# ---------------------------------------------------------------------------
# Font patching — convert embedded woff2 → TTF so cairosvg can render them
# ---------------------------------------------------------------------------
_FONT_FACE_RE = re.compile(
    r"@font-face\s*\{([^}]+)\}", re.DOTALL
)
_SRC_DATA_RE = re.compile(
    r'src\s*:[^;]*url\("?(data:font/[^;]+);base64,([A-Za-z0-9+/=]+)"?\)',
    re.DOTALL,
)
_FAMILY_RE = re.compile(r'font-family\s*:\s*[\'"]?([^;\'"]+)[\'"]?')

_installed_families: set = set()   # track what we've already registered


def _woff2_to_ttf_bytes(woff2_bytes: bytes) -> bytes:
    """Convert a WOFF2 blob to TTF bytes using fontTools (in-memory)."""
    from fontTools.ttLib import TTFont
    buf_in  = io.BytesIO(woff2_bytes)
    buf_out = io.BytesIO()
    font = TTFont(buf_in)
    font.flavor = None          # strip WOFF2 wrapper → plain TTF/OTF
    font.save(buf_out)
    return buf_out.getvalue()


def _woff_to_ttf_bytes(woff_bytes: bytes) -> bytes:
    """Convert a WOFF1 blob to TTF bytes using fontTools (in-memory)."""
    from fontTools.ttLib import TTFont
    buf_in  = io.BytesIO(woff_bytes)
    buf_out = io.BytesIO()
    font = TTFont(buf_in)
    font.flavor = None
    font.save(buf_out)
    return buf_out.getvalue()


def _install_font(ttf_bytes: bytes, family: str) -> None:
    """Write TTF to a temp dir and register it with fontconfig so Cairo finds it."""
    font_dir = os.path.join(tempfile.gettempdir(), "svg_exporter_fonts")
    os.makedirs(font_dir, exist_ok=True)
    safe_name = re.sub(r"[^\w-]", "_", family)
    ttf_path  = os.path.join(font_dir, f"{safe_name}.ttf")
    with open(ttf_path, "wb") as fh:
        fh.write(ttf_bytes)
    # Tell fontconfig about this directory (safe to call repeatedly)
    fc_conf = os.path.join(font_dir, "fonts.conf")
    if not os.path.exists(fc_conf):
        with open(fc_conf, "w") as fh:
            fh.write(f'''<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig><dir>{font_dir}</dir></fontconfig>''')
    os.environ.setdefault("FONTCONFIG_FILE", fc_conf)
    subprocess.run(["fc-cache", "-f", font_dir],
                   capture_output=True, check=False)


def patch_fonts(root) -> None:
    """
    Find every @font-face in the SVG <style> blocks that carries a
    data-URI font (woff2 or woff).  Convert each to TTF, install it
    system-wide for Cairo, and rewrite the data-URI to TTF so cairosvg
    can also load it directly.
    """
    for style_el in root.xpath("//*[local-name()='style']"):
        css = style_el.text or ""
        new_css = css

        for face_match in _FONT_FACE_RE.finditer(css):
            face_body = face_match.group(1)

            family_m = _FAMILY_RE.search(face_body)
            src_m    = _SRC_DATA_RE.search(face_body)
            if not family_m or not src_m:
                continue

            family    = family_m.group(1).strip()
            mime      = src_m.group(1)          # e.g. data:font/woff2
            b64_data  = src_m.group(2)

            if family in _installed_families:
                continue

            try:
                raw = base64.b64decode(b64_data)

                if "woff2" in mime:
                    ttf_bytes = _woff2_to_ttf_bytes(raw)
                elif "woff" in mime:
                    ttf_bytes = _woff_to_ttf_bytes(raw)
                else:
                    continue    # already ttf/otf — leave as-is

                _install_font(ttf_bytes, family)
                _installed_families.add(family)

                # Rewrite the src line to use TTF so cairosvg can parse it
                ttf_b64  = base64.b64encode(ttf_bytes).decode()
                new_src  = f'src: url("data:font/ttf;base64,{ttf_b64}") format("truetype")'
                old_src  = face_body[src_m.start():src_m.end()]
                new_body = face_body.replace(old_src, new_src)
                new_css  = new_css.replace(face_body, new_body)

            except Exception as exc:
                print(f"[font] Could not convert font '{family}': {exc}", file=sys.stderr)

        style_el.text = new_css



def fix_nested_svg_overflow(root):
    """
    cairosvg clips nested <svg> elements to their width/height exactly.
    Setting overflow="visible" on every nested <svg> disables that clipping
    so strokes on outer edges render correctly.
    """
    for el in root.iter():
        # lxml yields comment/PI nodes whose .tag is a callable, not a string
        if not isinstance(el.tag, str):
            continue
        local = el.tag.split('}')[-1] if '}' in el.tag else el.tag
        if local == 'svg' and el is not root:
            el.set('overflow', 'visible')


def rasterize(svg_bytes, t, width=None, height=None, bg_color=None,
              canvas_w=None, canvas_h=None, canvas_color=None):
    root = etree.fromstring(svg_bytes)

    css_filter = detach_filter(root)
    patch_xlink_refs(root)
    patch_fonts(root)
    fix_nested_svg_overflow(root)

    if bg_color:
        bg = next((r for r in root.xpath("//*[local-name()='rect']")
                   if r.get("x", "0") in ("0", "0.0") and r.get("y", "0") in ("0", "0.0")), None)
        if bg is not None:
            bg.set("fill", bg_color)

    resolve_smil(root, t)
    widen_textpaths(root)

    kwargs = {k: v for k, v in [("output_width", width), ("output_height", height)] if v}
    png = svg2png(bytestring=etree.tostring(root), **kwargs)

    arr = np.array(Image.open(io.BytesIO(png)).convert("RGBA"))
    if css_filter:
        rgb = apply_filter(arr[:, :, :3], css_filter)
        arr = np.dstack([rgb, arr[:, :, 3]])   # reattach alpha

    if canvas_color is not None:
        fh, fw = arr.shape[:2]
        cw = canvas_w if canvas_w is not None else fw
        ch = canvas_h if canvas_h is not None else fh
        return composite_onto_canvas(arr, cw, ch, canvas_color)

    # No canvas colour — return plain RGB (drop alpha)
    return arr[:, :, :3]


def export(input_svg, output_path, duration=10, fps=25, width=None, height=None,
           bg_color=None, frames_only=False, canvas_w=None, canvas_h=None,
           canvas_color=None):
    svg_bytes = open(input_svg, 'rb').read()
    total = int(duration * fps)
    frames = []

    label = "Rendering frames" if not frames_only else "Saving frames"
    with tqdm(total=total, desc=label, unit="frame",
              bar_format="{desc}: {n_fmt}/{total_fmt} [{bar}] {percentage:3.0f}%  {rate_fmt}  ETA {remaining}") as bar:
        for n in range(total):
            frame = rasterize(
                svg_bytes, n / fps,
                width=width, height=height,
                bg_color=bg_color,
                canvas_w=canvas_w, canvas_h=canvas_h, canvas_color=canvas_color,
            )
            if frames_only:
                Image.fromarray(frame).save(os.path.join(output_path, f'frame_{n:04d}.png'))
            else:
                frames.append(frame)
            bar.update(1)

    if not frames_only:
        print(f"\nEncoding video → {output_path}")
        clip = ImageSequenceClip.ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(output_path, codec='libx264', logger=None)
        print("Done.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Export an animated SVG to video or frames.")
    parser.add_argument("input",           help="Path to the input SVG file")
    parser.add_argument("output",          help="Output video file or frames directory")
    parser.add_argument("--duration",      type=int,   default=2,    help="Duration in seconds (default: 2)")
    parser.add_argument("--fps",           type=int,   default=25,   help="Frames per second (default: 25)")
    parser.add_argument("--width",         type=int,   default=None, help="SVG render width in pixels")
    parser.add_argument("--height",        type=int,   default=None, help="SVG render height in pixels")
    parser.add_argument("--bg-color",      default=None, help="Replace the SVG background rect fill (e.g. '#ff0000')")
    parser.add_argument("--frames",        action="store_true",      help="Save individual frames instead of a video")
    parser.add_argument("--canvas-w",      type=int,   default=None, help="Output canvas width  — pads with canvas-color if larger than SVG")
    parser.add_argument("--canvas-h",      type=int,   default=None, help="Output canvas height — pads with canvas-color if larger than SVG")
    parser.add_argument("--canvas-color",  default=None, help="Canvas background colour, hex or CSS name (e.g. '#1a1a2e', 'red', 'ff0000')")

    a = parser.parse_args()

    if a.frames and not os.path.isdir(a.output):
        os.mkdir(a.output)

    export(
        a.input, a.output,
        duration=a.duration, fps=a.fps,
        width=a.width, height=a.height,
        bg_color=a.bg_color, frames_only=a.frames,
        canvas_w=a.canvas_w, canvas_h=a.canvas_h, canvas_color=a.canvas_color,
    )