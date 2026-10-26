#!/usr/bin/env python3
import os
import sys
import glob
import json
import time
import shutil
import getopt
import threading
import traceback
import webbrowser
from threading import Timer
from pathlib import Path

# Auto-configure search paths so backend.py runs directly with system python without needing to activate a virtual environment
PROJECT_DIR = Path(__file__).resolve().parent

candidate_site_packages = [
    PROJECT_DIR / "packages",
    PROJECT_DIR / "local_packages",
]
if (PROJECT_DIR / "venv/lib").exists():
    for p in (PROJECT_DIR / "venv/lib").glob("python*/site-packages"):
        candidate_site_packages.append(p)
if (PROJECT_DIR / "env/lib").exists():
    for p in (PROJECT_DIR / "env/lib").glob("python*/site-packages"):
        candidate_site_packages.append(p)

for sdir in candidate_site_packages:
    if sdir.exists() and str(sdir) not in sys.path:
        sys.path.insert(0, str(sdir))

# Preload local shared libraries (libagg, libpotrace) if not installed system-wide
import ctypes
lib_search_dirs = [
    PROJECT_DIR / "local_libs/usr/lib/x86_64-linux-gnu",
    PROJECT_DIR / "local_libs/usr/lib",
    PROJECT_DIR / "local_libs/lib",
]
for ldir in lib_search_dirs:
    for soname in ["libagg.so.2", "libpotrace.so.0"]:
        so_file = ldir / soname
        if so_file.exists():
            try:
                ctypes.CDLL(str(so_file))
            except OSError:
                pass

from flask import Flask, request, render_template, jsonify, send_file, redirect
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import potrace
import cv2

app = Flask(__name__, template_folder='frontend', static_folder='frontend')
CORS(app)
PORT = 5000

# Base Configuration
FRAME_DIR = 'frames'
FILE_EXT = 'png'
COLOUR = '#2464b4'
SCREENSHOT_SIZE = [None, None]
SCREENSHOT_FORMAT = 'png'
OPEN_BROWSER = False
BILATERAL_FILTER = False
DOWNLOAD_IMAGES = False
USE_L2_GRADIENT = False
SHOW_GRID = False
CANNY_LOW = 30
CANNY_HIGH = 200
PRECISION = 2  # Decimal places for LaTeX equations

# Thread-safe Frame Cache
frame_cache = {}
cache_lock = threading.Lock()

# Dimensions state
image_width = 480
image_height = 270


def get_sorted_frame_files():
    """Discover and sort frame files naturally."""
    if not os.path.exists(FRAME_DIR):
        os.makedirs(FRAME_DIR, exist_ok=True)
        return []

    # Check for frame%d.%ext first
    candidates = []
    for ext in ('png', 'jpg', 'jpeg', 'webp', 'bmp'):
        candidates.extend(glob.glob(os.path.join(FRAME_DIR, f'*.{ext}')))
        candidates.extend(glob.glob(os.path.join(FRAME_DIR, f'*.{ext.upper()}')))

    def natural_sort_key(s):
        import re
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    return sorted(candidates, key=natural_sort_key)


def update_dimensions_from_files():
    """Read dimensions from the first available frame."""
    global image_width, image_height
    files = get_sorted_frame_files()
    if files:
        img = cv2.imread(files[0])
        if img is not None:
            image_height, image_width = img.shape[:2]


def get_contours(filename, nudge=0.33, bilateral=None, use_l2=None, canny_l=None, canny_h=None):
    """Perform Canny edge detection on an image file."""
    global image_width, image_height
    image = cv2.imread(filename)
    if image is None:
        return np.zeros((10, 10), dtype=np.uint8)

    h, w = image.shape[:2]
    image_height = max(image_height, h)
    image_width = max(image_width, w)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    use_bilateral = BILATERAL_FILTER if bilateral is None else bilateral
    use_l2_grad = USE_L2_GRADIENT if use_l2 is None else use_l2
    c_low = CANNY_LOW if canny_l is None else canny_l
    c_high = CANNY_HIGH if canny_h is None else canny_h

    if use_bilateral:
        median = max(10, min(245, float(np.median(gray))))
        lower = int(max(0, (1 - nudge) * median))
        upper = int(min(255, (1 + nudge) * median))
        filtered = cv2.bilateralFilter(gray, 5, 50, 50)
        edged = cv2.Canny(filtered, lower, upper, L2gradient=use_l2_grad)
    else:
        edged = cv2.Canny(gray, c_low, c_high, L2gradient=use_l2_grad)

    # Invert Y axis so bottom-left is origin (0, 0) in Desmos Cartesian coordinates
    return edged[::-1]


def get_trace(data):
    """Trace contours using Potrace into Bezier curves."""
    binary = np.ascontiguousarray((data > 0).astype(np.uint32))
    bmp = potrace.Bitmap(binary)
    return bmp.trace(2, potrace.TURNPOLICY_MINORITY, 1.0, 1, 0.5)


def get_latex(filename, bilateral=None, use_l2=None, canny_l=None, canny_h=None):
    """Generate Desmos parametric Bezier curve equations from an image."""
    latex = []
    contours = get_contours(filename, bilateral=bilateral, use_l2=use_l2, canny_l=canny_l, canny_h=canny_h)
    path = get_trace(contours)

    p = PRECISION
    fmt_corner = f'((1-t)%.{p}f+t%.{p}f,(1-t)%.{p}f+t%.{p}f)'
    fmt_cubic = (
        f'((1-t)((1-t)((1-t)%.{p}f+t%.{p}f)+t((1-t)%.{p}f+t%.{p}f))+t((1-t)((1-t)%.{p}f+t%.{p}f)+t((1-t)%.{p}f+t%.{p}f)),'
        f'(1-t)((1-t)((1-t)%.{p}f+t%.{p}f)+t((1-t)%.{p}f+t%.{p}f))+t((1-t)((1-t)%.{p}f+t%.{p}f)+t((1-t)%.{p}f+t%.{p}f)))'
    )

    for curve in path.curves:
        segments = curve.segments
        start = curve.start_point
        for segment in segments:
            x0, y0 = start
            if segment.is_corner:
                x1, y1 = segment.c
                x2, y2 = segment.end_point
                latex.append((fmt_corner % (x0, x1, y0, y1), [round(x1, p), round(y1, p)]))
                latex.append((fmt_corner % (x1, x2, y1, y2), [round(x2, p), round(y2, p)]))
            else:
                x1, y1 = segment.c1
                x2, y2 = segment.c2
                x3, y3 = segment.end_point
                latex.append((fmt_cubic % (
                    x0, x1, x1, x2, x1, x2, x2, x3,
                    y0, y1, y1, y2, y1, y2, y2, y3
                ), [round(x3, p), round(y3, p)]))
            start = segment.end_point
    return latex


def get_frame_expressions(frame_index_0_based, color=None):
    """Get or compute expressions for a specific frame."""
    files = get_sorted_frame_files()
    if not files or frame_index_0_based < 0 or frame_index_0_based >= len(files):
        return None

    filename = files[frame_index_0_based]
    use_color = COLOUR if color is None else color

    with cache_lock:
        if filename in frame_cache:
            raw_latex = frame_cache[filename]
        else:
            raw_latex = get_latex(filename)
            frame_cache[filename] = raw_latex

    exprs = []
    for i, item in enumerate(raw_latex, start=1):
        if isinstance(item, (list, tuple)):
            eq, pt = item
        else:
            eq, pt = item, [0, 0]
        exprs.append({
            'id': f'expr-{i}',
            'latex': eq,
            'color': use_color,
            'secret': True,
            'pt': pt
        })
    return exprs


# Background preloader for smooth playback
def preload_adjacent_frames(current_idx_0_based):
    def worker():
        files = get_sorted_frame_files()
        total = len(files)
        for offset in (1, 2, 3, -1):
            target = current_idx_0_based + offset
            if 0 <= target < total:
                target_file = files[target]
                with cache_lock:
                    if target_file not in frame_cache:
                        try:
                            frame_cache[target_file] = get_latex(target_file)
                        except Exception:
                            pass
    threading.Thread(target=worker, daemon=True).start()


# ================== HTTP ROUTES ==================

@app.route('/')
def root():
    # Backward compatibility: ?frame=<frame_num>
    if 'frame' in request.args:
        try:
            frame_idx = int(request.args.get('frame'))
            exprs = get_frame_expressions(frame_idx)
            return jsonify({'result': exprs})
        except Exception as e:
            return jsonify({'result': None, 'error': str(e)})
    return redirect('/calculator')


@app.route('/calculator')
def calculator():
    update_dimensions_from_files()
    files = get_sorted_frame_files()
    return render_template(
        'index.html',
        api_key='dcb31709b452b1cf9dc26972add0fda6',
        height=image_height,
        width=image_width,
        total_frames=len(files),
        download_images=DOWNLOAD_IMAGES,
        show_grid=SHOW_GRID,
        screenshot_size=SCREENSHOT_SIZE,
        screenshot_format=SCREENSHOT_FORMAT,
        default_color=COLOUR
    )


@app.route('/api/info')
def api_info():
    update_dimensions_from_files()
    files = get_sorted_frame_files()
    return jsonify({
        'total_frames': len(files),
        'width': image_width,
        'height': image_height,
        'current_settings': {
            'color': COLOUR,
            'bilateral': BILATERAL_FILTER,
            'use_l2_gradient': USE_L2_GRADIENT,
            'show_grid': SHOW_GRID,
            'canny_low': CANNY_LOW,
            'canny_high': CANNY_HIGH,
            'screenshot_format': SCREENSHOT_FORMAT
        },
        'cached_count': len(frame_cache)
    })


@app.route('/api/frame/<int:frame_num>')
@app.route('/api/frame')
def api_frame(frame_num=None):
    if frame_num is None:
        frame_num = int(request.args.get('frame', 1))

    # Convert 1-indexed to 0-indexed
    idx_0 = max(0, frame_num - 1)
    color = request.args.get('color', COLOUR)

    exprs = get_frame_expressions(idx_0, color=color)
    files = get_sorted_frame_files()
    total = len(files)

    if exprs is None:
        return jsonify({'success': False, 'error': f'Frame {frame_num} not found', 'result': []}), 404

    # Trigger async preloading for upcoming frames
    preload_adjacent_frames(idx_0)

    return jsonify({
        'success': True,
        'frame': frame_num,
        'total_frames': total,
        'count': len(exprs),
        'width': image_width,
        'height': image_height,
        'result': exprs
    })


@app.route('/api/export/<int:frame_num>')
@app.route('/api/export')
def api_export(frame_num=None):
    """Export mathematical equations in various formats for official Desmos."""
    if frame_num is None:
        frame_num = int(request.args.get('frame', 1))

    idx_0 = max(0, frame_num - 1)
    color = request.args.get('color', COLOUR)
    fmt = request.args.get('format', 'json').lower()

    exprs = get_frame_expressions(idx_0, color=color)
    files = get_sorted_frame_files()

    if exprs is None:
        return jsonify({'success': False, 'error': f'Frame {frame_num} not found'}), 404

    equations = [e['latex'] for e in exprs]
    w, h = image_width, image_height

    # Construct standard Desmos Calculator State format
    desmos_state = {
        "version": 11,
        "graph": {
            "viewport": {
                "xmin": 0,
                "xmax": w,
                "ymin": 0,
                "ymax": h
            },
            "showGrid": False,
            "showXAxis": False,
            "showYAxis": False
        },
        "expressions": {
            "list": [
                {
                    "type": "folder",
                    "id": "bezier_folder",
                    "title": f"Bézier Art ({len(equations)} đường cong)",
                    "collapsed": True
                }
            ] + [
                {
                    "type": "expression",
                    "id": f"curve_{i}",
                    "latex": eq,
                    "color": color,
                    "folderId": "bezier_folder"
                } for i, eq in enumerate(equations, 1)
            ]
        }
    }

    # Generate 1-click injection script for browser DevTools on https://www.desmos.com/calculator
    js_script = (
        f"// ==========================================\n"
        f"// Desmos Bézier Art Injection Script (Hiện tất cả)\n"
        f"// Chạy script này trong tab Console của https://www.desmos.com/calculator\n"
        f"// ==========================================\n"
        f"(function() {{\n"
        f"  if (typeof Calc === 'undefined') {{\n"
        f"    alert('Vui lòng mở trang https://www.desmos.com/calculator trước khi chạy mã này!');\n"
        f"    return;\n"
        f"  }}\n"
        f"  const state = {json.dumps(desmos_state)};\n"
        f"  Calc.setState(state, {{ allowUndo: false }});\n"
        f"  Calc.setMathBounds({{ left: 0, right: {w}, bottom: 0, top: {h} }});\n"
        f"  console.log('%c✓ Đã vẽ thành công {len(equations)} đường cong Bézier vào Desmos!', 'color: #10b981; font-weight: bold; font-size: 14px;');\n"
        f"}})();"
    )

    # Generate progressive live-sketch animation script for Desmos.com
    live_items = [{'latex': e['latex'], 'pt': e.get('pt', [0, 0])} for e in exprs]
    live_script = (
        f"// ==========================================\n"
        f"// Desmos Bézier Art - Chế độ Hoạt Họa Trình Diễn (Live Sketch Demo)\n"
        f"// Từng phương trình được vẽ lại đồng bộ với bảng công thức Desmos\n"
        f"// Chạy script này trong tab Console của https://www.desmos.com/calculator\n"
        f"// ==========================================\n"
        f"(function() {{\n"
        f"  if (typeof Calc === 'undefined') {{\n"
        f"    alert('Vui lòng mở trang https://www.desmos.com/calculator trước khi chạy mã này!');\n"
        f"    return;\n"
        f"  }}\n"
        f"  const width = {w};\n"
        f"  const height = {h};\n"
        f"  const color = '{color}';\n"
        f"  const batchSize = 10; // Số nét vẽ mỗi bước (Tăng/giảm để đổi tốc độ)\n"
        f"  const stepDelay = 25; // Độ trễ giữa các nét (ms)\n"
        f"  const curves = {json.dumps(live_items)};\n"
        f"  Calc.setMathBounds({{ left: 0, right: width, bottom: 0, top: height }});\n"
        f"  Calc.updateSettings({{ showGrid: false, showXAxis: false, showYAxis: false, expressions: true }});\n"
        f"  Calc.setExpression({{\n"
        f"    type: 'folder',\n"
        f"    id: 'live_art_folder',\n"
        f"    title: 'Bézier Live Art (' + curves.length + ' đường cong)',\n"
        f"    collapsed: false\n"
        f"  }});\n"
        f"  let idx = 0;\n"
        f"  console.log('%c🎨 Đang trình diễn vẽ tay ' + curves.length + ' đường cong...', 'color: #38bdf8; font-weight: bold; font-size: 14px;');\n"
        f"  const timer = setInterval(() => {{\n"
        f"    const next = Math.min(idx + batchSize, curves.length);\n"
        f"    const batch = [];\n"
        f"    for (let i = idx; i < next; i++) {{\n"
        f"      batch.push({{\n"
        f"        type: 'expression',\n"
        f"        id: 'curve_' + (i + 1),\n"
        f"        latex: curves[i].latex,\n"
        f"        color: color,\n"
        f"        folderId: 'live_art_folder'\n"
        f"      }});\n"
        f"    }}\n"
        f"    Calc.setExpressions(batch);\n"
        f"    const scrollEl = document.querySelector('.dcg-exppanel-container') || document.querySelector('.dcg-exppanel') || document.querySelector('.dcg-sliding-interior');\n"
        f"    if (scrollEl) scrollEl.scrollTop = scrollEl.scrollHeight;\n"
        f"    const lastExp = document.querySelector('.dcg-expressionitem:last-of-type');\n"
        f"    if (lastExp) lastExp.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});\n"
        f"    if (next > 0 && curves[next - 1].pt) {{\n"
        f"      const p = curves[next - 1].pt;\n"
        f"      Calc.setExpression({{\n"
        f"        id: 'pen_tip',\n"
        f"        latex: '(' + p[0] + ',' + p[1] + ')',\n"
        f"        color: '#ff2d55',\n"
        f"        pointSize: 9,\n"
        f"        dragMode: 'NONE'\n"
        f"      }});\n"
        f"    }}\n"
        f"    idx = next;\n"
        f"    if (idx >= curves.length) {{\n"
        f"      clearInterval(timer);\n"
        f"      Calc.removeExpression({{ id: 'pen_tip' }});\n"
        f"      console.log('%c✓ Hoàn thành tác phẩm trình diễn!', 'color: #10b981; font-size: 15px; font-weight: bold;');\n"
        f"    }}\n"
        f"  }}, stepDelay);\n"
        f"}})();"
    )

    if fmt == 'txt':
        import io
        buf = io.BytesIO("\n".join(equations).encode('utf-8'))
        return send_file(buf, mimetype='text/plain', as_attachment=True, download_name=f'desmos_equations_frame_{frame_num}.txt')
    elif fmt == 'state_json':
        import io
        buf = io.BytesIO(json.dumps(desmos_state, indent=2).encode('utf-8'))
        return send_file(buf, mimetype='application/json', as_attachment=True, download_name=f'desmos_state_frame_{frame_num}.json')
    elif fmt == 'js':
        import io
        buf = io.BytesIO(js_script.encode('utf-8'))
        return send_file(buf, mimetype='application/javascript', as_attachment=True, download_name=f'desmos_inject_frame_{frame_num}.js')
    elif fmt == 'live_js':
        import io
        buf = io.BytesIO(live_script.encode('utf-8'))
        return send_file(buf, mimetype='application/javascript', as_attachment=True, download_name=f'desmos_live_sketch_frame_{frame_num}.js')

    return jsonify({
        'success': True,
        'frame': frame_num,
        'count': len(equations),
        'width': w,
        'height': h,
        'color': color,
        'equations': equations,
        'js_script': js_script,
        'live_script': live_script,
        'desmos_state': desmos_state
    })


@app.route('/api/upload', methods=['POST'])
def api_upload():
    global image_width, image_height
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part in request'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    filename = secure_filename(file.filename).lower()
    ext = filename.rsplit('.', 1)[-1] if '.' in filename else ''

    max_frames = int(request.form.get('max_frames', 60))
    target_width = int(request.form.get('target_width', 480))

    temp_path = os.path.join('/tmp', f'upload_{int(time.time())}_{filename}')
    file.save(temp_path)

    image_exts = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
    video_exts = {'mp4', 'avi', 'mov', 'webm', 'mkv', 'gif'}

    try:
        # Clear existing frames directory
        shutil.rmtree(FRAME_DIR, ignore_errors=True)
        os.makedirs(FRAME_DIR, exist_ok=True)
        with cache_lock:
            frame_cache.clear()

        if ext in image_exts:
            # Process single image
            img = cv2.imread(temp_path)
            if img is None:
                return jsonify({'success': False, 'error': 'Unable to decode image'}), 400
            h, w = img.shape[:2]
            if target_width and w > target_width:
                target_height = int(h * (target_width / w))
                img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)

            dest_file = os.path.join(FRAME_DIR, 'frame1.png')
            cv2.imwrite(dest_file, img)
            image_height, image_width = img.shape[:2]
            total_saved = 1

        elif ext in video_exts:
            # Process video / animated GIF
            cap = cv2.VideoCapture(temp_path)
            total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_vid_frames <= 0:
                total_vid_frames = max_frames

            step = max(1, total_vid_frames // max_frames)
            saved_count = 0
            cur_idx = 0

            while cap.isOpened() and saved_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if cur_idx % step == 0:
                    h, w = frame.shape[:2]
                    if target_width and w > target_width:
                        target_height = int(h * (target_width / w))
                        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                    saved_count += 1
                    out_path = os.path.join(FRAME_DIR, f'frame{saved_count}.png')
                    cv2.imwrite(out_path, frame)
                    image_height, image_width = frame.shape[:2]
                cur_idx += 1
            cap.release()
            total_saved = saved_count

            if total_saved == 0:
                return jsonify({'success': False, 'error': 'Failed to extract video frames'}), 400
        else:
            return jsonify({'success': False, 'error': f'Unsupported file type: {ext}'}), 400

        # Remove temp upload
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            'success': True,
            'total_frames': total_saved,
            'width': image_width,
            'height': image_height
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/settings', methods=['POST'])
def api_update_settings():
    global COLOUR, BILATERAL_FILTER, USE_L2_GRADIENT, SHOW_GRID, CANNY_LOW, CANNY_HIGH
    data = request.get_json() or {}

    filters_changed = False

    if 'color' in data:
        COLOUR = str(data['color'])
    if 'bilateral' in data:
        if BILATERAL_FILTER != bool(data['bilateral']):
            BILATERAL_FILTER = bool(data['bilateral'])
            filters_changed = True
    if 'use_l2_gradient' in data:
        if USE_L2_GRADIENT != bool(data['use_l2_gradient']):
            USE_L2_GRADIENT = bool(data['use_l2_gradient'])
            filters_changed = True
    if 'canny_low' in data:
        c_low = int(data['canny_low'])
        if CANNY_LOW != c_low:
            CANNY_LOW = c_low
            filters_changed = True
    if 'canny_high' in data:
        c_high = int(data['canny_high'])
        if CANNY_HIGH != c_high:
            CANNY_HIGH = c_high
            filters_changed = True
    if 'show_grid' in data:
        SHOW_GRID = bool(data['show_grid'])

    if filters_changed:
        with cache_lock:
            frame_cache.clear()

    return jsonify({
        'success': True,
        'settings': {
            'color': COLOUR,
            'bilateral': BILATERAL_FILTER,
            'use_l2_gradient': USE_L2_GRADIENT,
            'show_grid': SHOW_GRID,
            'canny_low': CANNY_LOW,
            'canny_high': CANNY_HIGH
        }
    })


@app.route('/api/reset-default-frames', methods=['POST'])
def api_reset_default_frames():
    global image_width, image_height
    backup_dir = 'frames_default'
    if not os.path.exists(backup_dir):
        return jsonify({'success': False, 'error': 'Default frames backup not found'}), 404

    shutil.rmtree(FRAME_DIR, ignore_errors=True)
    shutil.copytree(backup_dir, FRAME_DIR)

    with cache_lock:
        frame_cache.clear()

    update_dimensions_from_files()
    files = get_sorted_frame_files()

    return jsonify({
        'success': True,
        'total_frames': len(files),
        'width': image_width,
        'height': image_height
    })


def help():
    print('backend.py -f <source> -e <extension> -c <colour> -b -d -l -g --yes --no-browser --size <widthxheight> --format <extension>\n')
    print('\t-h\t\t\tGet help\n')
    print('-Options\n')
    print('\t-f <source>\t\tThe directory from which the frames are stored (e.g. frames)')
    print('\t-e <extension>\t\tThe extension of the frame files (e.g. png)')
    print('\t-c <colour>\t\tThe colour of the lines to be drawn (e.g. #2464b4)')
    print('\t-b\t\t\tReduce number of lines with bilateral filter for simpler renders')
    print('\t-d\t\t\tDownload rendered frames automatically')
    print('\t-l\t\t\tReduce number of lines with L2 gradient for quicker renders')
    print('\t-g\t\t\tHide the grid in the background of the graph\n')
    print('\t--yes\t\t\tAgree to EULA without input prompt')
    print('\t--no-browser\t\tRun renderer server without opening a web browser')
    print('\t--size <widthxheight>\tDimensions for downloaded images (e.g. 3840x2160)')
    print('\t--format <extension>\tSpecify format when downloading frames: "svg" or "png" (default is "png")')


def find_available_port(start_port=5000):
    """Find an available port if default port is occupied."""
    import socket
    for p in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', p))
                return p
            except OSError:
                continue
    return start_port


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hf:e:c:bdlg", ['static', 'block=', 'maxpblock=', 'yes', 'no-browser', 'size=', 'format=', 'port='])
    except getopt.GetoptError:
        print('Error: Invalid argument(s)\n')
        help()
        sys.exit(2)

    eula = 'y' if not sys.stdin.isatty() else ''
    custom_port_specified = False

    for opt, arg in opts:
        if opt == '-h':
            help()
            sys.exit()
        elif opt == '-f':
            FRAME_DIR = arg
        elif opt == '-e':
            FILE_EXT = arg
        elif opt == '-c':
            COLOUR = arg
        elif opt == '-b':
            BILATERAL_FILTER = True
        elif opt == '-d':
            DOWNLOAD_IMAGES = True
        elif opt == '-l':
            USE_L2_GRADIENT = True
        elif opt == '-g':
            SHOW_GRID = False
        elif opt == '--yes':
            eula = 'y'
        elif opt == '--no-browser':
            OPEN_BROWSER = False
        elif opt == '--size':
            SCREENSHOT_SIZE = [int(n) for n in arg.split('x', maxsplit=1)]
        elif opt == '--format':
            SCREENSHOT_FORMAT = arg
        elif opt == '--port':
            PORT = int(arg)
            custom_port_specified = True

    if not custom_port_specified:
        PORT = find_available_port(PORT)

    print('''  _____                                
 |  __ \                               
 | |  | | ___  ___ _ __ ___   ___  ___ 
 | |  | |/ _ \/ __| '_ ` _ \ / _ \/ __|
 | |__| |  __/\__ \ | | | | | (_) \__ \\
 |_____/ \___||___/_| |_| |_|\___/|___/

                   BEZIER RENDERER (Web Edition)
''')

    update_dimensions_from_files()
    files = get_sorted_frame_files()
    print(f'--> Detected {len(files)} frame(s) in "{FRAME_DIR}" ({image_width}x{image_height})')
    print(f'--> Web server starting at: http://127.0.0.1:{PORT}/calculator\n')

    if OPEN_BROWSER:
        def open_browser():
            webbrowser.open(f'http://127.0.0.1:{PORT}/calculator')
        Timer(1.5, open_browser).start()

    app.run(host='0.0.0.0', port=PORT, threaded=True)
