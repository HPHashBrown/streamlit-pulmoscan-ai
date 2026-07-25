"""
components_ui.py

Small self-contained HTML/JS widgets that Streamlit doesn't provide
natively: a zoomable/pannable image viewer, and a "read aloud" button
using the browser's built-in speech synthesis. Both render via
streamlit.components.v1.html, which sandboxes them in an iframe -- so
styling is kept minimal/self-contained rather than trying to inherit
the app's theme colors.
"""

import json

import streamlit.components.v1 as components


def render_advanced_viewer(
    original_uri: str,
    heatmap_uri: str,
    mode: str = "Original",
    height: int = 420,
    pixel_spacing_mm: float = None,
    initial_zoom: float = 1.0,
    flicker_ms: int = 600,
):
    """
    The main image viewer: zoom/pan, 5 viewing modes (Original, Grad-CAM,
    Side-by-side, Split slider, Flicker), measurement/annotation tools
    (ruler, angle, freehand draw), keyboard shortcuts, and PNG snapshot
    export of exactly what's on screen (including any annotations).

    Zoom/pan and the annotation canvas are combined into a single
    component (rather than separate layered components) because they
    need to share one coordinate space -- an annotation drawn at a given
    screen position must land on the same image pixel regardless of the
    current zoom/pan state.

    pixel_spacing_mm: if known (e.g. from a DICOM file's PixelSpacing
    tag), ruler measurements are also shown converted to millimeters.

    Known limitation: because this is a single self-contained HTML
    component, any Streamlit-level rerun (e.g. dragging a slider outside
    this component) remounts it fresh, clearing zoom/pan and annotations.
    This is a constraint of Streamlit's component model, not a bug.
    """
    spacing_js = "null" if pixel_spacing_mm is None else json.dumps(pixel_spacing_mm)
    mode_js = json.dumps(mode)
    initial_zoom_js = json.dumps(max(0.5, min(6, initial_zoom)))
    flicker_ms_js = json.dumps(max(150, int(flicker_ms)))

    html = f"""
    <div style="font-family: -apple-system, sans-serif; user-select:none;">

      <div style="display:flex; gap:6px; margin-bottom:8px; flex-wrap:wrap; align-items:center;">
        <button class="tool-btn" id="tool-pan" title="Pan/select (default)">&#128073; Pan</button>
        <button class="tool-btn" id="tool-ruler" title="Ruler: click two points">&#128207; Ruler</button>
        <button class="tool-btn" id="tool-angle" title="Angle: click three points">&#128260; Angle</button>
        <button class="tool-btn" id="tool-draw" title="Freehand markup">&#9998; Draw</button>
        <button class="tool-btn" id="tool-clear" title="Clear all annotations">&#128465; Clear</button>
        <span style="flex:1;"></span>
        <button class="tool-btn" id="btn-export" title="Export snapshot as PNG">&#128247; Export PNG</button>
      </div>

      <div id="viewer-container" tabindex="0" style="
          position:relative; width:100%; height:{height}px; overflow:hidden;
          border-radius:12px; background:#0b1220; cursor:grab; outline:none;">
        <div id="transform-wrap" style="
            position:absolute; top:50%; left:50%;
            transform:translate(-50%,-50%) scale(1); width:100%; height:100%;">
          <img id="img-a" src="{original_uri}" draggable="false" style="
              position:absolute; top:0; left:0; width:100%; height:100%;
              object-fit:contain; pointer-events:none;">
          <img id="img-b" src="{heatmap_uri}" draggable="false" style="
              position:absolute; top:0; left:0; width:100%; height:100%;
              object-fit:contain; pointer-events:none; display:none;">
          <div id="split-clip" style="
              position:absolute; top:0; left:0; width:50%; height:100%; overflow:hidden; display:none;">
            <img src="{heatmap_uri}" draggable="false" style="
                position:absolute; top:0; left:0; width:200%; height:100%; object-fit:contain; pointer-events:none;">
          </div>
          <div id="split-handle" style="
              position:absolute; top:0; left:50%; width:2px; height:100%;
              background:#fff; display:none; box-shadow:0 0 6px rgba(0,0,0,0.5);"></div>
          <canvas id="annotation-canvas" style="position:absolute; top:0; left:0; width:100%; height:100%;"></canvas>
        </div>
      </div>

      <div style="display:flex; gap:8px; margin-top:8px; align-items:center; flex-wrap:wrap;">
        <button class="tool-btn small" id="zoom-out">&minus;</button>
        <span id="zoom-pct" style="font-size:12.5px; color:#667085; min-width:40px; text-align:center;">100%</span>
        <button class="tool-btn small" id="zoom-in">+</button>
        <button class="tool-btn small" id="zoom-reset">Reset</button>
        <button class="tool-btn small" id="btn-fullscreen">&#9974; Fullscreen</button>
        <span id="split-slider-wrap" style="display:none; align-items:center; gap:6px;">
          <span style="font-size:11.5px; color:#98a2b3;">Split</span>
          <input type="range" id="split-range" min="0" max="100" value="50" style="width:100px;">
        </span>
        <span id="measure-readout" style="font-size:12.5px; color:#1957D6; font-weight:600;"></span>
        <span style="flex:1;"></span>
        <span style="font-size:11px; color:#98a2b3;">Keys: Z zoom in &middot; F fullscreen &middot; R reset</span>
      </div>

      <style>
        .tool-btn {{
          border:1px solid #d0d5dd; background:#fff; border-radius:8px;
          padding:6px 12px; font-size:12.5px; font-weight:600; cursor:pointer; color:#344054;
        }}
        .tool-btn.active {{ background:#1957D6; color:#fff; border-color:#1957D6; }}
        .tool-btn.small {{ width:30px; height:30px; padding:0; font-size:15px; }}
      </style>

      <script>
      (function() {{
          const mode = {mode_js};
          const pixelSpacing = {spacing_js};
          const flickerMs = {flicker_ms_js};

          const container = document.getElementById('viewer-container');
          const wrap = document.getElementById('transform-wrap');
          const imgA = document.getElementById('img-a');
          const imgB = document.getElementById('img-b');
          const splitClip = document.getElementById('split-clip');
          const splitHandle = document.getElementById('split-handle');
          const splitRange = document.getElementById('split-range');
          const splitWrap = document.getElementById('split-slider-wrap');
          const canvas = document.getElementById('annotation-canvas');
          const ctx = canvas.getContext('2d');
          const pctLabel = document.getElementById('zoom-pct');
          const readout = document.getElementById('measure-readout');

          let scale = {initial_zoom_js}, posX = 0, posY = 0, isPanning = false, startX = 0, startY = 0;
          let tool = 'pan';
          let points = [];
          let strokes = [];
          let currentStroke = null;
          let flickerTimer = null;

          function resizeCanvas() {{
              canvas.width = container.clientWidth;
              canvas.height = container.clientHeight;
              redrawAnnotations();
          }}

          function applyMode() {{
              imgA.style.display = 'block';
              imgB.style.display = 'none';
              splitClip.style.display = 'none';
              splitHandle.style.display = 'none';
              splitWrap.style.display = 'none';
              if (flickerTimer) {{ clearInterval(flickerTimer); flickerTimer = null; }}

              if (mode === 'Grad-CAM') {{
                  imgA.style.display = 'none';
                  imgB.style.display = 'block';
              }} else if (mode === 'Side-by-side') {{
                  imgA.style.width = '50%';
                  imgB.style.display = 'block';
                  imgB.style.left = '50%';
                  imgB.style.width = '50%';
              }} else if (mode === 'Split slider') {{
                  splitClip.style.display = 'block';
                  splitHandle.style.display = 'block';
                  splitWrap.style.display = 'inline-flex';
                  updateSplit(50);
              }} else if (mode === 'Flicker') {{
                  let showingA = true;
                  flickerTimer = setInterval(function() {{
                      showingA = !showingA;
                      imgA.style.display = showingA ? 'block' : 'none';
                      imgB.style.display = showingA ? 'none' : 'block';
                  }}, flickerMs);
              }}
          }}

          function updateSplit(pct) {{
              splitClip.style.width = pct + '%';
              splitHandle.style.left = pct + '%';
          }}
          splitRange.addEventListener('input', function() {{ updateSplit(this.value); }});

          function updateTransform() {{
              wrap.style.transform = `translate(calc(-50% + ${{posX}}px), calc(-50% + ${{posY}}px)) scale(${{scale}})`;
              pctLabel.innerText = Math.round(scale * 100) + '%';
          }}

          container.addEventListener('wheel', function(e) {{
              e.preventDefault();
              const delta = e.deltaY < 0 ? 0.12 : -0.12;
              scale = Math.min(Math.max(0.5, scale + delta), 6);
              updateTransform();
          }}, {{ passive: false }});

          function setTool(name) {{
              tool = name;
              points = [];
              document.querySelectorAll('.tool-btn').forEach(function(b) {{ b.classList.remove('active'); }});
              const map = {{ pan: 'tool-pan', ruler: 'tool-ruler', angle: 'tool-angle', draw: 'tool-draw' }};
              if (map[name]) document.getElementById(map[name]).classList.add('active');
              container.style.cursor = name === 'pan' ? 'grab' : 'crosshair';
          }}
          document.getElementById('tool-pan').addEventListener('click', function() {{ setTool('pan'); }});
          document.getElementById('tool-ruler').addEventListener('click', function() {{ setTool('ruler'); }});
          document.getElementById('tool-angle').addEventListener('click', function() {{ setTool('angle'); }});
          document.getElementById('tool-draw').addEventListener('click', function() {{ setTool('draw'); }});
          document.getElementById('tool-clear').addEventListener('click', function() {{
              strokes = []; points = []; readout.innerText = '';
              redrawAnnotations();
          }});
          setTool('pan');

          function canvasPoint(e) {{
              const rect = canvas.getBoundingClientRect();
              return {{
                  x: (e.clientX - rect.left) / rect.width * canvas.width,
                  y: (e.clientY - rect.top) / rect.height * canvas.height,
              }};
          }}

          function dist(p1, p2) {{ return Math.hypot(p2.x - p1.x, p2.y - p1.y); }}

          function angleBetween(a, vertex, b) {{
              const v1 = {{ x: a.x - vertex.x, y: a.y - vertex.y }};
              const v2 = {{ x: b.x - vertex.x, y: b.y - vertex.y }};
              const dot = v1.x * v2.x + v1.y * v2.y;
              const mag = Math.hypot(v1.x, v1.y) * Math.hypot(v2.x, v2.y);
              if (mag === 0) return 0;
              return Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180 / Math.PI;
          }}

          function redrawAnnotations() {{
              ctx.clearRect(0, 0, canvas.width, canvas.height);
              ctx.lineWidth = 2;
              ctx.strokeStyle = '#F5B82E';
              ctx.fillStyle = '#F5B82E';
              ctx.font = '13px -apple-system, sans-serif';

              strokes.forEach(function(s) {{
                  if (s.type === 'ruler') {{
                      ctx.beginPath();
                      ctx.moveTo(s.p1.x, s.p1.y);
                      ctx.lineTo(s.p2.x, s.p2.y);
                      ctx.stroke();
                      [s.p1, s.p2].forEach(function(p) {{
                          ctx.beginPath(); ctx.arc(p.x, p.y, 3, 0, 7); ctx.fill();
                      }});
                      const mid = {{ x: (s.p1.x + s.p2.x) / 2, y: (s.p1.y + s.p2.y) / 2 }};
                      ctx.fillText(s.label, mid.x + 6, mid.y - 6);
                  }} else if (s.type === 'angle') {{
                      ctx.beginPath();
                      ctx.moveTo(s.p1.x, s.p1.y);
                      ctx.lineTo(s.vertex.x, s.vertex.y);
                      ctx.lineTo(s.p2.x, s.p2.y);
                      ctx.stroke();
                      [s.p1, s.vertex, s.p2].forEach(function(p) {{
                          ctx.beginPath(); ctx.arc(p.x, p.y, 3, 0, 7); ctx.fill();
                      }});
                      ctx.fillText(s.label, s.vertex.x + 8, s.vertex.y - 8);
                  }} else if (s.type === 'draw') {{
                      ctx.beginPath();
                      s.points.forEach(function(p, i) {{
                          if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
                      }});
                      ctx.stroke();
                  }}
              }});
          }}

          container.addEventListener('mousedown', function(e) {{
              if (tool === 'pan') {{
                  isPanning = true;
                  startX = e.clientX - posX;
                  startY = e.clientY - posY;
                  container.style.cursor = 'grabbing';
                  return;
              }}
              const p = canvasPoint(e);
              if (tool === 'draw') {{
                  currentStroke = {{ type: 'draw', points: [p] }};
                  strokes.push(currentStroke);
              }} else if (tool === 'ruler') {{
                  points.push(p);
                  if (points.length === 2) {{
                      const pixelDist = dist(points[0], points[1]);
                      let label = Math.round(pixelDist) + 'px';
                      if (pixelSpacing) {{
                          label += ' (' + (pixelDist * pixelSpacing).toFixed(1) + ' mm)';
                      }}
                      strokes.push({{ type: 'ruler', p1: points[0], p2: points[1], label: label }});
                      readout.innerText = 'Distance: ' + label;
                      points = [];
                      redrawAnnotations();
                  }}
              }} else if (tool === 'angle') {{
                  points.push(p);
                  if (points.length === 3) {{
                      const deg = angleBetween(points[0], points[1], points[2]);
                      const label = deg.toFixed(1) + '\\u00b0';
                      strokes.push({{ type: 'angle', p1: points[0], vertex: points[1], p2: points[2], label: label }});
                      readout.innerText = 'Angle: ' + label;
                      points = [];
                      redrawAnnotations();
                  }}
              }}
          }});

          window.addEventListener('mousemove', function(e) {{
              if (isPanning) {{
                  posX = e.clientX - startX;
                  posY = e.clientY - startY;
                  updateTransform();
                  return;
              }}
              if (tool === 'draw' && currentStroke) {{
                  currentStroke.points.push(canvasPoint(e));
                  redrawAnnotations();
              }}
          }});
          window.addEventListener('mouseup', function() {{
              isPanning = false;
              currentStroke = null;
              if (tool === 'pan') container.style.cursor = 'grab';
          }});

          document.getElementById('zoom-in').addEventListener('click', function() {{ scale = Math.min(scale + 0.3, 6); updateTransform(); }});
          document.getElementById('zoom-out').addEventListener('click', function() {{ scale = Math.max(scale - 0.3, 0.5); updateTransform(); }});
          document.getElementById('zoom-reset').addEventListener('click', function() {{ scale = 1; posX = 0; posY = 0; updateTransform(); }});

          document.getElementById('btn-fullscreen').addEventListener('click', function() {{
              if (container.requestFullscreen) container.requestFullscreen();
          }});

          container.addEventListener('keydown', function(e) {{
              if (e.key === 'z' || e.key === 'Z') {{ scale = Math.min(scale + 0.3, 6); updateTransform(); }}
              if (e.key === 'f' || e.key === 'F') {{ if (container.requestFullscreen) container.requestFullscreen(); }}
              if (e.key === 'r' || e.key === 'R') {{ scale = 1; posX = 0; posY = 0; updateTransform(); }}
          }});

          document.getElementById('btn-export').addEventListener('click', function() {{
              const script = document.createElement('script');
              script.src = 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js';
              script.onload = function() {{
                  html2canvas(container, {{ backgroundColor: '#0b1220' }}).then(function(canvasOut) {{
                      const link = document.createElement('a');
                      link.download = 'pulmoscan-snapshot.png';
                      link.href = canvasOut.toDataURL('image/png');
                      link.click();
                  }});
              }};
              document.body.appendChild(script);
          }});

          window.addEventListener('resize', resizeCanvas);
          applyMode();
          resizeCanvas();
          updateTransform();
      }})();
      </script>
    </div>
    """
    components.html(html, height=height + 130)

def render_read_aloud_button(text: str, label: str = "\U0001F50A Read my report aloud"):
    """
    A button that reads the given text aloud using the browser's native
    speech synthesis (no server-side processing, no audio files). Only
    starts speaking on an explicit click -- browsers require a real user
    gesture for audio anyway, which conveniently guarantees it can never
    fire automatically.
    """
    safe_text = json.dumps(text)
    html = f"""
    <div style="display:flex; gap:8px; font-family: -apple-system, sans-serif;">
      <button id="read-aloud-btn" style="
          background:#1957D6; color:#fff; border:none; border-radius:999px;
          padding:9px 18px; font-size:13.5px; font-weight:600; cursor:pointer;">
        {label}
      </button>
      <button id="stop-aloud-btn" style="
          background:#fff; color:#475467; border:1px solid #d0d5dd; border-radius:999px;
          padding:9px 16px; font-size:13.5px; font-weight:600; cursor:pointer;">
        &#9209; Stop
      </button>
    </div>
    <script>
    (function() {{
        const text = {safe_text};
        const speakBtn = document.getElementById('read-aloud-btn');
        const stopBtn = document.getElementById('stop-aloud-btn');

        speakBtn.addEventListener('click', function() {{
            if (!('speechSynthesis' in window)) {{
                alert('Sorry, your browser does not support text-to-speech.');
                return;
            }}
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 0.98;
            window.speechSynthesis.speak(utterance);
        }});

        stopBtn.addEventListener('click', function() {{
            window.speechSynthesis.cancel();
        }});
    }})();
    </script>
    """
    components.html(html, height=50)
