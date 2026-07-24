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


def render_zoomable_image(data_uri: str, height: int = 420):
    """
    Render an image with scroll-to-zoom and click-and-drag panning,
    plus +/- /reset controls. Useful for inspecting fine detail in an
    X-ray or its Grad-CAM heatmap.
    """
    html = f"""
    <div style="font-family: -apple-system, sans-serif;">
      <div id="zoom-container" style="
          position:relative; width:100%; height:{height}px; overflow:hidden;
          border-radius:12px; background:#0b1220; cursor:grab;">
        <img id="zoom-img" src="{data_uri}" draggable="false" style="
            position:absolute; top:50%; left:50%;
            transform:translate(-50%,-50%) scale(1);
            max-width:100%; max-height:100%; user-select:none; pointer-events:none;">
      </div>
      <div style="display:flex; gap:8px; margin-top:8px; align-items:center; justify-content:center;">
        <button id="zoom-out" style="border:1px solid #d0d5dd; background:#fff; border-radius:8px; width:30px; height:30px; cursor:pointer; font-size:16px;">&minus;</button>
        <span id="zoom-pct" style="font-size:12.5px; color:#667085; min-width:42px; text-align:center;">100%</span>
        <button id="zoom-in" style="border:1px solid #d0d5dd; background:#fff; border-radius:8px; width:30px; height:30px; cursor:pointer; font-size:16px;">+</button>
        <button id="zoom-reset" style="border:1px solid #d0d5dd; background:#fff; border-radius:8px; padding:0 12px; height:30px; cursor:pointer; font-size:12.5px; margin-left:6px;">Reset</button>
        <span style="font-size:11.5px; color:#98a2b3; margin-left:8px;">Scroll to zoom &middot; drag to pan</span>
      </div>
    </div>
    <script>
    (function() {{
        const container = document.getElementById('zoom-container');
        const img = document.getElementById('zoom-img');
        const pctLabel = document.getElementById('zoom-pct');
        let scale = 1, posX = 0, posY = 0, isPanning = false, startX = 0, startY = 0;

        function update() {{
            img.style.transform = `translate(calc(-50% + ${{posX}}px), calc(-50% + ${{posY}}px)) scale(${{scale}})`;
            pctLabel.innerText = Math.round(scale * 100) + '%';
        }}

        container.addEventListener('wheel', function(e) {{
            e.preventDefault();
            const delta = e.deltaY < 0 ? 0.12 : -0.12;
            scale = Math.min(Math.max(0.5, scale + delta), 6);
            update();
        }}, {{ passive: false }});

        container.addEventListener('mousedown', function(e) {{
            isPanning = true;
            startX = e.clientX - posX;
            startY = e.clientY - posY;
            container.style.cursor = 'grabbing';
        }});
        window.addEventListener('mousemove', function(e) {{
            if (!isPanning) return;
            posX = e.clientX - startX;
            posY = e.clientY - startY;
            update();
        }});
        window.addEventListener('mouseup', function() {{
            isPanning = false;
            container.style.cursor = 'grab';
        }});

        document.getElementById('zoom-in').addEventListener('click', function() {{
            scale = Math.min(scale + 0.3, 6); update();
        }});
        document.getElementById('zoom-out').addEventListener('click', function() {{
            scale = Math.max(scale - 0.3, 0.5); update();
        }});
        document.getElementById('zoom-reset').addEventListener('click', function() {{
            scale = 1; posX = 0; posY = 0; update();
        }});
    }})();
    </script>
    """
    components.html(html, height=height + 60)


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
