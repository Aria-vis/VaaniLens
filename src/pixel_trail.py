"""PixelTrail WebGL2 Interactive Component for VaaniLens Streamlit Frontend.

Adapted directly from React Bits (JavaScript + CSS variant):
https://reactbits.dev/components/pixel-trail
Renders a high-performance WebGL2 interactive fluid pixel trail with gooey physics,
lighting shaders, and cursor burst interaction.
"""

import streamlit.components.v1 as components


def get_pixel_trail_html(
    height: int = 180,
    grid_size: int = 45,
    trail_size: float = 0.12,
    max_age: int = 500,
    color: str = "#A855F7",
    trail_color: str = "#EC4899",
    goo_strength: float = 2.0,
    glow: float = 0.5,
    shine: float = 0.65,
    title: str = "🔍 VaaniLens",
    subtitle: str = "Multilingual, Explainable Sentiment, Entity & Translation Intelligence",
) -> str:
    """Generates the self-contained HTML/JS/WebGL2 bundle for the PixelTrail hero banner."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: radial-gradient(circle at 50% 20%, #1e1b4b 0%, #0f172a 70%, #020617 100%);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    user-select: none;
    -webkit-user-select: none;
    border-radius: 12px;
  }}
  #container {{
    position: relative;
    width: 100%;
    height: 100%;
    overflow: hidden;
    cursor: crosshair;
    border-radius: 12px;
    box-shadow: inset 0 0 40px rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(168, 85, 247, 0.2);
  }}
  canvas {{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    display: block;
    z-index: 1;
    pointer-events: none;
  }}
  .overlay {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 2;
    pointer-events: none;
    text-align: center;
    width: 95%;
  }}
  .title {{
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 40%, #c084fc 80%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
    filter: drop-shadow(0 2px 8px rgba(0,0,0,0.4));
  }}
  .subtitle {{
    font-size: 0.92rem;
    color: #cbd5e1;
    font-weight: 400;
    letter-spacing: 0.01em;
    margin-bottom: 10px;
    opacity: 0.95;
  }}
  .badges {{
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
  }}
  .badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    background: rgba(168, 85, 247, 0.16);
    color: #e9d5ff;
    border: 1px solid rgba(192, 132, 252, 0.35);
    backdrop-filter: blur(4px);
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
  }}
  .hint {{
    position: absolute;
    bottom: 8px;
    right: 14px;
    font-size: 0.7rem;
    color: #64748b;
    pointer-events: none;
    z-index: 3;
    opacity: 0.75;
  }}
</style>
</head>
<body>
<div id="container">
  <canvas id="pixelCanvas"></canvas>
  <div class="overlay">
    <div class="title">{title}</div>
    <div class="subtitle">{subtitle}</div>
    <div class="badges">
      <span class="badge">🌐 Hindi & English</span>
      <span class="badge">⚡ DistilBERT SST-2</span>
      <span class="badge">🔄 MarianMT NMT</span>
      <span class="badge">💡 LIME Explainability</span>
      <span class="badge">🛡️ Ethical Safety Rails</span>
    </div>
  </div>
  <div class="hint">✨ Move mouse or click inside banner</div>
</div>

<script>
(function() {{
  const TRAIL_SIZE = 512;
  const AGE_SIZE = 256;
  const SHAPES = {{ square: 0, circle: 1, diamond: 2 }};
  const DECAYS = {{ fade: 0, shrink: 1, dither: 2 }};

  const VERTEX = `#version 300 es
  in vec2 a_position;
  void main() {{
    gl_Position = vec4(a_position, 0.0, 1.0);
  }}
  `;

  const FRAGMENT = `#version 300 es
  precision highp float;
  uniform sampler2D u_trail;
  uniform sampler2D u_image;
  uniform sampler2D u_age;
  uniform vec2 u_resolution;
  uniform float u_gridSize;
  uniform float u_sigma;
  uniform vec3 u_color;
  uniform vec3 u_trailColor;
  uniform float u_ramp;
  uniform float u_shine;
  uniform float u_glow;
  uniform int u_shape;
  uniform int u_decay;
  uniform float u_gap;
  uniform float u_sparkle;
  uniform float u_time;
  uniform float u_hasImage;
  out vec4 outColor;

  const float BAYER[16] = float[16](0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);
  const vec3 LIGHT = vec3(-0.42, 0.58, 0.7);

  float hash(vec2 p) {{
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
  }}

  float erfApprox(float x) {{
    float s = sign(x);
    x = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * x);
    float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
    return s * y;
  }}

  float cellValue(ivec2 c) {{
    vec2 center = (vec2(c) + 0.5) / u_gridSize;
    if (center.x < 0.0 || center.y < 0.0 || center.x > 1.0 || center.y > 1.0) return 0.0;
    float t = texelFetch(u_trail, clamp(ivec2(center * 512.0), ivec2(0), ivec2(511)), 0).r;
    if (t <= 0.0) return 0.0;
    if (u_sparkle > 0.0) {{
      float n = hash(vec2(c) + floor(u_time * 14.0) * 7.31);
      float dim = 1.0 - t;
      if (n < u_sparkle * dim * 0.35) t = max(t, 0.85);
      else if (n > 1.0 - u_sparkle * dim * 0.5) t = 0.0;
    }}
    if (u_decay == 2 && t * 2.2 <= (BAYER[(c.y & 3) * 4 + (c.x & 3)] + 0.5) / 16.0) return 0.0;
    return t;
  }}

  vec3 baseColor(ivec2 owner, vec2 uv) {{
    if (u_hasImage > 0.5) return texelFetch(u_image, clamp(owner, ivec2(0), ivec2(textureSize(u_image, 0)) - 1), 0).rgb;
    if (u_ramp > 0.5) return mix(u_trailColor, u_color, smoothstep(0.05, 0.95, texture(u_age, uv).r));
    return u_color;
  }}

  vec3 tone(vec3 base, float strength) {{
    vec3 body = mix(base * 0.55, base, smoothstep(0.3, 0.75, strength));
    return mix(body, mix(base, vec3(1.0), 0.22), smoothstep(0.85, 1.0, strength) * u_shine);
  }}

  float seams(vec2 q, float cs) {{
    if (u_gap <= 0.0) return 0.0;
    vec2 f = fract(q / cs);
    float border = min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y)) * cs;
    float width = max(1.0, u_gap * cs * 0.5);
    return 1.0 - smoothstep(width - 0.75, width + 0.75, border);
  }}

  vec3 shade(vec3 base, vec3 n) {{
    vec3 l = normalize(LIGHT);
    vec3 h = normalize(l + vec3(0.0, 0.0, 1.0));
    float lit = dot(n, l) - l.z;
    vec3 color = base * (1.0 + lit * 0.9 * u_shine);
    float spec = pow(max(dot(n, h), 0.0), 64.0) * 0.85 + pow(max(dot(n, h), 0.0), 10.0) * 0.08;
    color += spec * u_shine * mix(vec3(1.0), base, 0.15) * step(0.001, 1.0 - n.z);
    return clamp(color, 0.0, 1.0);
  }}

  void main() {{
    float maxRes = max(u_resolution.x, u_resolution.y);
    vec2 q = gl_FragCoord.xy - 0.5 * u_resolution + 0.5 * maxRes;
    vec2 uv = q / maxRes;
    float cs = maxRes / u_gridSize;
    ivec2 base = ivec2(floor(q / cs));

    float alpha = 0.0;
    vec3 color = vec3(0.0);

    if (u_sigma > 0.0) {{
      float inv = 0.70710678 / u_sigma;
      float norm = 0.39894228 / u_sigma;
      float spread = 0.5 / (u_sigma * u_sigma);
      int reach = int(clamp(ceil(3.0 * u_sigma / cs), 1.0, 3.0));
      float field = 0.0;
      vec2 grad = vec2(0.0);
      float best = -1.0;
      ivec2 owner = base;
      float wx[7], gx[7], wy[7], gy[7];
      for (int i = 0; i < 7; i++) {{
        int offset = i - 3;
        if (abs(offset) > reach) continue;
        vec2 lo = vec2(base + ivec2(offset)) * cs - q;
        vec2 hi = lo + cs;
        wx[i] = 0.5 * (erfApprox(hi.x * inv) - erfApprox(lo.x * inv));
        wy[i] = 0.5 * (erfApprox(hi.y * inv) - erfApprox(lo.y * inv));
        gx[i] = norm * (exp(-lo.x * lo.x * spread) - exp(-hi.x * hi.x * spread));
        gy[i] = norm * (exp(-lo.y * lo.y * spread) - exp(-hi.y * hi.y * spread));
      }}
      for (int j = 0; j < 7; j++) {{
        if (abs(j - 3) > reach) continue;
        for (int i = 0; i < 7; i++) {{
          if (abs(i - 3) > reach) continue;
          ivec2 c = base + ivec2(i - 3, j - 3);
          float t = cellValue(c);
          if (t <= 0.0) continue;
          float weight = t * wx[i] * wy[j];
          field += weight;
          grad += t * vec2(gx[i] * wy[j], wx[i] * gy[j]);
          if (weight > best) {{
            best = weight;
            owner = c;
          }}
        }}
      }}
      float edge = max(0.0263, length(grad) * 0.6);
      alpha = smoothstep(0.5 - edge, 0.5 + edge, field);
      if (alpha > 0.0) {{
        vec3 n = normalize(vec3(-grad * cs * 0.55, 1.0));
        float strength = max(cellValue(base), field);
        vec3 body = tone(baseColor(owner, uv), strength);
        body *= 1.0 - seams(q, cs) * 0.32;
        color = shade(body, n);
      }}
    }} else {{
      float t = cellValue(base);
      if (t > 0.0) {{
        float halfSize = 0.5 * (1.0 - u_gap);
        float fade = t;
        vec2 local = fract(q / cs) - 0.5;
        vec2 a = abs(local) - halfSize;
        float d = max(a.x, a.y);
        vec2 dir = a.x > a.y ? vec2(sign(local.x), 0.0) : vec2(0.0, sign(local.y));
        float px = 1.0 / cs;
        float coverage = clamp(0.5 - d / px, 0.0, 1.0);
        alpha = coverage * fade;
        if (alpha > 0.0) {{
          float bevel = smoothstep(-0.16 * (1.0 - u_gap), 0.0, d);
          vec3 n = normalize(vec3(dir * bevel * 0.9, 1.0));
          color = shade(tone(baseColor(base, uv), t), n);
        }}
      }}
    }}

    vec4 pixel = vec4(color * alpha, alpha);
    if (u_glow > 0.0) {{
      float glow = texture(u_trail, uv).r;
      float amount = smoothstep(0.0, 0.7, glow) * u_glow * 0.55 * (1.0 - alpha);
      vec3 tint = u_ramp > 0.5 ? mix(u_trailColor, u_color, 0.5) : u_color;
      pixel += vec4(tint * amount, amount);
    }}
    if (pixel.a <= 0.002) discard;
    outColor = pixel;
  }}
  `;

  function hexToRgb(hex) {{
    hex = hex.replace('#', '');
    if (hex.length === 3) hex = hex.split('').map(c => c + c).join('');
    const num = parseInt(hex, 16);
    return [(num >> 16 & 255) / 255, (num >> 8 & 255) / 255, (num & 255) / 255];
  }}

  function compileShader(gl, type, src) {{
    const shader = gl.createShader(type);
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {{
      console.error(gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }}
    return shader;
  }}

  const container = document.getElementById('container');
  const canvas = document.getElementById('pixelCanvas');
  const gl = canvas.getContext('webgl2', {{
    alpha: true,
    premultipliedAlpha: true,
    antialias: false,
    powerPreference: 'high-performance'
  }});

  if (!gl) return;

  const vs = compileShader(gl, gl.VERTEX_SHADER, VERTEX);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  gl.useProgram(prog);

  const uniforms = {{}};
  const count = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < count; i++) {{
    const info = gl.getActiveUniform(prog, i);
    uniforms[info.name] = gl.getUniformLocation(prog, info.name);
  }}

  const vao = gl.createVertexArray();
  const buf = gl.createBuffer();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

  function createTexture(unit) {{
    const t = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0,0,0,255]));
    return t;
  }}

  const trailTex = createTexture(0);
  const imgTex = createTexture(1);
  const ageTex = createTexture(2);
  gl.uniform1i(uniforms.u_trail, 0);
  gl.uniform1i(uniforms.u_image, 1);
  gl.uniform1i(uniforms.u_age, 2);

  const trailCanvas = document.createElement('canvas');
  trailCanvas.width = trailCanvas.height = TRAIL_SIZE;
  const trailCtx = trailCanvas.getContext('2d');
  trailCtx.fillStyle = 'black';
  trailCtx.fillRect(0, 0, TRAIL_SIZE, TRAIL_SIZE);

  const ageCanvas = document.createElement('canvas');
  ageCanvas.width = ageCanvas.height = AGE_SIZE;
  const ageCtx = ageCanvas.getContext('2d');

  const settings = {{
    gridSize: {grid_size},
    trailSize: {trail_size},
    maxAge: {max_age},
    interpolate: 5,
    goo: {goo_strength},
    color: hexToRgb('{color}'),
    trailColor: hexToRgb('{trail_color}'),
    shape: 0,
    gap: 0.08,
    decay: 0,
    flow: 0.15,
    gravity: 0.05,
    sparkle: 0.12,
    shine: {shine},
    glow: {glow}
  }};

  const state = {{
    alive: true,
    last: 0,
    time: 0,
    width: 1,
    height: 1,
    ratio: Math.min(window.devicePixelRatio || 1, 2),
    points: [],
    force: 0,
    pointer: {{ x: 0.5, y: 0.5, at: 0, vx: 0, vy: 0, inside: false }}
  }};

  function addTouch(x, y, vx, vy) {{
    const pts = state.points;
    const last = pts[pts.length - 1];
    if (last) {{
      const dx = last.x - x;
      const dy = last.y - y;
      const dd = dx * dx + dy * dy;
      const force = Math.max(0.3, Math.min(dd * 10000, 1));
      state.force = force;
      const lines = Math.ceil(dd / Math.pow((settings.trailSize * 0.5) / settings.interpolate, 2));
      if (lines > 1) {{
        for (let i = 1; i < lines; i++) {{
          pts.push({{ x: last.x - (dx / lines) * i, y: last.y - (dy / lines) * i, age: 0, force, vx, vy }});
        }}
      }}
    }}
    pts.push({{ x, y, age: 0, force: state.force || 0.6, vx, vy }});
  }}

  function drawTouch(point) {{
    const x = point.x * TRAIL_SIZE;
    const y = (1 - point.y) * TRAIL_SIZE;
    const intensity = Math.max(0, 1 - point.age / settings.maxAge) * point.force;
    trailCtx.globalCompositeOperation = 'screen';
    const radius = TRAIL_SIZE * settings.trailSize * intensity;
    if (radius <= 0) return;
    const grad = trailCtx.createRadialGradient(x, y, Math.max(0, radius * 0.25), x, y, radius);
    grad.addColorStop(0, 'rgba(255, 255, 255, 0.25)');
    grad.addColorStop(1, 'rgba(0, 0, 0, 0.0)');
    trailCtx.beginPath();
    trailCtx.fillStyle = grad;
    trailCtx.arc(x, y, radius, 0, Math.PI * 2);
    trailCtx.fill();

    const scale = AGE_SIZE / TRAIL_SIZE;
    const fresh = Math.round(255 * Math.min(Math.max(1 - point.age / settings.maxAge, 0), 1));
    const glaze = ageCtx.createRadialGradient(x * scale, y * scale, 0, x * scale, y * scale, radius * scale);
    glaze.addColorStop(0, `rgb(${{fresh}}, ${{fresh}}, ${{fresh}})`);
    glaze.addColorStop(0.75, `rgb(${{fresh}}, ${{fresh}}, ${{fresh}})`);
    glaze.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ageCtx.globalCompositeOperation = 'lighten';
    ageCtx.beginPath();
    ageCtx.fillStyle = glaze;
    ageCtx.arc(x * scale, y * scale, radius * scale, 0, Math.PI * 2);
    ageCtx.fill();
  }}

  function burst(x, y) {{
    const count = 16;
    for (let k = 0; k < count; k++) {{
      const angle = (k / count) * Math.PI * 2 + Math.random() * 0.2;
      const speed = 0.35 + Math.random() * 0.3;
      state.points.push({{
        x, y, age: 0, force: 1,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed
      }});
    }}
  }}

  function updateTrail(delta) {{
    trailCtx.globalCompositeOperation = 'source-over';
    trailCtx.fillStyle = 'black';
    trailCtx.fillRect(0, 0, TRAIL_SIZE, TRAIL_SIZE);

    ageCtx.globalCompositeOperation = 'source-over';
    ageCtx.fillStyle = 'black';
    ageCtx.fillRect(0, 0, AGE_SIZE, AGE_SIZE);

    state.points.forEach((point, i) => {{
      point.age += delta * 1000;
      if (point.age > settings.maxAge) state.points.splice(i, 1);
    }});

    const drag = Math.exp(-3 * delta);
    for (const p of state.points) {{
      p.vx *= drag;
      p.vy = p.vy * drag - settings.gravity * 1.6 * delta;
      p.x += p.vx * delta;
      p.y += p.vy * delta;
    }}
    state.points.forEach(drawTouch);
  }}

  function resize() {{
    const w = Math.max(1, container.clientWidth);
    const h = Math.max(1, container.clientHeight);
    state.width = w;
    state.height = h;
    const pw = Math.floor(w * state.ratio);
    const ph = Math.floor(h * state.ratio);
    if (canvas.width !== pw || canvas.height !== ph) {{
      canvas.width = pw;
      canvas.height = ph;
    }}
  }}

  function draw() {{
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, trailTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, trailCanvas);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, ageTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, ageCanvas);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);

    gl.uniform2f(uniforms.u_resolution, state.width * state.ratio, state.height * state.ratio);
    gl.uniform1f(uniforms.u_gridSize, settings.gridSize);
    gl.uniform3fv(uniforms.u_color, settings.color);
    gl.uniform3fv(uniforms.u_trailColor, settings.trailColor);
    gl.uniform1f(uniforms.u_ramp, 1);
    gl.uniform1f(uniforms.u_sigma, settings.goo * state.ratio);
    gl.uniform1f(uniforms.u_shine, settings.shine);
    gl.uniform1f(uniforms.u_glow, settings.glow);
    gl.uniform1i(uniforms.u_shape, settings.shape);
    gl.uniform1i(uniforms.u_decay, settings.decay);
    gl.uniform1f(uniforms.u_gap, settings.gap);
    gl.uniform1f(uniforms.u_sparkle, settings.sparkle);
    gl.uniform1f(uniforms.u_time, state.time);
    gl.uniform1f(uniforms.u_hasImage, 0);

    gl.bindVertexArray(vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }}

  function frame(now) {{
    if (!state.alive) return;
    const delta = state.last ? (now - state.last) / 1000 : 1 / 60;
    state.last = now;
    state.time += delta;
    resize();
    updateTrail(delta);
    draw();
    requestAnimationFrame(frame);
  }}

  function toUv(e) {{
    const rect = container.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const side = Math.max(rect.width, rect.height);
    return {{
      x: (x - rect.width / 2) / side + 0.5,
      y: (rect.height / 2 - y) / side + 0.5
    }};
  }}

  container.addEventListener('pointermove', function(e) {{
    const uv = toUv(e);
    const p = state.pointer;
    const now = performance.now();
    const elapsed = Math.max(0.008, (now - p.at) / 1000);
    p.vx = ((uv.x - p.x) / elapsed) * 0.3;
    p.vy = ((uv.y - p.y) / elapsed) * 0.3;
    p.x = uv.x;
    p.y = uv.y;
    p.at = now;
    addTouch(uv.x, uv.y, p.vx * 0.15, p.vy * 0.15);
  }});

  container.addEventListener('pointerdown', function(e) {{
    const uv = toUv(e);
    burst(uv.x, uv.y);
  }});

  // Initial welcome burst in center
  setTimeout(() => {{
    burst(0.5, 0.5);
  }}, 250);

  resize();
  requestAnimationFrame(frame);
}})();
</script>
</body>
</html>"""


def render_pixel_trail(
    height: int = 185,
    grid_size: int = 45,
    trail_size: float = 0.12,
    max_age: int = 500,
    color: str = "#A855F7",
    trail_color: str = "#EC4899",
    goo_strength: float = 2.0,
    glow: float = 0.55,
    title: str = "🔍 VaaniLens",
    subtitle: str = "Multilingual, Explainable Sentiment, Entity & Translation Intelligence",
):
    """Embeds the interactive WebGL2 PixelTrail hero into the Streamlit application."""
    html_code = get_pixel_trail_html(
        height=height,
        grid_size=grid_size,
        trail_size=trail_size,
        max_age=max_age,
        color=color,
        trail_color=trail_color,
        goo_strength=goo_strength,
        glow=glow,
        title=title,
        subtitle=subtitle,
    )
    components.html(html_code, height=height, scrolling=False)

