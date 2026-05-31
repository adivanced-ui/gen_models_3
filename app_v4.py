"""
app_v4.py — Версия 4: Function Calling — ИИ сам решает когда генерировать

Что добавилось по сравнению с v3:
  - Убрана отдельная кнопка "Нарисуй"
  - Модель сама анализирует запрос и решает: ответить текстом ИЛИ вызвать generate_image
  - Модель может вызвать второй инструмент calculate_expression
  - Модель может вызвать третий инструмент get_weather
  - Два запроса к API: 1-й — с tools, 2-й — получить итоговый ответ

Стек: Flask + OpenAI-совместимый API (chat: openrouter/free или free, генерация: gpt-image-1)
Запуск: python app_v4.py
Открой: http://localhost:5004
"""

import ast
import math
import os
import json
import base64
import operator
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI

# ═══════════════════════════════════════════════════════════════
#  КОНФИГУРАЦИЯ — все важные параметры здесь
# ═══════════════════════════════════════════════════════════════

# Провайдер и модель выбираются через переменные окружения.
# OpenRouter:
# LLM_PROVIDER=openrouter
# OPENROUTER_API_KEY=<твой_ключ>
# CHAT_MODEL=openrouter/free
#
# BotHub:
# LLM_PROVIDER=bothub
# BOTHUB_API_KEY=<твой_ключ>
# CHAT_MODEL=free
#
# Если чат идёт через OpenRouter, а картинки нужно генерировать через другой
# OpenAI-совместимый сервис, можно отдельно задать IMAGE_API_KEY и IMAGE_BASE_URL.
PROVIDER_CONFIG = {
    "openrouter": {
        "api_key_envs": ("OPENROUTER_API_KEY",),
        "base_url_envs": ("OPENROUTER_API_LINK", "OPENROUTER_BASE_URL"),
        "base_url": "https://openrouter.ai/api/v1",
        "chat_model": "openrouter/free",
    },
    "bothub": {
        "api_key_envs": ("BOTHUB_API_KEY",),
        "base_url_envs": ("BOTHUB_API_LINK", "BOTHUB_BASE_URL"),
        "base_url": "https://openai.bothub.chat/v1",
        "chat_model": "free",
    },
}


def first_env(names, default=""):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openrouter").strip().lower()
if LLM_PROVIDER not in PROVIDER_CONFIG:
    raise ValueError(
        f"Неизвестный LLM_PROVIDER={LLM_PROVIDER!r}. "
        f"Доступно: {', '.join(PROVIDER_CONFIG)}"
    )

provider_settings = PROVIDER_CONFIG[LLM_PROVIDER]
API_KEY = first_env(provider_settings["api_key_envs"], "missing-api-key")
BASE_URL = first_env(provider_settings["base_url_envs"], provider_settings["base_url"])

CHAT_MODEL = os.environ.get("CHAT_MODEL", provider_settings["chat_model"])
IMAGE_API_KEY = os.environ.get("IMAGE_API_KEY") or first_env(("BOTHUB_API_KEY",), API_KEY)
IMAGE_BASE_URL = os.environ.get("IMAGE_BASE_URL") or first_env(
    ("BOTHUB_API_LINK", "BOTHUB_BASE_URL"),
    BASE_URL
)
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "gpt-image-1")  # модель генерации изображений
IMAGE_SIZE = os.environ.get("IMAGE_SIZE", "1024x1024")      # размер генерируемого изображения
SYSTEM_PROMPT = (
    "Ты — полезный ассистент. Отвечай кратко. "
    "Когда пользователь просит нарисовать, создать, визуализировать "
    "или сгенерировать любое изображение — используй инструмент generate_image. "
    "Когда пользователь просит посчитать выражение, процент, степень, корень "
    "или другую точную арифметику — используй инструмент calculate_expression. "
    "Когда пользователь спрашивает текущую погоду в городе — используй инструмент get_weather."
)
PORT = 5004

# ─── Описание инструмента (JSON-схема для модели) ────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": (
                "Generates an image from a text description. "
                "Call this when the user wants to draw, create, visualize, "
                "or generate any visual content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Detailed image description in English. "
                            "Be specific about style, composition, colors."
                        )
                    }
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_expression",
            "description": (
                "Safely evaluates a mathematical expression. "
                "Call this for arithmetic, percentages, powers, roots, "
                "trigonometry, min/max, rounding, or exact numeric calculations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "Math expression using numbers and operators like "
                            "+, -, *, /, //, %, **. Functions allowed: sqrt, sin, "
                            "cos, tan, log, ln, abs, round, min, max. Constants: pi, e."
                        )
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Gets current weather for a city using a free weather API. "
                "Call this when the user asks about current weather, "
                "temperature, wind, humidity, or conditions in a location."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": (
                            "City name, optionally with country or region, "
                            "for example: Moscow, Paris, Tokyo, New York."
                        )
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# ═══════════════════════════════════════════════════════════════

app = Flask(__name__)


def make_client(api_key, base_url, default_headers=None):
    params = {"api_key": api_key, "base_url": base_url}
    if default_headers:
        params["default_headers"] = default_headers
    return OpenAI(**params)


openrouter_headers = None
if LLM_PROVIDER == "openrouter":
    openrouter_headers = {
        "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", f"http://localhost:{PORT}"),
        "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "genmod3-function-calling-homework"),
    }

client = make_client(API_KEY, BASE_URL, openrouter_headers)
image_client = make_client(IMAGE_API_KEY, IMAGE_BASE_URL)
HISTORY_FILE = Path(os.environ.get("HISTORY_FILE", "history.json"))


def load_history():
    if not HISTORY_FILE.exists():
        return []
    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return []


def save_history():
    with HISTORY_FILE.open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)


def clear_history_file():
    try:
        HISTORY_FILE.unlink()
    except FileNotFoundError:
        pass


history = load_history()

HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Чат v4 — Function Calling</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #f8fafc; color: #1e293b;
           display: flex; flex-direction: column; height: 100vh; }
    header { padding: 14px 20px; background: #ffffff; border-bottom: 1px solid #e2e8f0;
             font-weight: 700; font-size: 18px; display: flex; align-items: center; gap: 10px; }
    .tag { font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 10px;
           background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
    #chat { flex: 1; overflow-y: auto; padding: 20px; display: flex;
            flex-direction: column; gap: 12px; }
    .msg { max-width: 80%; padding: 10px 14px; border-radius: 12px;
           line-height: 1.6; font-size: 15px; white-space: pre-wrap; }
    .msg.user      { align-self: flex-end; background: #2563eb; color: white; }
    .msg.assistant { align-self: flex-start; background: #ffffff; color: #1e293b;
                     border: 1px solid #e2e8f0; box-shadow: 0 1px 3px #0001; }
    .msg.system    { align-self: center; font-size: 12px; color: #94a3b8; font-style: italic; }
    .msg.error     { align-self: center; background: #fee2e2; color: #dc2626;
                     border: 1px solid #fca5a5; font-size: 13px; }
    .msg.tool-call { align-self: flex-start; background: #fffbeb; color: #92400e;
                     border: 1px solid #fde68a; font-size: 12px; font-family: monospace; }
    .msg.image-result { align-self: flex-start; background: transparent; padding: 0; }
    .msg.image-result img { max-width: 380px; border-radius: 12px;
                             border: 2px solid #e2e8f0; display: block;
                             box-shadow: 0 4px 12px #0002; }
    .msg img.inline { max-width: 220px; border-radius: 8px; display: block; margin-bottom: 6px; }
    #form { padding: 12px 20px 16px; background: #ffffff; border-top: 1px solid #e2e8f0; }
    #preview-area { display: none; align-items: center; gap: 8px; margin-bottom: 10px;
                    padding: 8px 10px; background: #f1f5f9; border-radius: 8px; }
    #preview-area img { max-height: 80px; border-radius: 6px; border: 1px solid #e2e8f0; }
    #preview-area button { background: none; border: none; color: #94a3b8;
                           cursor: pointer; font-size: 18px; }
    #inputs { display: flex; gap: 8px; }
    #file-label { padding: 10px 12px; border-radius: 8px; background: #f1f5f9; color: #64748b;
                  cursor: pointer; font-size: 18px; border: 1px solid #e2e8f0;
                  display: flex; align-items: center; }
    #file-label:hover { background: #e2e8f0; }
    #input { flex: 1; padding: 10px 14px; border-radius: 8px; border: 1px solid #cbd5e1;
             background: #f8fafc; color: #1e293b; font-size: 15px; outline: none;
             resize: none; height: 44px; }
    #input:focus { border-color: #2563eb; box-shadow: 0 0 0 3px #dbeafe; }
    .btn { padding: 10px 18px; border-radius: 8px; border: none; cursor: pointer;
           font-size: 14px; font-weight: 600; transition: opacity .2s; }
    .btn:hover    { opacity: 0.85; }
    .btn:disabled { opacity: 0.4; cursor: not-allowed; }
    .btn-send  { background: #2563eb; color: white; }
    .btn-reset { background: #f1f5f9; color: #64748b; }
  </style>
</head>
<body>
  <header>
    🤖 Чат-бот
    <span class="tag">v4 — Function Calling</span>
  </header>

  <div id="chat">
    <div class="msg system">
      Умный чат: просите нарисовать что-нибудь — модель сама решит вызвать генерацию.<br>
      Например: «Нарисуй закат над горами» или «Создай логотип для кафе»
    </div>
  </div>

  <div id="form">
    <div id="preview-area">
      <img id="preview-img" src="" alt="">
      <span style="font-size:13px;color:#64748b;flex:1" id="preview-name"></span>
      <button onclick="clearImage()">✕</button>
    </div>
    <div id="inputs">
      <label id="file-label" for="file-input" title="Прикрепить изображение">📎</label>
      <input id="file-input" type="file" accept="image/*" style="display:none">
      <textarea id="input" placeholder='Напишите сообщение. Например: "Нарисуй кота-астронавта"'></textarea>
      <button class="btn btn-send"  id="send-btn">Отправить</button>
      <button class="btn btn-reset" id="reset-btn">↺</button>
    </div>
  </div>

  <script>
    const chat    = document.getElementById('chat');
    const input   = document.getElementById('input');
    const sendBtn = document.getElementById('send-btn');
    let selectedFile = null;

    document.getElementById('file-input').onchange = (e) => {
      selectedFile = e.target.files[0];
      if (!selectedFile) return;
      document.getElementById('preview-img').src = URL.createObjectURL(selectedFile);
      document.getElementById('preview-name').textContent = selectedFile.name;
      document.getElementById('preview-area').style.display = 'flex';
    };

    function clearImage() {
      selectedFile = null;
      document.getElementById('file-input').value = '';
      document.getElementById('preview-area').style.display = 'none';
    }

    function addMsg(text, role) {
      const div = document.createElement('div');
      div.className = 'msg ' + role;
      div.textContent = text;
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
      return div;
    }

    async function sendMessage() {
      const text = input.value.trim();
      if (!text && !selectedFile) return;

      const formData = new FormData();
      if (text)         formData.append('message', text);
      if (selectedFile) formData.append('image', selectedFile);

      const userDiv = document.createElement('div');
      userDiv.className = 'msg user';
      if (selectedFile) {
        const img = document.createElement('img');
        img.className = 'inline';
        img.src = URL.createObjectURL(selectedFile);
        userDiv.appendChild(img);
      }
      if (text) userDiv.appendChild(document.createTextNode(text));
      chat.appendChild(userDiv);
      chat.scrollTop = chat.scrollHeight;

      input.value = '';
      clearImage();
      sendBtn.disabled = true;

      const thinking = addMsg('🤔 Думаю...', 'assistant');

      try {
        const res  = await fetch('/chat', { method: 'POST', body: formData });
        const data = await res.json();

        thinking.remove();

        if (!res.ok || data.error) {
          addMsg('Ошибка: ' + (data.error || res.status), 'error');
          return;
        }

        // Если были вызваны функции — показываем детали каждого вызова
        if (data.tool_calls && data.tool_calls.length) {
          data.tool_calls.forEach(call => {
            const args = JSON.stringify(call.arguments, null, 2);
            const result = call.result ? '\\nРезультат: ' + call.result : '';
            addMsg('Вызов инструмента: ' + call.name + '\\nАргументы: ' + args + result, 'tool-call');
          });
        }

        // Показываем сгенерированные изображения
        const imageUrls = data.image_urls || (data.image_url ? [data.image_url] : []);
        imageUrls.forEach(imageUrl => {
          const imgDiv = document.createElement('div');
          imgDiv.className = 'msg image-result';
          const img = document.createElement('img');
          img.src = imageUrl;
          img.alt = 'Сгенерированное изображение';
          imgDiv.appendChild(img);
          chat.appendChild(imgDiv);
          chat.scrollTop = chat.scrollHeight;
        });

        // Текстовый ответ модели
        if (data.reply) addMsg(data.reply, 'assistant');

      } catch (e) {
        thinking.remove();
        addMsg('Ошибка: ' + e.message, 'error');
      } finally {
        sendBtn.disabled = false;
        input.focus();
      }
    }

    sendBtn.onclick = sendMessage;
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });
    document.getElementById('reset-btn').onclick = async () => {
      await fetch('/reset', { method: 'POST' });
      chat.innerHTML = '<div class="msg system">История сброшена.</div>';
    };
  </script>
</body>
</html>
"""

# ─── Вспомогательная функция: генерация изображения ─────────────────────────

def do_generate_image(prompt: str) -> str:
    """Вызывает gpt-image-1, возвращает изображение как data URL (base64)."""
    response = image_client.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size=IMAGE_SIZE,
        n=1
    )
    # gpt-image-1 возвращает base64 (b64_json), не URL
    b64 = response.data[0].b64_json
    return f"data:image/png;base64,{b64}"


# ─── Вспомогательная функция: калькулятор ────────────────────────

ALLOWED_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

ALLOWED_NAMES = {
    "pi": math.pi,
    "e": math.e,
}

ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log10,
    "ln": math.log,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}


def ensure_reasonable_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("only numerics are allowed!")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("number must be finite!")
    if abs(value) > 10 ** 100:
        raise ValueError("result too big!")
    return value


def eval_math_node(node):
    if isinstance(node, ast.Expression):
        return eval_math_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numerics are allowed!")
        return node.value

    if isinstance(node, ast.Name):
        if node.id not in ALLOWED_NAMES:
            raise ValueError(f"Unknown const: {node.id}")
        return ALLOWED_NAMES[node.id]

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_BINARY_OPERATORS:
            raise ValueError("Unresolved operator!")
        left = eval_math_node(node.left)
        right = eval_math_node(node.right)
        if op_type is ast.Pow and abs(right) > 12:
            raise ValueError("POW too big!")
        return ensure_reasonable_number(ALLOWED_BINARY_OPERATORS[op_type](left, right))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_UNARY_OPERATORS:
            raise ValueError("Unary operator not allowed!")
        return ensure_reasonable_number(ALLOWED_UNARY_OPERATORS[op_type](eval_math_node(node.operand)))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
            raise ValueError("function not allowed!")
        if node.keywords:
            raise ValueError("named args unsupported!")
        if len(node.args) > 5:
            raise ValueError("Too many function args!")
        args = [eval_math_node(arg) for arg in node.args]
        return ensure_reasonable_number(ALLOWED_FUNCTIONS[node.func.id](*args))

    raise ValueError("Expression contains Unresolved syntax!")


def format_number(value):
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def do_calculate_expression(expression: str) -> str:
    """Безопасно считает арифметическое выражение без eval()."""
    expression = expression.strip().replace("^", "**")
    if not expression:
        raise ValueError("empty expression")
    if len(expression) > 200:
        raise ValueError("expression too long!")

    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Unresolved math expression!") from exc

    result = ensure_reasonable_number(eval_math_node(parsed))
    return format_number(result)


# ─── Вспомогательная функция: погода через Open-Meteo ───────────────────────

WEATHER_CODES = {
    0: "ясно",
    1: "преимущественно ясно",
    2: "переменная облачность",
    3: "пасмурно",
    45: "туман",
    48: "изморозь и туман",
    51: "слабая морось",
    53: "умеренная морось",
    55: "сильная морось",
    61: "слабый дождь",
    63: "умеренный дождь",
    65: "сильный дождь",
    71: "слабый снег",
    73: "умеренный снег",
    75: "сильный снег",
    80: "слабые ливни",
    81: "умеренные ливни",
    82: "сильные ливни",
    95: "гроза",
    96: "гроза с градом",
    99: "сильная гроза с градом",
}


def fetch_json(url):
    with urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def do_get_weather(city: str) -> str:
    """Возвращает текущую погоду по названию города через Open-Meteo."""
    city = city.strip()
    if not city:
        raise ValueError("get_weather needs city")

    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search?" + urlencode({
        "name": city,
        "count": 1,
        "language": "ru",
        "format": "json",
    })
    geocoding_data = fetch_json(geocoding_url)
    locations = geocoding_data.get("results") or []
    if not locations:
        raise ValueError(f"City not found: {city}")

    location = locations[0]
    latitude = location["latitude"]
    longitude = location["longitude"]
    location_name = location.get("name", city)
    country = location.get("country", "")
    admin1 = location.get("admin1", "")
    place = ", ".join(part for part in (location_name, admin1, country) if part)

    forecast_url = "https://api.open-meteo.com/v1/forecast?" + urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join([
            "temperature_2m",
            "apparent_temperature",
            "relative_humidity_2m",
            "precipitation",
            "weather_code",
            "wind_speed_10m",
        ]),
        "timezone": "auto",
    })
    forecast_data = fetch_json(forecast_url)
    current = forecast_data.get("current") or {}
    units = forecast_data.get("current_units") or {}

    weather_code = current.get("weather_code")
    description = WEATHER_CODES.get(weather_code, f"код погоды {weather_code}")

    return (
        f"{place}: {description}. "
        f"Температура {current.get('temperature_2m')} {units.get('temperature_2m', '°C')}, "
        f"ощущается как {current.get('apparent_temperature')} "
        f"{units.get('apparent_temperature', '°C')}. "
        f"Влажность {current.get('relative_humidity_2m')} "
        f"{units.get('relative_humidity_2m', '%')}, "
        f"ветер {current.get('wind_speed_10m')} {units.get('wind_speed_10m', 'км/ч')}, "
        f"осадки {current.get('precipitation')} {units.get('precipitation', 'мм')}."
    )


def execute_tool(function_name, args):
    if function_name == "generate_image":
        prompt = (args.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("generate_image needs prompt!")
        image_url = do_generate_image(prompt)
        return {
            "content": "Image generated successfully.",
            "public_result": "Изображение сгенерировано",
            "image_url": image_url,
        }

    if function_name == "calculate_expression":
        expression = (args.get("expression") or "").strip()
        if not expression:
            raise ValueError("calculate_expression needs expression")
        result = do_calculate_expression(expression)
        return {
            "content": json.dumps(
                {"expression": expression, "result": result},
                ensure_ascii=False
            ),
            "public_result": f"{expression} = {result}",
        }

    if function_name == "get_weather":
        city = (args.get("city") or "").strip()
        if not city:
            raise ValueError("get_weather needs city")
        weather = do_get_weather(city)
        return {
            "content": json.dumps(
                {"city": city, "weather": weather},
                ensure_ascii=False
            ),
            "public_result": weather,
        }

    raise ValueError(f"Неизвестный инструмент: {function_name}")


# ─── Маршруты ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.form.get('message', '').strip()
        image_file   = request.files.get('image')

        # ── Формируем content (текст или текст+картинка) ──────────────────
        if image_file and image_file.filename:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            mime_type  = image_file.content_type or 'image/jpeg'
            content = [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
                {"type": "text", "text": user_message or "Опиши картинку"}
            ]
            history_text = f"[картинка] {user_message}" if user_message else "[картинка]"
        else:
            if not user_message:
                return jsonify({"error": "Пустое сообщение"}), 400
            content      = user_message
            history_text = user_message

        history.append({"role": "user", "content": content})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

        # ── Первый запрос: передаём tools, модель может вернуть tool_call ──
        resp1 = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOLS
        )

        msg = resp1.choices[0].message
        tool_calls = msg.tool_calls or []
        image_urls = []
        tool_calls_info = []

        # ── Проверяем: хочет ли модель вызвать функцию? ───────────────────
        if tool_calls:
            # Добавляем в messages ответ ассистента со всеми tool_calls.
            messages.append({
                "role":       "assistant",
                "content":    msg.content,
                "tool_calls": [tc.model_dump() for tc in tool_calls]
            })

            # Выполняем каждый tool_call. Модель только попросила вызвать функцию,
            # а реальную работу делает наш Python-код.
            for tc in tool_calls:
                function_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Некорректные аргументы для {function_name}: {exc}"
                    ) from exc

                tool_result = execute_tool(function_name, args)
                if tool_result.get("image_url"):
                    image_urls.append(tool_result["image_url"])

                tool_calls_info.append({
                    "name": function_name,
                    "arguments": args,
                    "result": tool_result.get("public_result", tool_result["content"]),
                })

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      tool_result["content"]
                })

            # ── Второй запрос: получаем финальный текстовый ответ ─────────
            resp2 = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages
            )
            reply = resp2.choices[0].message.content
        else:
            # Нет tool_call — обычный текстовый ответ
            reply = msg.content

        # ── Обновляем историю (без base64 чтобы не раздувать контекст) ────
        history[-1] = {"role": "user", "content": history_text}
        history.append({"role": "assistant", "content": reply or ""})
        save_history()

        result = {
            "reply": reply,
            "tool_called": bool(tool_calls_info),
            "tool_calls": tool_calls_info,
        }
        if image_urls:
            result["image_urls"] = image_urls
            result["image_url"] = image_urls[0]  # совместимость со старым фронтендом
        image_tool = next(
            (call for call in tool_calls_info if call["name"] == "generate_image"),
            None
        )
        if image_tool:
            result["tool_prompt"] = image_tool["arguments"].get("prompt", "")
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset():
    history.clear()
    clear_history_file()
    return jsonify({"status": "ok"})


# ─── Запуск ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    debug_mode = os.environ.get("FLASK_DEBUG", "1").lower() not in {"0", "false", "no"}
    print("=" * 50)
    print(f"Чат-бот v4 — Function Calling  |  http://localhost:{PORT}")
    print(f"Провайдер: {LLM_PROVIDER}  |  Chat model: {CHAT_MODEL}")
    print(f"Base URL: {BASE_URL}")
    print(f"Image model: {IMAGE_MODEL}  |  Image base URL: {IMAGE_BASE_URL}")
    print("=" * 50)
    app.run(debug=debug_mode, port=PORT, use_reloader=debug_mode)
