"""Krishi Mitra - camera feed -> Gemini (free tier) -> disease + treatment + pump time + live weather."""
import io, json, os, re, time
import cv2, requests, streamlit as st
from PIL import Image
from google import genai
from google.genai import types

# =====================  EDIT THESE  =====================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6IQBAIR_mt4xmYhiVnnp_Q47t9uxmkCGQTp7HWxD23ymw")
PUMP_ML_PER_SEC = 50.0          # farmer's pump flow
DEFAULT_CITY = "Ludhiana"       # weather location (can be changed in the app)
FALLBACK_MODELS = ["gemini-3.5-flash-lite"]
# ========================================================

st.set_page_config(page_title="Krishi Mitra", page_icon="🌾", layout="wide")

# ---------------- translations ----------------
LANGS = {"English": "en", "हिन्दी": "hi", "ਪੰਜਾਬੀ": "pa"}
AI_LANG = {"en": "English", "hi": "Hindi (Devanagari script)", "pa": "Punjabi (Gurmukhi script)"}

T = {
 "en": dict(tag="Live camera → crop & disease detection → treatment → dose & pump time",
  camera="1 · Camera", live="📡 Live URL", grab="Grab frame from URL", upload="📷 Upload / webcam",
  photo="Leaf / plant photo", usecam="Or use this device's camera", frame="Frame to analyze",
  nourl="Enter the camera URL in the sidebar.", connecting="Connecting...",
  field="2 · Field", area="Field area", unit="Unit", analyze="🔍 Analyze crop",
  diag="3 · Diagnosis & plan", hint="Grab a frame, enter the field area, and press **Analyze crop**.",
  nokey="Put your Gemini API key in app.py (GEMINI_API_KEY).", examining="Analyzing the plant...",
  noplant="No clear plant found", tryagain="Try a closer, sharp photo of a leaf.",
  plant="Plant", problem="Disease / problem", conf="confidence", symptoms="Symptoms seen",
  healthy_msg="Plant looks healthy. No spraying needed. ✅", rec="Recommended",
  product="Product needed", water="Water to mix", pump="Pump ON time", min="min",
  pumpnote="Pump time = total spray liquid ({ml:,.0f} ml) ÷ {rate:g} ml/s. Mix the product into the water first; the pump moves the mixed liquid.",
  notes="Application notes", organic="🌱 Organic option", safety="⚠️ Safety",
  warn="AI suggestion only. Check the dose on the product label or with your local KVK / agriculture officer before spraying.",
  sidebar="📡 Camera", url="Camera URL", urlcap="MJPEG / RTSP / snapshot (.jpg) URL, e.g. 'IP Webcam' app.",
  weather="🌤️ Live weather", city="Location", feels="Feels like", humidity="Humidity", wind="Wind", rain="Rain chance",
  fc="5-day forecast", spray_ok="✅ Good time to spray", spray_rain="🌧️ Don't spray now: rain expected, it will wash the spray off.",
  spray_wind="💨 Don't spray now: wind is too strong, spray will drift.", wfail="Couldn't load weather. Check internet / city name.",
  health=dict(healthy="HEALTHY", mild="MILD", moderate="MODERATE", severe="SEVERE"),
  types=dict(pesticide="pesticide", fungicide="fungicide", insecticide="insecticide", fertilizer="fertilizer"),
  wx=dict(clear="Clear sky", mostly="Mostly clear", partly="Partly cloudy", overcast="Overcast", fog="Fog",
          drizzle="Drizzle", rain="Rain", snow="Snow", storm="Thunderstorm")),
 "hi": dict(tag="लाइव कैमरा → फसल व रोग की पहचान → उपचार → मात्रा व पंप का समय",
  camera="१ · कैमरा", live="📡 लाइव URL", grab="URL से फ्रेम लें", upload="📷 फोटो अपलोड / वेबकैम",
  photo="पत्ती / पौधे की फोटो", usecam="या इस डिवाइस का कैमरा इस्तेमाल करें", frame="जांच के लिए फ्रेम",
  nourl="साइडबार में कैमरा URL डालें।", connecting="कनेक्ट हो रहा है...",
  field="२ · खेत", area="खेत का क्षेत्रफल", unit="इकाई", analyze="🔍 फसल की जांच करें",
  diag="३ · जांच व योजना", hint="फ्रेम लें, खेत का क्षेत्रफल डालें और **फसल की जांच करें** दबाएं।",
  nokey="app.py में अपनी Gemini API key डालें (GEMINI_API_KEY)।", examining="पौधे की जांच हो रही है...",
  noplant="कोई साफ पौधा नहीं मिला", tryagain="पत्ती की नज़दीक से साफ फोटो लें।",
  plant="पौधा", problem="रोग / समस्या", conf="भरोसा", symptoms="दिखे लक्षण",
  healthy_msg="पौधा स्वस्थ दिखता है। छिड़काव की ज़रूरत नहीं। ✅", rec="सुझाव",
  product="दवा की मात्रा", water="मिलाने वाला पानी", pump="पंप चलाने का समय", min="मिनट",
  pumpnote="पंप समय = कुल छिड़काव घोल ({ml:,.0f} मि.ली.) ÷ {rate:g} मि.ली./सेकंड। पहले दवा को पानी में मिलाएं।",
  notes="छिड़काव के नोट्स", organic="🌱 जैविक विकल्प", safety="⚠️ सावधानी",
  warn="यह केवल AI का सुझाव है। छिड़काव से पहले दवा के लेबल या अपने कृषि अधिकारी / KVK से मात्रा ज़रूर जांचें।",
  sidebar="📡 कैमरा", url="कैमरा URL", urlcap="MJPEG / RTSP / फोटो (.jpg) URL, जैसे 'IP Webcam' ऐप।",
  weather="🌤️ लाइव मौसम", city="स्थान", feels="महसूस होता है", humidity="नमी", wind="हवा", rain="बारिश की संभावना",
  fc="5 दिन का पूर्वानुमान", spray_ok="✅ छिड़काव के लिए अच्छा समय", spray_rain="🌧️ अभी छिड़काव न करें: बारिश की संभावना, दवा धुल जाएगी।",
  spray_wind="💨 अभी छिड़काव न करें: हवा तेज़ है, दवा उड़ जाएगी।", wfail="मौसम लोड नहीं हुआ। इंटरनेट / शहर का नाम जांचें।",
  health=dict(healthy="स्वस्थ", mild="हल्का", moderate="मध्यम", severe="गंभीर"),
  types=dict(pesticide="कीटनाशक", fungicide="फफूंदनाशक", insecticide="कीटनाशक", fertilizer="खाद"),
  wx=dict(clear="साफ आसमान", mostly="ज़्यादातर साफ", partly="आंशिक बादल", overcast="घने बादल", fog="कोहरा",
          drizzle="बूंदाबांदी", rain="बारिश", snow="बर्फ", storm="आंधी-तूफान")),
 "pa": dict(tag="ਲਾਈਵ ਕੈਮਰਾ → ਫ਼ਸਲ ਤੇ ਬਿਮਾਰੀ ਦੀ ਪਛਾਣ → ਇਲਾਜ → ਮਾਤਰਾ ਤੇ ਪੰਪ ਦਾ ਸਮਾਂ",
  camera="੧ · ਕੈਮਰਾ", live="📡 ਲਾਈਵ URL", grab="URL ਤੋਂ ਫ੍ਰੇਮ ਲਓ", upload="📷 ਫੋਟੋ ਅੱਪਲੋਡ / ਵੈਬਕੈਮ",
  photo="ਪੱਤੇ / ਪੌਦੇ ਦੀ ਫੋਟੋ", usecam="ਜਾਂ ਇਸ ਡਿਵਾਈਸ ਦਾ ਕੈਮਰਾ ਵਰਤੋ", frame="ਜਾਂਚ ਲਈ ਫ੍ਰੇਮ",
  nourl="ਸਾਈਡਬਾਰ ਵਿੱਚ ਕੈਮਰਾ URL ਪਾਓ।", connecting="ਕਨੈਕਟ ਹੋ ਰਿਹਾ ਹੈ...",
  field="੨ · ਖੇਤ", area="ਖੇਤ ਦਾ ਰਕਬਾ", unit="ਇਕਾਈ", analyze="🔍 ਫ਼ਸਲ ਦੀ ਜਾਂਚ ਕਰੋ",
  diag="੩ · ਜਾਂਚ ਤੇ ਯੋਜਨਾ", hint="ਫ੍ਰੇਮ ਲਓ, ਖੇਤ ਦਾ ਰਕਬਾ ਪਾਓ ਤੇ **ਫ਼ਸਲ ਦੀ ਜਾਂਚ ਕਰੋ** ਦਬਾਓ।",
  nokey="app.py ਵਿੱਚ ਆਪਣੀ Gemini API key ਪਾਓ (GEMINI_API_KEY)।", examining="ਪੌਦੇ ਦੀ ਜਾਂਚ ਹੋ ਰਹੀ ਹੈ...",
  noplant="ਕੋਈ ਸਾਫ਼ ਪੌਦਾ ਨਹੀਂ ਮਿਲਿਆ", tryagain="ਪੱਤੇ ਦੀ ਨੇੜਿਓਂ ਸਾਫ਼ ਫੋਟੋ ਲਓ।",
  plant="ਪੌਦਾ", problem="ਬਿਮਾਰੀ / ਸਮੱਸਿਆ", conf="ਭਰੋਸਾ", symptoms="ਦਿਸੇ ਲੱਛਣ",
  healthy_msg="ਪੌਦਾ ਸਿਹਤਮੰਦ ਹੈ। ਛਿੜਕਾਅ ਦੀ ਲੋੜ ਨਹੀਂ। ✅", rec="ਸਿਫ਼ਾਰਸ਼",
  product="ਦਵਾਈ ਦੀ ਮਾਤਰਾ", water="ਮਿਲਾਉਣ ਵਾਲਾ ਪਾਣੀ", pump="ਪੰਪ ਚਲਾਉਣ ਦਾ ਸਮਾਂ", min="ਮਿੰਟ",
  pumpnote="ਪੰਪ ਸਮਾਂ = ਕੁੱਲ ਛਿੜਕਾਅ ਘੋਲ ({ml:,.0f} ਮਿ.ਲੀ.) ÷ {rate:g} ਮਿ.ਲੀ./ਸਕਿੰਟ। ਪਹਿਲਾਂ ਦਵਾਈ ਨੂੰ ਪਾਣੀ ਵਿੱਚ ਮਿਲਾਓ।",
  notes="ਛਿੜਕਾਅ ਦੇ ਨੋਟਸ", organic="🌱 ਜੈਵਿਕ ਵਿਕਲਪ", safety="⚠️ ਸਾਵਧਾਨੀ",
  warn="ਇਹ ਸਿਰਫ਼ AI ਦਾ ਸੁਝਾਅ ਹੈ। ਛਿੜਕਾਅ ਤੋਂ ਪਹਿਲਾਂ ਦਵਾਈ ਦੇ ਲੇਬਲ ਜਾਂ ਆਪਣੇ ਖੇਤੀਬਾੜੀ ਅਫ਼ਸਰ / KVK ਤੋਂ ਮਾਤਰਾ ਜ਼ਰੂਰ ਜਾਂਚੋ।",
  sidebar="📡 ਕੈਮਰਾ", url="ਕੈਮਰਾ URL", urlcap="MJPEG / RTSP / ਫੋਟੋ (.jpg) URL, ਜਿਵੇਂ 'IP Webcam' ਐਪ।",
  weather="🌤️ ਲਾਈਵ ਮੌਸਮ", city="ਥਾਂ", feels="ਮਹਿਸੂਸ ਹੁੰਦਾ", humidity="ਨਮੀ", wind="ਹਵਾ", rain="ਮੀਂਹ ਦੀ ਸੰਭਾਵਨਾ",
  fc="5 ਦਿਨ ਦਾ ਪੂਰਵ-ਅਨੁਮਾਨ", spray_ok="✅ ਛਿੜਕਾਅ ਲਈ ਚੰਗਾ ਸਮਾਂ", spray_rain="🌧️ ਹੁਣੇ ਛਿੜਕਾਅ ਨਾ ਕਰੋ: ਮੀਂਹ ਦੀ ਸੰਭਾਵਨਾ, ਦਵਾਈ ਧੋਤੀ ਜਾਵੇਗੀ।",
  spray_wind="💨 ਹੁਣੇ ਛਿੜਕਾਅ ਨਾ ਕਰੋ: ਹਵਾ ਤੇਜ਼ ਹੈ, ਦਵਾਈ ਉੱਡ ਜਾਵੇਗੀ।", wfail="ਮੌਸਮ ਲੋਡ ਨਹੀਂ ਹੋਇਆ। ਇੰਟਰਨੈੱਟ / ਸ਼ਹਿਰ ਦਾ ਨਾਮ ਜਾਂਚੋ।",
  health=dict(healthy="ਸਿਹਤਮੰਦ", mild="ਹਲਕਾ", moderate="ਦਰਮਿਆਨਾ", severe="ਗੰਭੀਰ"),
  types=dict(pesticide="ਕੀਟਨਾਸ਼ਕ", fungicide="ਉੱਲੀਨਾਸ਼ਕ", insecticide="ਕੀਟਨਾਸ਼ਕ", fertilizer="ਖਾਦ"),
  wx=dict(clear="ਸਾਫ਼ ਅਸਮਾਨ", mostly="ਜ਼ਿਆਦਾਤਰ ਸਾਫ਼", partly="ਕੁਝ ਬੱਦਲ", overcast="ਸੰਘਣੇ ਬੱਦਲ", fog="ਧੁੰਦ",
          drizzle="ਕਣੀਆਂ", rain="ਮੀਂਹ", snow="ਬਰਫ਼", storm="ਹਨੇਰੀ-ਤੂਫ਼ਾਨ")),
}

# ---------------- styling + language picker ----------------
st.markdown("""
<style>
.block-container{padding-top:1.2rem;max-width:1200px}
.hero{background:linear-gradient(120deg,#166534,#22c55e);color:#fff;padding:1.3rem 1.8rem;border-radius:18px;margin-bottom:1rem}
.hero h1{margin:0;font-size:2rem}.hero p{margin:.3rem 0 0;opacity:.92}
.card{background:var(--secondary-background-color);border-radius:16px;padding:1rem 1.2rem;margin-bottom:.8rem;border:1px solid rgba(128,128,128,.2)}
.big{font-size:2.1rem;font-weight:700;line-height:1.15}
.lbl{font-size:.78rem;opacity:.7;text-transform:uppercase;letter-spacing:.05em}
.pill{display:inline-block;padding:.15rem .7rem;border-radius:999px;font-weight:600;font-size:.85rem}
.ok{background:#dcfce7;color:#166534}.warn{background:#fef9c3;color:#854d0e}.bad{background:#fee2e2;color:#991b1b}
.wxnow{display:flex;align-items:center;gap:1rem}.wxicon{font-size:4rem;line-height:1}
.fc{text-align:center;padding:.6rem .2rem}.fc .i{font-size:1.9rem}.fc .d{font-size:.8rem;opacity:.7}
</style>""", unsafe_allow_html=True)

_, lang_col = st.columns([5, 1.3])
lang_name = lang_col.selectbox("🌐", list(LANGS), label_visibility="collapsed")
L = LANGS[lang_name]; t = T[L]

st.markdown(f'<div class="hero"><h1>🌾 Krishi Mitra</h1><p>{t["tag"]}</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header(t["sidebar"])
    stream_url = st.text_input(t["url"], placeholder="http://192.168.1.50:8080/video")
    st.caption(t["urlcap"])

# ---------------- weather ----------------
def wx_group(code: int):
    if code == 0: return "clear", "☀️"
    if code == 1: return "mostly", "🌤️"
    if code == 2: return "partly", "⛅"
    if code == 3: return "overcast", "☁️"
    if code in (45, 48): return "fog", "🌫️"
    if code in (51, 53, 55, 56, 57): return "drizzle", "🌦️"
    if code in (61, 63, 65, 66, 67, 80, 81, 82): return "rain", "🌧️"
    if code in (71, 73, 75, 77, 85, 86): return "snow", "❄️"
    if code >= 95: return "storm", "⛈️"
    return "partly", "⛅"

@st.cache_data(ttl=600, show_spinner=False)
def get_weather(city: str):
    g = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": city, "count": 1}, timeout=8).json()["results"][0]
    w = requests.get("https://api.open-meteo.com/v1/forecast", timeout=8, params={
        "latitude": g["latitude"], "longitude": g["longitude"], "timezone": "auto", "forecast_days": 5,
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"}).json()
    return g["name"], w

with st.expander(t["weather"], expanded=True):
    city = st.text_input(t["city"], DEFAULT_CITY)
    try:
        cname, w = get_weather(city)
        cur, day = w["current"], w["daily"]
        grp, icon = wx_group(cur["weather_code"])
        rain_p = day["precipitation_probability_max"][0] or 0
        st.markdown(f"""<div class="card wxnow"><div class="wxicon">{icon}</div><div>
        <div class="lbl">{cname}</div><div class="big">{cur['temperature_2m']:.0f}°C</div>
        <div>{t['wx'][grp]} · {t['feels']} {cur['apparent_temperature']:.0f}°C</div></div></div>""", unsafe_allow_html=True)
        a, b, c = st.columns(3)
        a.metric("💧 " + t["humidity"], f"{cur['relative_humidity_2m']}%")
        b.metric("💨 " + t["wind"], f"{cur['wind_speed_10m']:.0f} km/h")
        c.metric("🌧️ " + t["rain"], f"{rain_p}%")
        if cur["precipitation"] > 0 or rain_p >= 50 or grp in ("rain", "storm", "drizzle"):
            st.warning(t["spray_rain"])
        elif cur["wind_speed_10m"] >= 15:
            st.warning(t["spray_wind"])
        else:
            st.success(t["spray_ok"])
        st.caption(t["fc"])
        cols = st.columns(5)
        for i, col in enumerate(cols):
            _, ic = wx_group(day["weather_code"][i])
            col.markdown(f"""<div class="card fc"><div class="d">{day['time'][i][8:]}/{day['time'][i][5:7]}</div>
            <div class="i">{ic}</div><b>{day['temperature_2m_max'][i]:.0f}°</b>
            <span class="d">{day['temperature_2m_min'][i]:.0f}°</span><div class="d">🌧️ {day['precipitation_probability_max'][i] or 0}%</div></div>""",
            unsafe_allow_html=True)
    except Exception:
        st.error(t["wfail"])

# ---------------- camera + AI helpers ----------------
def grab_frame(url: str) -> Image.Image:
    try:
        r = requests.get(url, timeout=6, stream=True)
        if r.headers.get("content-type", "").startswith("image/"):
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        r.close()
    except Exception:
        pass
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError("Could not open the camera URL. Check the address and that the camera is on the same network.")
    frame = None
    for _ in range(8):
        ok, f = cap.read()
        if ok: frame = f
    cap.release()
    if frame is None:
        raise RuntimeError("Connected, but no frame received.")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

PROMPT = """You are an expert Indian agronomist (Punjab/North India conditions).
Look at this crop image and respond ONLY with JSON in this exact schema:
{
 "plant_detected": bool,
 "plant_name": str,
 "plant_confidence": number 0-100,
 "health_status": "healthy" | "mild" | "moderate" | "severe",
 "disease_or_problem": str,
 "disease_type": "fungal"|"bacterial"|"viral"|"pest"|"nutrient deficiency"|"none",
 "disease_confidence": number 0-100,
 "symptoms_seen": [str],
 "treatment_type": "pesticide"|"fungicide"|"insecticide"|"fertilizer"|"none",
 "recommended_product": str,
 "dose_per_acre_ml_or_g": number,
 "dose_unit": "ml"|"g",
 "water_per_acre_litres": number,
 "application_notes": [str],
 "safety_notes": [str],
 "organic_alternative": str
}
Rules: keep all enum values and keys exactly in English. Write every free-text value
(plant_name, disease_or_problem, symptoms_seen, recommended_product, application_notes, safety_notes,
organic_alternative) in {LANG}. Use "None" equivalent in {LANG} when healthy.
If the image is blurry or has no plant, set plant_detected=false and explain in disease_or_problem.
Be honest about uncertainty; do not guess a disease you cannot see."""

@st.cache_data(ttl=600, show_spinner=False)
def pick_models(key: str):
    try:
        out = [m.name.replace("models/", "") for m in genai.Client(api_key=key).models.list()
               if "generateContent" in (getattr(m, "supported_actions", None) or [])
               and "flash" in m.name and not re.search(r"image|tts|audio|live|embed|robotics|computer", m.name)]
        ver = lambda n: [int(x) for x in re.findall(r"\d+", n.split("flash")[0])] or [0]
        found = sorted(set(out), key=lambda n: (ver(n), "lite" not in n), reverse=True)[:3]
        return found or FALLBACK_MODELS
    except Exception:
        return FALLBACK_MODELS

def analyze(img, lang_code, status=None):
    img = img.copy(); img.thumbnail((1024, 1024))          # smaller image = much faster
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=85)
    opts = dict(timeout=40_000)                              # give up on a model after 40 s
    if hasattr(types, "HttpRetryOptions"):
        opts["retry_options"] = types.HttpRetryOptions(attempts=1)   # no long retry waits
    client = genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(**opts))
    prompt = PROMPT.replace("{LANG}", AI_LANG[lang_code])
    parts = [types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"), prompt]
    errors = []
    for m in pick_models(GEMINI_API_KEY):
        if status: status.update(label=f"{t['examining']} ({m})")
        for think in ({"thinking_level": "low"}, None):      # try fast mode, then plain
            try:
                cfg = dict(response_mime_type="application/json", temperature=0.2)
                if think: cfg["thinking_config"] = types.ThinkingConfig(**think)
                resp = client.models.generate_content(model=m, contents=parts,
                                                      config=types.GenerateContentConfig(**cfg))
                return json.loads(resp.text), m
            except Exception as e:
                errors.append(f"{m}: {str(e)[:160]}")
    raise RuntimeError("All models failed:\n" + "\n".join(errors))

AREA_TO_ACRE = {"acre": 1.0, "hectare": 2.471, "kanal": 0.125, "marla": 0.00625, "sq. metre": 0.000247}

# ---------------- main ----------------
left, right = st.columns(2, gap="large")

with left:
    st.subheader(t["camera"])
    tab_live, tab_up = st.tabs([t["live"], t["upload"]])
    with tab_live:
        if st.button(t["grab"], use_container_width=True, type="primary"):
            if not stream_url:
                st.warning(t["nourl"])
            else:
                with st.spinner(t["connecting"]):
                    try: st.session_state.img = grab_frame(stream_url)
                    except Exception as e: st.error(str(e))
    with tab_up:
        up = st.file_uploader(t["photo"], type=["jpg", "jpeg", "png"])
        cam = st.camera_input(t["usecam"])
        if cam or up:
            st.session_state.img = Image.open(cam or up).convert("RGB")

    img = st.session_state.get("img")
    if img:
        st.image(img, caption=t["frame"], use_container_width=True)

    st.subheader(t["field"])
    c1, c2 = st.columns([2, 1])
    area_val = c1.number_input(t["area"], min_value=0.0, value=1.0, step=0.5)
    area_unit = c2.selectbox(t["unit"], list(AREA_TO_ACRE))
    acres = area_val * AREA_TO_ACRE[area_unit]
    st.caption(f"= {acres:.3f} acre")
    go = st.button(t["analyze"], use_container_width=True, type="primary", disabled=img is None)

if go:
    if GEMINI_API_KEY.startswith("PASTE_"):
        st.error(t["nokey"])
    else:
        with st.status(t["examining"], expanded=False) as status:
            try:
                st.session_state.result, st.session_state.model = analyze(img, L, status)
                st.session_state.analyzed_at = time.strftime("%H:%M:%S")
                status.update(state="complete")
            except Exception as e:
                status.update(state="error")
                st.error(str(e))

with right:
    st.subheader(t["diag"])
    r = st.session_state.get("result")
    if not r:
        st.info(t["hint"])
    elif not r.get("plant_detected", True):
        st.warning(f"{t['noplant']}: {r.get('disease_or_problem','')}. {t['tryagain']}")
    else:
        sev = r.get("health_status", "healthy")
        cls = {"healthy": "ok", "mild": "warn", "moderate": "warn", "severe": "bad"}.get(sev, "warn")
        st.markdown(f"""<div class="card"><div class="lbl">{t['plant']}</div>
        <div class="big">{r['plant_name']}</div>
        <span class="pill {cls}">{t['health'].get(sev, sev)}</span> &nbsp;
        <span class="lbl">{t['conf']} {r['plant_confidence']:.0f}%</span></div>
        <div class="card"><div class="lbl">{t['problem']}</div>
        <div class="big">{r['disease_or_problem']}</div>
        <span class="lbl">{t['conf']} {r['disease_confidence']:.0f}%</span></div>""", unsafe_allow_html=True)
        if r.get("symptoms_seen"):
            st.markdown(f"**{t['symptoms']}:** " + ", ".join(r["symptoms_seen"]))

        if r["treatment_type"] == "none":
            st.success(t["healthy_msg"])
        else:
            dose = r["dose_per_acre_ml_or_g"] * acres
            water_l = r["water_per_acre_litres"] * acres
            secs = water_l * 1000 / PUMP_ML_PER_SEC
            st.markdown(f"""<div class="card"><div class="lbl">{t['rec']} · {t['types'].get(r['treatment_type'], '')}</div>
            <div class="big">{r['recommended_product']}</div></div>""", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric(t["product"], f"{dose:,.0f} {r['dose_unit']}")
            m2.metric(t["water"], f"{water_l:,.1f} L")
            m3.metric(t["pump"], f"{secs:,.0f} s", f"{secs/60:.1f} {t['min']}")
            st.caption(t["pumpnote"].format(ml=water_l * 1000, rate=PUMP_ML_PER_SEC))
            with st.expander(t["notes"]):
                for n in r.get("application_notes", []): st.write("• " + n)
            if r.get("organic_alternative"):
                st.write(f"{t['organic']}: " + r["organic_alternative"])
        with st.expander(t["safety"]):
            for n in r.get("safety_notes", []): st.write("• " + n)
        st.caption(f"{st.session_state.model} · {st.session_state.analyzed_at}")
        st.warning(t["warn"])