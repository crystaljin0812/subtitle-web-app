import os
import json
import math
import time
import subprocess
import tempfile
from flask import Flask, render_template, request, send_file

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

app = Flask(__name__)

# 建立暫存資料夾
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------- 字幕處理核心工具函式 ----------

SUBTITLE_PROMPT = """\
你是專業的影片字幕師。請處理這段音訊，音訊內容是韓文或英文（可能混雜），請完成以下工作：
1. 辨識音訊中所有的語音內容，依照自然的語意/停頓切成適合當作字幕的短句。
2. 將每一句話翻譯成自然、口語化、適合觀眾閱讀的「繁體中文」字幕。
3. 格式限制：每一則字幕只能有「一行」文字，最多 14 個中文字，不要加任何標點符號。如果超過 14 個字請拆開。
4. 提供每一句字幕在音訊中精確的起始與結束時間（單位：秒，可為小數）。
5. 如果音訊中有一整段沒有語音內容，可以跳過不輸出。

請只回傳 JSON，格式如下：
{
  "segments": [
    {"start_seconds": 0.0, "end_seconds": 2.5, "text": "繁體中文字幕內容"}
  ]
}
"""

SEGMENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                    "text": {"type": "string"},
                },
                "required": ["start_seconds", "end_seconds", "text"],
            },
        }
    },
    "required": ["segments"],
}

MAX_CHARS_PER_LINE = 14
PUNCTUATION_CHARS = "。！？，、；：,.!?;:「」『』【】()（）《》〈〉—…～~"

def strip_punctuation(text):
    return "".join(ch for ch in text if ch not in PUNCTUATION_CHARS)

def split_long_segment(start, end, text):
    text = text.strip()
    if len(text) <= MAX_CHARS_PER_LINE:
        return [(start, end, text)]
    split_pos = MAX_CHARS_PER_LINE
    left_text = text[:split_pos].strip()
    right_text = text[split_pos:].strip()
    if not right_text:
        return [(start, end, left_text)]
    total_len = len(left_text) + len(right_text)
    duration = end - start
    mid = start + duration * (len(left_text) / total_len if total_len else 0.5)
    return split_long_segment(start, mid, left_text) + split_long_segment(mid, end, right_text)

def format_srt_timestamp(seconds):
    if seconds < 0: seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def build_srt(all_segments, output_path):
    lines = []
    idx = 0
    for start, end, text in all_segments:
        text = (text or "").strip()
        if not text: continue
        idx += 1
        lines.append(str(idx))
        lines.append(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}")
        lines.append(text)
        lines.append("")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def extract_audio(video_path, out_dir):
    audio_path = os.path.join(out_dir, "extracted_audio.mp3")
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", audio_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return audio_path

def get_audio_duration(audio_path):
    probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", audio_path]
    result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return float(json.loads(result.stdout)["format"]["duration"])

def split_audio_if_needed(audio_path, out_dir, chunk_seconds=1800):
    duration = get_audio_duration(audio_path)
    if duration <= chunk_seconds:
        return [(audio_path, 0.0)]
    chunks = []
    n_chunks = math.ceil(duration / chunk_seconds)
    for i in range(n_chunks):
        start = i * chunk_seconds
        chunk_path = os.path.join(out_dir, f"chunk_{i}.mp3")
        cmd = ["ffmpeg", "-y", "-i", audio_path, "-ss", str(start), "-t", str(chunk_seconds), "-ac", "1", "-ar", "16000", "-b:a", "64k", chunk_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        chunks.append((chunk_path, float(start)))
    return chunks

def transcribe_and_translate(client, model_name, audio_path):
    print(f"上傳音訊至 Gemini: {os.path.basename(audio_path)} ...")
    uploaded_file = client.files.upload(file=audio_path)
    while uploaded_file.state.name == "PROCESSING":
        time.sleep(2)
        uploaded_file = client.files.get(name=uploaded_file.name)
    if uploaded_file.state.name == "FAILED":
        raise RuntimeError("音訊上傳失敗")
    
    print("開始辨識與翻譯...")
    response = client.models.generate_content(
        model=model_name,
        contents=[uploaded_file, SUBTITLE_PROMPT],
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=SEGMENTS_SCHEMA),
    )
    segments = json.loads(response.text).get("segments", [])
    try:
        client.files.delete(name=uploaded_file.name)
    except Exception:
        pass
    return segments

# ---------- Flask 路由 ----------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_video():
    if genai is None:
        return "伺服器未安裝 google-genai 套件", 500

    api_key = request.form.get("api_key")
    video_file = request.files.get("video_file")
    model_name = "gemini-3.5-flash" # 預設使用這個模型
    
    if not video_file or video_file.filename == '':
        return "請上傳影片檔案", 400
    if not api_key:
        return "請提供 API Key", 400

    # 1. 儲存上傳的影片
    video_filename = video_file.filename
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    video_file.save(video_path)
    
    # 輸出 SRT 檔名設定
    base_name = os.path.splitext(video_filename)[0]
    srt_filename = f"{base_name}.srt"
    srt_path = os.path.join(OUTPUT_FOLDER, srt_filename)

    try:
        # 2. 初始化 Gemini Client
        client = genai.Client(api_key=api_key.strip())

        # 3. 在暫存資料夾內處理影音
        with tempfile.TemporaryDirectory() as tmp_dir:
            print("擷取音訊中...")
            audio_path = extract_audio(video_path, tmp_dir)
            chunks = split_audio_if_needed(audio_path, tmp_dir)
            
            all_segments = []
            for chunk_path, offset in chunks:
                segs = transcribe_and_translate(client, model_name, chunk_path)
                for seg in segs:
                    start = float(seg.get("start_seconds", 0)) + offset
                    end = float(seg.get("end_seconds", 0)) + offset
                    text = str(seg.get("text", "")).strip()
                    all_segments.append((start, end, text))

            if not all_segments:
                return "沒有辨識到任何語音內容", 400

            all_segments.sort(key=lambda s: s[0])
            
            # 清理標點與長度限制
            processed_segments = []
            for start, end, text in all_segments:
                clean_text = strip_punctuation(text)
                processed_segments.extend(split_long_segment(start, end, clean_text))

            # 4. 建立 SRT 檔案
            build_srt(processed_segments, srt_path)
            
    except Exception as e:
        return f"處理發生錯誤：{str(e)}", 500
    finally:
        # 處理完畢後刪除伺服器上的原始影片節省空間
        if os.path.exists(video_path):
            os.remove(video_path)

    # 5. 將生成的 SRT 檔案回傳給使用者下載
    return send_file(srt_path, as_attachment=True, download_name=srt_filename)

if __name__ == "__main__":
    app.run(debug=True, port=5000)