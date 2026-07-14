import os
import json
import math
import time
import subprocess
import tempfile
import traceback
from flask import Flask, render_template, request, send_file, redirect, url_for
from werkzeug.utils import secure_filename

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------- 字幕處理核心工具函式 ----------

def get_prompt(target_language):
    if target_language == "original":
        lang_instruction = "請保留音訊原本的語言，忠實呈現原始內容，不需要翻譯。"
    else:
        lang_instruction = f"請將每一句話翻譯成自然、口語化、適合觀眾閱讀的「{target_language}」。"

    return f"""\
你是專業的影片字幕師。請處理這段音訊，請完成以下工作：
1. 辨識音訊中所有的語音內容，依照自然的語意/停頓切成適合當作字幕的短句。
2. {lang_instruction}
3. 格式限制：每一則字幕只能有「一行」文字，最多 14 個字（依據該語言習慣調整），不要加任何標點符號。
4. 提供每一句字幕在音訊中精確的起始與結束時間（單位：秒，可為小數）。
5. 如果音訊中有一整段沒有語音內容，可以跳過不輸出。

請只回傳 JSON，格式如下：
{{
  "segments": [
    {{"start_seconds": 0.0, "end_seconds": 2.5, "text": "字幕文字內容"}}
  ]
}}
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

PUNCTUATION_CHARS = "。！？，、；：,.!?;:「」『』【】()（）《》〈〉—…～~"

def strip_punctuation(text):
    return "".join(ch for ch in text if ch not in PUNCTUATION_CHARS)

def split_long_segment(start, end, text, max_chars=14):
    text = text.strip()
    if len(text) <= max_chars:
        return [(start, end, text)]
    split_pos = max_chars
    left_text = text[:split_pos].strip()
    right_text = text[split_pos:].strip()
    if not right_text:
        return [(start, end, left_text)]
    total_len = len(left_text) + len(right_text)
    duration = end - start
    mid = start + duration * (len(left_text) / total_len if total_len else 0.5)
    return split_long_segment(start, mid, left_text, max_chars) + split_long_segment(mid, end, right_text, max_chars)

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

def build_txt(all_segments, output_path):
    lines = []
    for start, end, text in all_segments:
        text = (text or "").strip()
        if text:
            timestamp = f"[{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}]"
            lines.append(f"{timestamp} {text}")
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

# ★ 重點修改：將切割時間改為 120 秒 (2分鐘) ★
def split_audio_if_needed(audio_path, out_dir, chunk_seconds=120):
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

def transcribe_and_translate(client, model_name, audio_path, prompt):
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
        contents=[uploaded_file, prompt],
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

@app.route("/process", methods=["GET", "POST"])
def process_video():
    if request.method == "GET":
        return redirect(url_for("index"))

    video_path = None
    try:
        if genai is None:
            return "伺服器未安裝 google-genai 套件", 500

        api_key = request.form.get("api_key")
        video_file = request.files.get("video_file")
        target_language = request.form.get("target_language", "繁體中文")
        output_format = request.form.get("output_format", "srt")
        model_name = "gemini-3.5-flash"
        
        if not video_file or video_file.filename == '':
            return "請上傳影片檔案", 400
        if not api_key:
            return "請提供 API Key", 400

        # ★ 重點修改：安全檔名處理 ★
        # 過濾掉危險字元，如果檔名是純中文被過濾到變成空的，就給它一個預設時間戳記檔名
        safe_filename = secure_filename(video_file.filename)
        if not safe_filename:
            safe_filename = f"audio_{int(time.time())}.mp3"

        video_path = os.path.join(UPLOAD_FOLDER, safe_filename)
        video_file.save(video_path)
        
        # 決定輸出檔名與路徑 (固定使用英文避免下載時的編碼錯誤)
        out_filename = f"transcript_result.{output_format}"
        out_path = os.path.join(OUTPUT_FOLDER, out_filename)

        # 開始呼叫 Gemini
        client = genai.Client(api_key=api_key.strip())
        prompt = get_prompt(target_language)

        with tempfile.TemporaryDirectory() as tmp_dir:
            print("擷取音訊中...")
            audio_path = extract_audio(video_path, tmp_dir)
            
            # 這裡會觸發 120 秒切割邏輯
            chunks = split_audio_if_needed(audio_path, tmp_dir)
            
            all_segments = []
            for chunk_path, offset in chunks:
                segs = transcribe_and_translate(client, model_name, chunk_path, prompt)
                for seg in segs:
                    start = float(seg.get("start_seconds", 0)) + offset
                    end = float(seg.get("end_seconds", 0)) + offset
                    text = str(seg.get("text", "")).strip()
                    all_segments.append((start, end, text))

            if not all_segments:
                return "沒有辨識到任何語音內容", 400

            all_segments.sort(key=lambda s: s[0])
            
            processed_segments = []
            for start, end, text in all_segments:
                clean_text = strip_punctuation(text)
                processed_segments.extend(split_long_segment(start, end, clean_text))

            # 根據選擇產生不同格式檔案
            if output_format == "txt":
                build_txt(processed_segments, out_path)
            else:
                build_srt(processed_segments, out_path)
        
        # 將處理完成的檔案回傳
        return send_file(out_path, as_attachment=True, download_name=out_filename)

    # ★ 重點修改：擴大錯誤捕捉範圍 ★
    except Exception as e:
        traceback.print_exc()
        return f"處理發生錯誤：{str(e)}", 500
        
    finally:
        # 確保無論成功或失敗，原始上傳的龐大影片檔都會被刪除，不佔用雲端空間
        if video_path and os.path.exists(video_path):
            os.remove(video_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)