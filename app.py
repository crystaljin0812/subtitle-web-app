import os
import re
import json
import math
import time
import uuid
import threading
import subprocess
import tempfile
import traceback

from flask import Flask, render_template, request, send_file, jsonify, abort
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

# ---------- 檔案大小限制 ----------
# 現在瀏覽器端會先用 ffmpeg.wasm 把影片壓縮成小體積音訊再上傳，
# 正常情況下傳到這裡的檔案應該都只有幾 MB～幾十 MB。
# 這個上限主要是「備援保護」：當使用者瀏覽器不支援/壓縮失敗、退回直接上傳原始影片時，
# 避免真的誇張大的檔案把 Render 免費方案 512MB 記憶體塞爆。
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({
        "error": "檔案太大了（上限 1GB）。這通常代表瀏覽器端壓縮沒有成功、退回上傳原始影片，建議換個瀏覽器或先自行壓縮影片後再試一次。"
    }), 413


# ---------- 工作狀態儲存 ----------
# 用記憶體內的字典追蹤每個處理工作的進度。
# 注意：這個做法只在 gunicorn 用單一 worker（沒有加 --workers 參數）時才會正確運作，
# 因為多個 worker 是不同的行程，彼此記憶體不共享，狀態會對不起來。
# 若之後想開多個 worker 處理更高流量，需要改成 Redis 之類的共用儲存。
JOBS = {}
JOBS_LOCK = threading.Lock()


def set_job(job_id, **kwargs):
    with JOBS_LOCK:
        JOBS[job_id].update(kwargs)


def get_job(job_id):
    with JOBS_LOCK:
        return dict(JOBS.get(job_id, {}))


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
    if seconds < 0:
        seconds = 0
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
        if not text:
            continue
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
    try:
        return float(json.loads(result.stdout)["format"]["duration"]), True
    except Exception:
        pass

    probe_cmd2 = ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration", "-of", "json", audio_path]
    result2 = subprocess.run(probe_cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        return float(json.loads(result2.stdout)["streams"][0]["duration"]), True
    except Exception:
        # 兩種方式都讀不到長度：回傳 False 讓呼叫端知道這是「猜測值」，而非真實時長
        return 0.0, False


def split_audio_if_needed(audio_path, out_dir, chunk_seconds=600, job_id=None):
    """
    預設每段 10 分鐘（比原本的 2 分鐘更省來回次數，比舊版的 30 分鐘更能提供有意義的進度更新）。
    若偵測不到真實時長，會用「保守估計」的長分鐘數切割，並在 job 狀態留下警告，
    避免長影片被默默截斷而使用者毫無所知。
    """
    duration, duration_known = get_audio_duration(audio_path)

    if not duration_known:
        # 猜不到真實長度時，寧可高估也不要低估，避免內容被默默截斷。
        # 這裡假設最長 4 小時；真的超過會在下面的迴圈自然繼續切下去到抓不出新內容為止。
        duration = 4 * 3600.0
        if job_id:
            set_job(job_id, warning="偵測不到音訊實際長度，已採用保守估計值處理，請確認產出字幕是否涵蓋整支影片。")

    if duration <= chunk_seconds:
        return [(audio_path, 0.0)]

    chunks = []
    n_chunks = math.ceil(duration / chunk_seconds)
    for i in range(n_chunks):
        start = i * chunk_seconds
        chunk_path = os.path.join(out_dir, f"chunk_{i}.mp3")
        cmd = ["ffmpeg", "-y", "-i", audio_path, "-ss", str(start), "-t", str(chunk_seconds),
               "-ac", "1", "-ar", "16000", "-b:a", "64k", chunk_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # 如果切出來的片段幾乎沒有內容（代表已經超過音訊實際長度），就不用再往下切了
        if os.path.getsize(chunk_path) < 2000:
            break
        chunks.append((chunk_path, float(start)))
    return chunks


def transcribe_and_translate(client, model_name, audio_path, prompt):
    uploaded_file = client.files.upload(file=audio_path)
    while uploaded_file.state.name == "PROCESSING":
        time.sleep(2)
        uploaded_file = client.files.get(name=uploaded_file.name)
    if uploaded_file.state.name == "FAILED":
        raise RuntimeError("音訊上傳失敗")

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


# ---------- 背景處理主流程 ----------

def run_job(job_id, video_path, api_key, target_language, output_format, original_stem):
    try:
        set_job(job_id, status="processing", progress=2, message="正在從影片擷取音訊...")

        model_name = "gemini-3.5-flash"
        client = genai.Client(api_key=api_key.strip())
        prompt = get_prompt(target_language)

        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = extract_audio(video_path, tmp_dir)
            set_job(job_id, progress=8, message="正在分析音訊長度...")

            chunks = split_audio_if_needed(audio_path, tmp_dir, job_id=job_id)
            total_chunks = len(chunks)

            all_segments = []
            for i, (chunk_path, offset) in enumerate(chunks):
                pct = 10 + int((i / total_chunks) * 80)
                set_job(job_id, progress=pct,
                        message=f"辨識與翻譯中... ({i + 1}/{total_chunks} 段)")
                segs = transcribe_and_translate(client, model_name, chunk_path, prompt)
                for seg in segs:
                    start = float(seg.get("start_seconds", 0)) + offset
                    end = float(seg.get("end_seconds", 0)) + offset
                    text = str(seg.get("text", "")).strip()
                    all_segments.append((start, end, text))

            if not all_segments:
                set_job(job_id, status="error", message="沒有辨識到任何語音內容，請確認影片是否有聲音。")
                return

            set_job(job_id, progress=92, message="正在整理字幕格式...")
            all_segments.sort(key=lambda s: s[0])

            processed_segments = []
            for start, end, text in all_segments:
                clean_text = strip_punctuation(text)
                processed_segments.extend(split_long_segment(start, end, clean_text))

            out_filename = f"{job_id}.{output_format}"
            out_path = os.path.join(OUTPUT_FOLDER, out_filename)

            if output_format == "txt":
                build_txt(processed_segments, out_path)
            else:
                build_srt(processed_segments, out_path)

            set_job(job_id, status="done", progress=100, message="完成！",
                    output_path=out_path, download_name=f"{original_stem}.{output_format}")

    except Exception as e:
        traceback.print_exc()
        set_job(job_id, status="error", message=f"處理發生錯誤：{str(e)}")
    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)


# ---------- Flask 路由 ----------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_video():
    if genai is None:
        return jsonify({"error": "伺服器未安裝 google-genai 套件"}), 500

    api_key = request.form.get("api_key")
    video_file = request.files.get("video_file")
    target_language = request.form.get("target_language", "繁體中文")
    output_format = request.form.get("output_format", "srt")
    # 瀏覽器端如果有先壓縮成音訊，上傳的檔名會變成 compressed_audio.mp3，
    # 所以另外用這個欄位保留「使用者原本選的影片檔名」，下載時才能對應回去。
    original_filename = request.form.get("original_filename") or (video_file.filename if video_file else None)

    if not video_file or video_file.filename == '':
        return jsonify({"error": "請上傳影片檔案"}), 400
    if not api_key:
        return jsonify({"error": "請提供 API Key"}), 400
    if output_format not in ("srt", "txt"):
        output_format = "srt"

    job_id = uuid.uuid4().hex

    safe_filename = secure_filename(video_file.filename) or f"upload_{job_id}"
    # 用 job_id 當前綴，避免多人同時使用時檔名互相覆蓋
    video_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{safe_filename}")
    video_file.save(video_path)

    safe_original = secure_filename(original_filename) if original_filename else ""
    original_stem = os.path.splitext(safe_original)[0] if safe_original else ""
    if not original_stem:
        original_stem = "subtitles"

    with JOBS_LOCK:
        JOBS[job_id] = {"status": "queued", "progress": 0, "message": "已加入處理佇列..."}

    thread = threading.Thread(
        target=run_job,
        args=(job_id, video_path, api_key, target_language, output_format, original_stem),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>", methods=["GET"])
def job_status(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "找不到這個工作編號"}), 404
    # 不把完整檔案路徑洩漏給前端，只回傳必要資訊
    safe = {k: v for k, v in job.items() if k != "output_path"}
    return jsonify(safe)


@app.route("/download/<job_id>", methods=["GET"])
def download_result(job_id):
    job = get_job(job_id)
    if not job or job.get("status") != "done":
        abort(404)
    return send_file(job["output_path"], as_attachment=True,
                      download_name=job.get("download_name", "subtitles.srt"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
