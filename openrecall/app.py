import os
import sqlite3
import sys
import threading
import time

import mss
import numpy as np
from doctr.models import ocr_predictor
from flask import Flask, render_template, render_template_string, request, send_from_directory, jsonify
from PIL import Image
from sentence_transformers import SentenceTransformer


def get_appdata_folder(app_name="openrecall"):
    """
    Get the path to the application data folder.

    Args:
        app_name (str): The name of the application.

    Returns:
        str: The path to the application data folder.
    """
    if sys.platform == "win32":
        appdata = os.getenv("APPDATA")
        if not appdata:
            raise EnvironmentError("APPDATA environment variable is not set.")
        path = os.path.join(appdata, app_name)
    elif sys.platform == "darwin":
        home = os.path.expanduser("~")
        path = os.path.join(home, "Library", "Application Support", app_name)
    else:  # Linux and other Unix-like systems
        home = os.path.expanduser("~")
        path = os.path.join(home, ".local", "share", app_name)

    if not os.path.exists(path):
        os.makedirs(path)

    return path


appdata_folder = get_appdata_folder()

print(f"All data is stored in: {appdata_folder}")

db_path = os.path.join(appdata_folder, "recall.db")

screenshots_path = os.path.join(appdata_folder, "screenshots")

# ensure the screenshots folder exists
if not os.path.exists(screenshots_path):
    try:
        os.makedirs(screenshots_path)
    except:
        pass


def get_active_app_name_osx():
    """Returns the name of the active application."""
    from AppKit import NSWorkspace

    active_app = NSWorkspace.sharedWorkspace().activeApplication()
    return active_app["NSApplicationName"]


def get_active_window_title_osx():
    """Returns the title of the active window."""
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGNullWindowID,
        kCGWindowListOptionOnScreenOnly,
    )

    app_name = get_active_app_name_osx()
    windows = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly, kCGNullWindowID
    )

    for window in windows:
        if window["kCGWindowOwnerName"] == app_name:
            return window.get("kCGWindowName", "Unknown")

    return None


def get_active_app_name_windows():
    """returns the app's name .exe"""
    import psutil
    import win32gui

    pid = win32gui.GetForegroundWindow()
    exe = psutil.Process(win32gui.GetWindowThreadProcessId(pid)[-1]).name()
    return exe


def get_active_window_title_windows():
    """Returns the title of the active window."""
    import win32gui

    hwnd = win32gui.GetForegroundWindow()
    window_title = win32gui.GetWindowText(hwnd)
    return window_title


def get_linux_window_info():
    try:
        import os, json
        filepath = "/tmp/openrecall_active_window.json"
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
                return data.get("app", "Linux App"), data.get("title", "Linux Window")
    except Exception:
        pass
    return "Linux App", "Linux Window"


def get_active_app_name_linux():
    app_name, _ = get_linux_window_info()
    return app_name


def get_active_window_title_linux():
    _, title = get_linux_window_info()
    return title


def get_active_app_name():
    if sys.platform == "win32":
        return get_active_app_name_windows()
    elif sys.platform == "darwin":
        return get_active_app_name_osx()
    else:
        return get_active_app_name_linux()


def get_active_window_title():
    if sys.platform == "win32":
        return get_active_window_title_windows()
    elif sys.platform == "darwin":
        return get_active_window_title_osx()
    else:
        return get_active_window_title_linux()


def create_db():
    # create table if not exists for entries, with columns id, text, datetime, and embedding (blob)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS entries
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, app TEXT, title TEXT, text TEXT, timestamp INTEGER, embedding BLOB)"""
    )
    conn.commit()
    conn.close()


_embedding_model = None

def get_embedding(text):
    global _embedding_model
    if _embedding_model is None:
        # Initialize the model once
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Split text into sentences
    sentences = text.split("\n")

    # Get sentence embeddings
    sentence_embeddings = _embedding_model.encode(sentences)

    # Aggregate embeddings (mean pooling in this example)
    mean = np.mean(sentence_embeddings, axis=0)
    # convert to float64
    mean = mean.astype(np.float64)
    return mean


ocr = ocr_predictor(
    pretrained=True,
    det_arch="db_mobilenet_v3_large",
    reco_arch="crnn_mobilenet_v3_large",
)


def take_screenshot(monitor=1):
    """
    Take a screenshot of the specified monitor.

    Args:
        monitor (int): The index of the monitor to capture the screenshot from.

    Returns:
        numpy.ndarray: The screenshot image as a numpy array.
    """
    import sys
    if sys.platform.startswith("linux"):
        import os
        import subprocess
        import re
        import time
        from PIL import Image
        
        is_wayland = "WAYLAND_DISPLAY" in os.environ or os.environ.get("XDG_SESSION_TYPE") == "wayland"
        
        if is_wayland:
            filename = f"openrecall_screen_{int(time.time())}.png"
            try:
                result = subprocess.run(
                    ["gdbus", "call", "--session",
                     "--dest", "org.gnome.Shell.Screenshot",
                     "--object-path", "/org/gnome/Shell/Screenshot",
                     "--method", "org.gnome.Shell.Screenshot.Screenshot",
                     "false", "false", filename],
                    capture_output=True, text=True, timeout=5
                )
                output = result.stdout
                if result.returncode == 0 and "true," in output:
                    match = re.search(r"'(.*?)'", output)
                    if match:
                        file_path = match.group(1)
                        if os.path.exists(file_path):
                            img = Image.open(file_path).convert("RGB")
                            screenshot = np.array(img)
                            try:
                                os.remove(file_path)
                            except:
                                pass
                            return screenshot
                raise Exception(f"GNOME Screenshot DBus API failed (output: {output})")
            except Exception as e:
                raise Exception(f"Wayland DBus screenshot failed: {e}")
        # If it's Linux but not Wayland, we can fall through to mss

    # Fallback for Windows/Mac or X11
    with mss.mss() as sct:
        monitor_ = sct.monitors[monitor]
        screenshot = np.array(sct.grab(monitor_))
        screenshot = screenshot[:, :, [2, 1, 0]]
        return screenshot

def record_screenshot_thread():
    """
    Thread function to continuously record screenshots and process them.
    """
    print("[OpenRecall Thread] Thread started successfully.", flush=True)
    last_screenshot = None
    while last_screenshot is None:
        try:
            last_screenshot = take_screenshot()
            print("[OpenRecall Thread] Initial screenshot captured successfully.", flush=True)
        except Exception as e:
            print(f"[OpenRecall Thread] Error taking initial screenshot: {e}. Retrying in 5 seconds...", flush=True)
            time.sleep(5)

    first_run = True

    while True:
        try:
            screenshot = take_screenshot()
            similarity = mean_structured_similarity_index(screenshot, last_screenshot)
            print(f"[OpenRecall Thread] Captured frame. Similarity with previous: {similarity:.4f}", flush=True)

            if first_run or similarity < 0.95:
                if first_run:
                    print("[OpenRecall Thread] Saving first screenshot immediately...", flush=True)
                    first_run = False
                else:
                    print(f"[OpenRecall Thread] Screen changed (similarity={similarity:.4f} < 0.95). Saving...", flush=True)
                
                last_screenshot = screenshot
                image = Image.fromarray(screenshot)
                timestamp = int(time.time())
                image.save(
                    os.path.join(screenshots_path, f"{timestamp}.webp"),
                    format="webp",
                    lossless=True,
                )
                print(f"[OpenRecall Thread] WebP image saved at: {timestamp}.webp. Running OCR...", flush=True)
                
                result = ocr([screenshot])
                text = ""

                for page in result.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            for word in line.words:
                                text += word.value + " "
                            text += "\n"
                        text += "\n"

                print(f"[OpenRecall Thread] OCR complete. Characters extracted: {len(text)}. Computing embedding...", flush=True)
                embedding = get_embedding(text)
                active_app_name = get_active_app_name()
                active_window_title = get_active_window_title()
                
                print(f"[OpenRecall Thread] Active App: {active_app_name} | Window: {active_window_title}", flush=True)

                # connect to db
                conn = sqlite3.connect(db_path)
                c = conn.cursor()

                # Insert the entry into the database
                embedding_bytes = embedding.tobytes()
                c.execute(
                    "INSERT INTO entries (text, timestamp, embedding, app, title) VALUES (?, ?, ?, ?, ?)",
                    (
                        text,
                        timestamp,
                        embedding_bytes,
                        active_app_name,
                        active_window_title,
                    ),
                )

                # Commit the transaction
                conn.commit()
                conn.close()
                print("[OpenRecall Thread] Saved frame successfully to SQLite database.", flush=True)
        except Exception as e:
            print(f"[OpenRecall Thread] Error in screenshot loop: {e}. Continuing in 3 seconds...", flush=True)

        time.sleep(3)


def mean_structured_similarity_index(img1, img2, L=255):
    """Compute the mean Structural Similarity Index between two images."""
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2

    # Convert images to grayscale
    def rgb2gray(img):
        return 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]

    img1_gray = rgb2gray(img1)
    img2_gray = rgb2gray(img2)

    # Means
    mu1 = np.mean(img1_gray)
    mu2 = np.mean(img2_gray)

    # Variances and covariances
    sigma1_sq = np.var(img1_gray)
    sigma2_sq = np.var(img2_gray)
    sigma12 = np.mean((img1_gray - mu1) * (img2_gray - mu2))

    # SSIM computation
    ssim_index = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_index


def is_similar(img1, img2, similarity_threshold=0.9):
    """Check if two images are similar based on a given similarity threshold."""
    similarity = mean_structured_similarity_index(img1, img2)
    return similarity >= similarity_threshold


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


app = Flask(__name__)


def human_readable_time(timestamp):
    import datetime

    if timestamp is None or str(timestamp) == 'Undefined':
        return "N/A"
    try:
        now = datetime.datetime.now()
        dt_object = datetime.datetime.fromtimestamp(float(timestamp))
        diff = now - dt_object

        if diff.days > 0:
            return f"{diff.days} days ago"
        elif diff.seconds < 60:
            return f"{diff.seconds} seconds ago"
        elif diff.seconds < 3600:
            return f"{diff.seconds // 60} minutes ago"
        else:
            return f"{diff.seconds // 3600} hours ago"
    except Exception:
        return "N/A"


def timestamp_to_human_readable(timestamp):
    import datetime

    if timestamp is None or str(timestamp) == 'Undefined':
        return "N/A"
    try:
        dt_object = datetime.datetime.fromtimestamp(float(timestamp))
        return dt_object.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "N/A"


app.jinja_env.filters["human_readable_time"] = human_readable_time
app.jinja_env.filters["timestamp_to_human_readable"] = timestamp_to_human_readable


def classify_activity(app_name, title, ocr_text):
    """
    Classify the activity based on the application name, window title, and OCR text.
    Returns: (category, action_summary)
    """
    app_lower = (app_name or "").lower()
    title_lower = (title or "").lower()
    ocr_lower = (ocr_text or "").lower()

    # 1. Chat / Communication
    chat_apps = ["slack", "discord", "whatsapp", "teams", "messenger", "telegram", "skype", "zoom"]
    chat_keywords = ["messenger.com", "whatsapp.com", "chat.openai.com", "claude.ai", "chatgpt", "gemini.google.com"]
    if any(x in app_lower for x in chat_apps) or any(x in title_lower for x in chat_keywords):
        summary = "Chatting / Communication"
        if "messenger" in title_lower or "messenger" in app_lower:
            summary = "Messaging on Messenger"
        elif "whatsapp" in title_lower or "whatsapp" in app_lower:
            summary = "Chatting on WhatsApp"
        elif "slack" in title_lower or "slack" in app_lower:
            summary = "Communicating on Slack"
        elif "chatgpt" in title_lower or "chat.openai" in title_lower:
            summary = "Conversing with ChatGPT"
        elif "claude" in title_lower or "claude.ai" in title_lower:
            summary = "Conversing with Claude AI"
        elif "gemini" in title_lower or "gemini.google" in title_lower:
            summary = "Conversing with Gemini AI"
        return "Chatting", summary

    # 2. Debugging / Bug Fixing
    debug_keywords = [
        "traceback (most recent call last):",
        "exception:",
        "syntaxerror:",
        "typeerror:",
        "keyerror:",
        "indexerror:",
        "valueerror:",
        "uncaught exception",
        "fatal error",
        "npm err!",
        "failed to compile",
        "failed with exit code",
        "assertionerror",
        "stack trace",
        "invalid syntax",
        "deprecationwarning"
    ]
    if any(x in ocr_lower for x in debug_keywords) or "devtools" in title_lower or "debugger" in title_lower:
        summary = "Debugging and fixing errors"
        if "." in title:
            parts = title.split()
            for part in parts:
                if "." in part and part.split(".")[-1] in ["py", "js", "html", "css", "json", "go", "rs", "cpp", "c", "sh"]:
                    summary = f"Debugging errors in {part}"
                    break
        return "Debugging", summary

    # 3. Coding / Development
    code_editors = ["code", "pycharm", "cursor", "sublime", "notepad++", "intellij", "eclipse", "neovim", "vim", "emacs", "windsurf"]
    code_extensions = [".py", ".html", ".js", ".css", ".jsx", ".tsx", ".json", ".go", ".rs", ".cpp", ".c", ".h", ".sh", ".yaml", ".yml", ".md"]
    terminal_apps = ["cmd", "powershell", "bash", "zsh", "terminal", "wt.exe", "conhost"]
    
    is_editor = any(x in app_lower for x in code_editors) or any(x in title_lower for x in code_editors)
    is_terminal = any(x in app_lower for x in terminal_apps)
    has_code_file = any(ext in title_lower for ext in code_extensions)
    has_code_keywords = any(kw in ocr_lower for kw in ["def ", "import ", "const ", "let ", "function", "class ", "return ", "public static void", "git commit", "git push", "npm run"])

    if is_editor or (is_terminal and has_code_keywords) or (has_code_file and (is_editor or is_terminal or has_code_keywords)):
        filename = "codebase"
        for part in title.split():
            clean_part = part.strip("● \u25cf*")
            if "." in clean_part and clean_part.split(".")[-1] in ["py", "js", "html", "css", "json", "jsx", "tsx", "go", "rs", "cpp", "c", "sh", "md"]:
                filename = clean_part
                break
        return "Coding", f"Writing code in {filename}"

    # 4. Research / Searching
    search_keywords = ["google search", "google.com/search", "bing.com/search", "duckduckgo.com", "yahoo.com/search"]
    is_search = any(x in title_lower for x in ["google search", "bing search"]) or any(x in ocr_lower for x in search_keywords)
    is_tech_site = any(x in title_lower or x in ocr_lower for x in ["stackoverflow.com", "stack overflow", "github.com", "w3schools", "developer.mozilla", "medium.com"])
    
    if is_search:
        query = "something"
        for sep in ["- Google Search", "- Bing", "- DuckDuckGo"]:
            if sep in title:
                query = title.split(sep)[0].strip()
                break
        return "Researching", f"Searching for: \"{query}\""
    elif is_tech_site:
        if "stackoverflow" in title_lower:
            return "Researching", "Researching on Stack Overflow"
        elif "github" in title_lower:
            repo_name = "GitHub"
            if "/" in title:
                parts = title.split()
                for p in parts:
                    if "/" in p:
                        repo_name = p
                        break
            return "Researching", f"Browsing GitHub repo: {repo_name}"
        else:
            return "Researching", "Researching technical topics"

    # 5. Browsing / Entertainment
    ent_keywords = ["youtube.com", "youtube", "netflix.com", "netflix", "facebook.com", "twitter.com", "x.com", "reddit.com", "reddit", "instagram"]
    if any(x in title_lower for x in ent_keywords) or any(x in app_lower for x in ["spotify", "vlc", "netflix"]):
        summary = "Browsing social media / Entertainment"
        if "youtube" in title_lower:
            video_title = title.replace("- YouTube", "").strip()
            summary = f"Watching YouTube: \"{video_title}\""
        elif "facebook" in title_lower:
            summary = "Browsing Facebook Feed"
        elif "reddit" in title_lower:
            summary = "Reading Reddit threads"
        return "Browsing", summary

    if title:
        short_title = title[:40] + "..." if len(title) > 40 else title
        return "General Work", f"Active in {short_title}"
    
    return "General Work", "Active on desktop"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/timeline")
def api_timeline():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    results = c.execute(
        "SELECT id, app, title, text, timestamp FROM entries ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()
    
    entries = []
    for row in results:
        category, action_summary = classify_activity(row[1], row[2], row[3])
        entries.append({
            "id": row[0],
            "app": row[1],
            "title": row[2],
            "text": row[3],
            "timestamp": row[4],
            "category": category,
            "action_summary": action_summary,
            "image_path": f"/static/{row[4]}.webp"
        })
    return jsonify(entries)


@app.route("/api/stats")
def api_stats():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    results = c.execute(
        "SELECT app, COUNT(*) as cnt FROM entries GROUP BY app ORDER BY cnt DESC"
    ).fetchall()
    conn.close()
    
    if not results:
        return jsonify({"stats": [], "total_screenshots": 0})
        
    stats = []
    total = sum(row[1] for row in results)
    for row in results:
        app_name = row[0] or "Unknown"
        stats.append({
            "app": app_name,
            "count": row[1],
            "percentage": round((row[1] / total) * 100, 2) if total > 0 else 0
        })
    return jsonify({"stats": stats, "total_screenshots": total})


@app.route("/api/delete/<int:timestamp>", methods=["POST", "DELETE"])
def api_delete(timestamp):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("DELETE FROM entries WHERE timestamp = ?", (timestamp,))
    conn.commit()
    conn.close()

    file_path = os.path.join(screenshots_path, f"{timestamp}.webp")
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            return jsonify({"status": "error", "message": f"Failed to delete file: {e}"}), 500
            
    return jsonify({"status": "success", "message": f"Entry for {timestamp} deleted successfully."})


def clean_filename(name):
    import re
    cleaned = re.sub(r'[^\w\s-]', '', name or '')
    cleaned = re.sub(r'\s+', '_', cleaned.strip())
    return cleaned[:50]


@app.route("/api/export_last_5")
def export_last_5():
    try:
        export_dir = os.path.join(appdata_folder, "exported_text")
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        results = c.execute(
            "SELECT timestamp, text, app, title FROM entries ORDER BY timestamp DESC LIMIT 5"
        ).fetchall()
        conn.close()
        
        exported_files = []
        for timestamp, text, app_name, title in results:
            clean_app = clean_filename(app_name)
            clean_title = clean_filename(title)
            name_parts = [str(timestamp)]
            if clean_app:
                name_parts.append(clean_app)
            if clean_title:
                name_parts.append(clean_title)
            
            file_name = "_".join(name_parts) + ".txt"
            file_path = os.path.join(export_dir, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text or "")
            exported_files.append(file_path)
            
        return jsonify({
            "status": "success",
            "message": "Last 5 screenshot texts exported successfully.",
            "export_directory": export_dir,
            "files": exported_files
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
@app.route("/api/ai_critic", methods=["GET", "POST"])
def api_critic():
    import requests
    import base64
    import json
    
    try:
        # Read API key
        try:
            with open(r"C:\Users\Irak\Desktop\AI_Agent\DigitalHistory\groqapi.txt", "r") as f:
                api_key = f.read().strip().replace('"', '').replace(',', '')
        except Exception as e:
            return jsonify({"status": "error", "message": "API key file not found: " + str(e)}), 500

        payload_json = request.get_json() if request.is_json else {}
        start_time = payload_json.get("start_time")
        end_time = payload_json.get("end_time")
        max_frames = payload_json.get("max_frames", 5)

        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        if start_time and end_time:
            results = c.execute(
                "SELECT timestamp, app, title, text FROM entries WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp DESC LIMIT ?",
                (start_time, end_time, max_frames)
            ).fetchall()
        else:
            results = c.execute(
                "SELECT timestamp, app, title, text FROM entries ORDER BY timestamp DESC LIMIT ?", (max_frames,)
            ).fetchall()
        conn.close()

        if not results:
            return jsonify({
                "status": "success",
                "critic": "<p>কোনো কার্যক্রম পাওয়া যায়নি। প্রথমে কিছু সময় কম্পিউটারে কাজ করুন যাতে ওপেনরিকল আপনার অ্যাক্টিভিটি ক্যাপচার করতে পারে।</p>"
            })

        workflow_text = "Here is the chronological OCR text and window titles of my workflow:\n\n"
        for row in reversed(results):
            ts = row[0]
            app_name = row[1] or "Unknown App"
            title = row[2] or "Unknown Window"
            ocr_text = row[3] or ""
            workflow_text += f"--- Time: {ts} ---\nApp: {app_name}\nWindow Title: {title}\nScreen Text/OCR:\n{ocr_text}\n\n"

        prompt_text = (
            "Analyze my workflow based on the following timeline of window titles and on-screen text.\n"
            "Please provide a critical analysis of my workflow based on what you see, similar to a strict supervisor who asks critical questions about my actions. Format your response exactly like this in English (HTML format string):\n\n"
            "<p><strong>1. How did you think about this?</strong><br>[Your answer]</p>\n"
            "<p><strong>2. How did you do this task?</strong><br>[Your answer]</p>\n"
            "<p><strong>3. What exactly are you doing here? Are you waiting for an AI or code editor reply?</strong><br><ul><li><strong>What you are doing:</strong> [Your answer]</li><li><strong>Waiting or not:</strong> [Your answer]</li></ul></p>\n"
            "<hr style=\"border-color: rgba(255, 0, 255, 0.2); margin: 25px 0;\">\n"
            "<p style=\"color: var(--accent-fuchsia); font-weight: bold; text-shadow: var(--glow-fuchsia); text-transform: uppercase; letter-spacing: 1px;\">10 Critical Questions and Answers based on the workflow:</p>\n"
            "<p><strong>Question 1: [Question]</strong><br><strong>Answer:</strong> [Answer]</p>\n"
            "... up to 10 questions. Ensure everything is in English. Do NOT use markdown codeblocks for HTML, just return raw HTML.\n\n"
            f"{workflow_text}"
        )

        messages = [
            {
                "role": "user",
                "content": prompt_text
            }
        ]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "openai/gpt-oss-120b",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048
        }

        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
        if resp.status_code == 200:
            result_json = resp.json()
            critic_text = result_json["choices"][0]["message"]["content"]
            
            # Remove markdown code block if the AI wrapped it
            if critic_text.startswith("```html"):
                critic_text = critic_text[7:]
            if critic_text.endswith("```"):
                critic_text = critic_text[:-3]
                
            return jsonify({
                "status": "success",
                "critic": critic_text.strip()
            })
        else:
            return jsonify({"status": "error", "message": f"Groq API Error: {resp.text}"}), 500

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "")
    if not q:
        return jsonify([])

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    results = c.execute("SELECT id, app, title, text, timestamp, embedding FROM entries").fetchall()
    conn.close()
    
    if not results:
        return jsonify([])

    embeddings = []
    for result in results:
        embeddings.append(np.frombuffer(result[5], dtype=np.float64))

    embeddings = np.array(embeddings)
    query_embedding = get_embedding(q)

    similarities = []
    for embedding in embeddings:
        similarities.append(cosine_similarity(query_embedding, embedding))

    indices = np.argsort(similarities)[::-1]

    entries = []
    for i in indices:
        if similarities[i] < 0.15:
            continue
        result = results[i]
        entries.append(
            {
                "id": result[0],
                "app": result[1],
                "title": result[2],
                "text": result[3],
                "timestamp": result[4],
                "similarity": float(similarities[i]),
                "image_path": f"/static/{result[4]}.webp",
            }
        )

    return jsonify(entries)


import re
import datetime
import difflib

# ─── Regex Patterns for Deep OCR Text Parsing ───
youtube_time_pattern = re.compile(r'\b(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\s*/\s*(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\b')
url_pattern = re.compile(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+')
email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
number_pattern = re.compile(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b')
# Excel/Sheets tab pattern — common OCR artifacts for sheet tabs
sheet_tab_pattern = re.compile(r'(?:Sheet\d+|TASKLIST|EXPENSE|BUDGET|DATA|SUMMARY|INVOICE|SALES|INVENTORY)', re.IGNORECASE)
# Facebook/social patterns
fb_group_pattern = re.compile(r'(?:Group|গ্রুপ|Page|পেজ)\s*[:\-]?\s*([^\n]{3,60})', re.IGNORECASE)
fb_action_pattern = re.compile(r'(?:Like|Liked|Comment|Commented|Share|Shared|Reply|Replied|React|Reacted|Follow|Following)', re.IGNORECASE)
# Code patterns
code_func_pattern = re.compile(r'(?:def|function|class|const|let|var|import|from|public|private|void)\s+(\w+)', re.IGNORECASE)
error_pattern = re.compile(r'(?:Error|Exception|Traceback|Failed|FATAL|WARNING|npm\s*ERR!)[:\s].*', re.IGNORECASE)
git_pattern = re.compile(r'(?:git\s+(?:commit|push|pull|merge|checkout|branch|add|status|diff|log|clone))', re.IGNORECASE)
# Terminal/command patterns
cmd_pattern = re.compile(r'\b(?:python|pip|npm|node|yarn|docker|mkdir|curl|wget)\s+[a-zA-Z0-9_\-\.]+\b', re.IGNORECASE)
# Search query pattern
search_query_pattern = re.compile(r'(?:Search|খুঁজুন|Search for|Searching)\s*[:\-]?\s*([^\n]{3,80})', re.IGNORECASE)
# YouTube channel pattern
yt_channel_pattern = re.compile(r'(?:@\w[\w.-]{2,30})', re.IGNORECASE)
# Like/subscribe/view counts
yt_stats_pattern = re.compile(r'\b(\d[\d,.]*[KMB]?)\s*(?:views|likes|subscribers|comments|dislike)', re.IGNORECASE)
# Chat message patterns
chat_message_pattern = re.compile(r'(?:You|Me|আমি|তুমি|আপনি)\s*[:]\s*(.{5,120})', re.IGNORECASE)


def parse_youtube_time(ocr_text):
    """Extract current playback position and total duration from YouTube player OCR text."""
    if not ocr_text:
        return None
    match = youtube_time_pattern.search(ocr_text)
    if match:
        try:
            h1 = int(match.group(1)) if match.group(1) else 0
            m1 = int(match.group(2))
            s1 = int(match.group(3))
            h2 = int(match.group(4)) if match.group(4) else 0
            m2 = int(match.group(5))
            s2 = int(match.group(6))
            curr = h1 * 3600 + m1 * 60 + s1
            dur = h2 * 3600 + m2 * 60 + s2
            return curr, dur
        except Exception:
            return None
    return None


def format_seconds(secs):
    """Convert seconds to human-readable time string."""
    if secs < 0:
        secs = abs(secs)
    if secs < 60:
        return f"{secs}s"
    mins = secs // 60
    s = secs % 60
    if mins < 60:
        return f"{mins}m {s}s" if s > 0 else f"{mins}m"
    hrs = mins // 60
    mins = mins % 60
    return f"{hrs}h {mins}m"


def extract_frame_details(text, title, app_name):
    """
    Deep OCR text analysis for a single frame.
    Extracts structured details depending on the platform/application detected.
    Returns a dict of all detected context from this single frame.
    """
    ocr = (text or "").strip()
    ocr_lower = ocr.lower()
    title_lower = (title or "").lower()
    app_lower = (app_name or "").lower()
    
    details = {
        "raw_text_length": len(ocr),
        "platform": "unknown",
        "extracted_items": []
    }
    
    # ─── YouTube Frame ───
    if "youtube" in title_lower or "youtube.com" in ocr_lower:
        details["platform"] = "youtube"
        # Video title from window title
        vid_title = (title or "").replace("- YouTube", "").replace("- Google Chrome", "").replace("- Mozilla Firefox", "").replace("- Microsoft Edge", "").strip()
        if vid_title:
            details["video_title"] = vid_title
        # Playback time
        yt_time = parse_youtube_time(ocr)
        if yt_time:
            details["playback_current"] = yt_time[0]
            details["playback_duration"] = yt_time[1]
        # Channel
        channels = yt_channel_pattern.findall(ocr)
        if channels:
            details["channels"] = list(set(channels[:3]))
        # Stats (views, likes)
        stats = yt_stats_pattern.findall(ocr)
        if stats:
            details["visible_stats"] = stats[:5]
        # Comments visible
        comment_keywords = ["comment", "reply", "replies", "pinned"]
        if any(kw in ocr_lower for kw in comment_keywords):
            details["comment_section_visible"] = True
        # Like state
        if "liked" in ocr_lower:
            details["like_state"] = "liked"
        elif "like" in ocr_lower:
            details["like_state"] = "not_liked"
        # Subscribe state
        if "subscribed" in ocr_lower:
            details["subscribe_state"] = "subscribed"
        elif "subscribe" in ocr_lower:
            details["subscribe_state"] = "not_subscribed"
    
    # ─── Excel / Spreadsheet Frame ───
    elif any(x in app_lower for x in ["excel", "calc", "sheets", "wps"]) or \
         any(x in title_lower for x in ["excel", ".xlsx", ".xls", ".csv", "sheets", "spreadsheet"]):
        details["platform"] = "spreadsheet"
        # Sheet tab names
        tabs = sheet_tab_pattern.findall(ocr)
        if tabs:
            details["visible_sheet_tabs"] = list(set(tabs))
        # Try to detect active sheet from title or prominent OCR
        if "tasklist" in ocr_lower or "task list" in ocr_lower:
            details["active_context"] = "TaskList"
        elif "expense" in ocr_lower:
            details["active_context"] = "Expense"
        elif "budget" in ocr_lower:
            details["active_context"] = "Budget"
        elif "invoice" in ocr_lower:
            details["active_context"] = "Invoice"
        # Numbers visible (potential data entries)
        numbers = number_pattern.findall(ocr)
        if numbers:
            # Filter out very small numbers and keep meaningful ones
            meaningful_nums = [n for n in numbers if len(n) > 1 and n not in ("00", "01", "10")]
            details["visible_numbers"] = meaningful_nums[:15]
        # Cell references
        cell_refs = re.findall(r'\b[A-Z]{1,3}\d{1,5}\b', ocr)
        if cell_refs:
            details["cell_references"] = list(set(cell_refs[:10]))
        # Formula detection
        formulas = re.findall(r'=\s*(?:SUM|AVERAGE|COUNT|IF|VLOOKUP|INDEX|MATCH|MAX|MIN|CONCATENATE)\s*\(', ocr, re.IGNORECASE)
        if formulas:
            details["formulas_visible"] = formulas[:5]
    
    # ─── Facebook Frame ───
    elif "facebook" in title_lower or "facebook.com" in ocr_lower or "fb.com" in ocr_lower:
        details["platform"] = "facebook"
        # Group/Page name
        groups = fb_group_pattern.findall(ocr)
        if groups:
            details["group_or_page"] = groups[0].strip()
        # Actions visible
        actions = fb_action_pattern.findall(ocr)
        if actions:
            details["visible_actions"] = list(set(a.lower() for a in actions))
        # Post content snippets - look for longer text blocks
        lines = []
        ui_words = ["my drive", "search", "google", "settings", "notifications", "messenger", "friends", "watch", "marketplace"]
        for l in ocr.split('\n'):
            l = l.strip()
            if len(l) > 30 and not any(uw in l.lower() for uw in ui_words):
                lines.append(l)
        if lines:
            details["post_snippets"] = lines[:3]
        # User names (look for profile-like patterns)
        names = re.findall(r'(?:^|\n)([A-Z][a-z]+ [A-Z][a-z]+)', ocr)
        if names:
            details["visible_user_names"] = list(set(names[:5]))
    
    # ─── Code Editor Frame ───
    elif any(x in app_lower for x in ["code", "pycharm", "cursor", "sublime", "intellij", "vim", "neovim", "windsurf"]) or \
         any(title_lower.endswith(ext) or (ext + " ") in title_lower for ext in [".py", ".js", ".html", ".css", ".tsx", ".jsx", ".go", ".rs", ".cpp", ".c", ".java"]):
        details["platform"] = "code_editor"
        # Active filename from title
        filename = None
        for part in (title or "").split():
            clean = part.strip("● ✦*[]")
            if "." in clean and clean.split(".")[-1] in ["py", "js", "html", "css", "json", "jsx", "tsx", "go", "rs", "cpp", "c", "sh", "md", "yaml", "yml", "java"]:
                filename = clean
                break
        if filename:
            details["active_file"] = filename
        # Functions/classes visible
        funcs = code_func_pattern.findall(ocr)
        if funcs:
            details["visible_symbols"] = list(set(funcs[:10]))
        # Imports
        imports = re.findall(r'(?:import|from|require|include)\s+(\S+)', ocr)
        if imports:
            details["imports"] = list(set(imports[:8]))
        # Errors in the code
        errors = error_pattern.findall(ocr)
        if errors:
            details["visible_errors"] = [e.strip()[:100] for e in errors[:3]]
        # Git commands
        git_cmds = git_pattern.findall(ocr)
        if git_cmds:
            details["git_operations"] = list(set(git_cmds))
        # Terminal commands
        cmds = cmd_pattern.findall(ocr)
        if cmds:
            details["terminal_commands"] = list(set(cmds[:5]))
        # Line numbers (to estimate which part of the file user is looking at)
        line_nums = re.findall(r'^\s*(\d{1,5})\s', ocr, re.MULTILINE)
        if line_nums:
            try:
                nums = [int(n) for n in line_nums[:20]]
                if nums:
                    details["line_range"] = f"lines {min(nums)}-{max(nums)}"
            except Exception:
                pass
    
    # ─── Chat / Messaging Frame ───
    elif any(x in app_lower or x in title_lower for x in ["slack", "discord", "whatsapp", "teams", "messenger", "telegram", "chatgpt", "claude", "gemini"]):
        details["platform"] = "chat"
        # Platform name
        for p in ["slack", "discord", "whatsapp", "teams", "messenger", "telegram", "chatgpt", "claude", "gemini"]:
            if p in app_lower or p in title_lower:
                details["chat_platform"] = p.capitalize()
                break
        # Chat partner / channel from title
        chat_context = (title or "").split(" - ")[0].strip()
        if chat_context and len(chat_context) > 1:
            details["chat_context"] = chat_context
        # Messages visible
        messages = chat_message_pattern.findall(ocr)
        if messages:
            details["visible_messages"] = [m.strip()[:100] for m in messages[:5]]
        # Message count estimate
        msg_lines = [l for l in ocr.split('\n') if len(l.strip()) > 15]
        details["visible_message_count"] = len(msg_lines)
    
    # ─── Browser / Search Frame ───
    elif any(x in app_lower for x in ["chrome", "firefox", "edge", "brave", "safari", "opera"]):
        details["platform"] = "browser"
        # URLs visible
        urls = url_pattern.findall(ocr)
        if urls:
            details["visible_urls"] = urls[:3]
        # Search queries
        queries = search_query_pattern.findall(ocr)
        if queries:
            details["search_queries"] = [q.strip() for q in queries[:3]]
        # Check if it's a Google search
        if "google" in title_lower and ("search" in title_lower or "google.com/search" in ocr_lower):
            query = (title or "").split(" - Google")[0].strip()
            if query:
                details["search_query"] = query
        # Page title
        page_title = (title or "").strip()
        if page_title:
            # Clean common browser suffixes
            for suffix in ["- Google Chrome", "- Mozilla Firefox", "- Microsoft Edge", "- Brave", "- Opera"]:
                page_title = page_title.replace(suffix, "").strip()
            details["page_title"] = page_title
    
    # ─── Terminal Frame ───
    elif any(x in app_lower for x in ["cmd", "powershell", "bash", "terminal", "wt.exe", "conhost", "iterm", "alacritty"]):
        details["platform"] = "terminal"
        cmds = cmd_pattern.findall(ocr)
        if cmds:
            details["commands"] = list(set(cmds[:8]))
        git_cmds = git_pattern.findall(ocr)
        if git_cmds:
            details["git_operations"] = list(set(git_cmds))
        errors = error_pattern.findall(ocr)
        if errors:
            details["errors"] = [e.strip()[:100] for e in errors[:3]]
    
    # ─── Generic / Unknown ───
    else:
        details["platform"] = "general"
        # Just capture the key visible text
        lines = [l.strip() for l in ocr.split('\n') if len(l.strip()) > 20]
        if lines:
            details["visible_text_lines"] = lines[:5]
        urls = url_pattern.findall(ocr)
        if urls:
            details["visible_urls"] = urls[:3]
            
    # ─── Generic Data Record Extraction ───
    # Find lines that look like structured data/tasks (a mix of letters and numbers/codes)
    data_records = []
    ui_boilerplate = ["my drive", "search", "inbox", "outbox", "starred", "file", "edit", "view", "insert", "format", "tools", "extensions", "help", "share", "google", "yahoo", "chrome", "firefox", "window", "open", "save", "print", "settings", "recent", "trash", "storage"]
    for line in ocr.split('\n'):
        line = line.strip()
        if len(line) < 10 or len(line) > 120: continue
        lower_line = line.lower()
        if any(ui in lower_line for ui in ui_boilerplate):
            continue
        # Heuristic for a record: contains letters and numbers, or specific uppercase codes
        if (re.search(r'\d', line) and re.search(r'[a-zA-Z]{4,}', line)) or re.search(r'\b[A-Z]{2,4}\d{1,4}\b', line):
            data_records.append(line)
    
    if data_records:
        details["data_records"] = list(set(data_records))

    return details


def compute_text_diff_summary(prev_text, curr_text):
    """
    Compare OCR text between two consecutive frames.
    Returns a list of meaningful changes detected.
    """
    if not prev_text or not curr_text:
        return []
    
    prev_lines = set(l.strip() for l in prev_text.split('\n') if len(l.strip()) > 5)
    curr_lines = set(l.strip() for l in curr_text.split('\n') if len(l.strip()) > 5)
    
    added = curr_lines - prev_lines
    removed = prev_lines - curr_lines
    
    changes = []
    
    # Filter for meaningful additions (not UI chrome)
    ui_noise = {"search", "file", "edit", "view", "help", "window", "tools", "home", "back", "forward", "reload", "my drive", "inbox", "starred", "recent"}
    meaningful_added = []
    for l in added:
        l_lower = l.lower()
        if len(l) > 15 and not any(n in l_lower for n in ui_noise):
            meaningful_added.append(l)
            
    meaningful_removed = []
    for l in removed:
        l_lower = l.lower()
        if len(l) > 15 and not any(n in l_lower for n in ui_noise):
            meaningful_removed.append(l)
    
    if meaningful_added:
        changes.append({"type": "text_appeared", "items": list(meaningful_added)[:5]})
    if meaningful_removed:
        changes.append({"type": "text_disappeared", "items": list(meaningful_removed)[:5]})
    
    return changes


def generate_session_detail_points(session_frames):
    """
    Generate 10+ detailed action points from a list of frames belonging to one session.
    Each frame is a tuple: (app_name, title, text, timestamp, embedding_bytes)
    
    Returns a list of detail point dicts with 'action', 'time', 'icon'.
    """
    if not session_frames:
        return []
    
    points = []
    
    # ─── 1. Session Start Point ───
    first = session_frames[0]
    app_name, title, text, timestamp, _ = first
    first_details = extract_frame_details(text, title, app_name)
    dt = datetime.datetime.fromtimestamp(timestamp)
    time_str = dt.strftime("%I:%M:%S %p")
    
    # Platform-specific session start
    platform = first_details.get("platform", "general")
    
    if platform == "youtube":
        vid_title = first_details.get("video_title", title or "a video")
        channels = first_details.get("channels", [])
        channel_str = f" on channel {channels[0]}" if channels else ""
        points.append({"action": f"Opened YouTube video \"{vid_title}\"{channel_str}", "time": time_str, "icon": "▶️"})
        if first_details.get("playback_current") is not None:
            pts = format_seconds(first_details["playback_current"])
            dur = format_seconds(first_details.get("playback_duration", 0))
            points.append({"action": f"Video playback started at {pts} / {dur}", "time": time_str, "icon": "⏱️"})
    elif platform == "spreadsheet":
        ctx = first_details.get("active_context", "")
        tabs = first_details.get("visible_sheet_tabs", [])
        tabs_str = f" (visible tabs: {', '.join(tabs)})" if tabs else ""
        workbook = (title or "").replace(".xlsx", "").replace(".xls", "").replace(" - Excel", "").strip()
        if ctx:
            points.append({"action": f"Opened workbook \"{workbook}\" on {ctx} sheet{tabs_str}", "time": time_str, "icon": "📊"})
        else:
            points.append({"action": f"Working in spreadsheet \"{workbook}\"{tabs_str}", "time": time_str, "icon": "📊"})
    elif platform == "code_editor":
        filename = first_details.get("active_file", "codebase")
        symbols = first_details.get("visible_symbols", [])
        line_range = first_details.get("line_range", "")
        sym_str = f" — visible functions: {', '.join(symbols[:4])}" if symbols else ""
        lr_str = f" ({line_range})" if line_range else ""
        points.append({"action": f"Opened \"{filename}\" in code editor{lr_str}{sym_str}", "time": time_str, "icon": "💻"})
    elif platform == "facebook":
        group = first_details.get("group_or_page", "")
        grp_str = f" in \"{group}\"" if group else ""
        points.append({"action": f"Browsing Facebook{grp_str}", "time": time_str, "icon": "👤"})
    elif platform == "chat":
        chat_plat = first_details.get("chat_platform", "Chat")
        chat_ctx = first_details.get("chat_context", "")
        ctx_str = f" with/in \"{chat_ctx}\"" if chat_ctx else ""
        points.append({"action": f"Started conversation on {chat_plat}{ctx_str}", "time": time_str, "icon": "💬"})
    elif platform == "browser":
        page_title = first_details.get("page_title", title or "a page")
        search_q = first_details.get("search_query", "")
        if search_q:
            points.append({"action": f"Searching Google for \"{search_q}\"", "time": time_str, "icon": "🔍"})
        else:
            points.append({"action": f"Browsing \"{page_title}\"", "time": time_str, "icon": "🌐"})
    elif platform == "terminal":
        cmds = first_details.get("commands", [])
        if cmds:
            points.append({"action": f"Working in terminal — running: {cmds[0]}", "time": time_str, "icon": "⌨️"})
        else:
            points.append({"action": "Opened terminal / command prompt", "time": time_str, "icon": "⌨️"})
    else:
        short = (title or app_name or "application")[:60]
        points.append({"action": f"Started working in \"{short}\"", "time": time_str, "icon": "📌"})
    
    # ─── 2. Frame-by-Frame Transition Analysis ───
    prev_details = first_details
    prev_frame = first
    youtube_watch_start = first_details.get("playback_current") if platform == "youtube" else None
    youtube_watch_segments = []
    last_emitted_action = ""
    
    for i in range(1, len(session_frames)):
        frame = session_frames[i]
        f_app, f_title, f_text, f_ts, f_emb = frame
        curr_details = extract_frame_details(f_text, f_title, f_app)
        f_dt = datetime.datetime.fromtimestamp(f_ts)
        f_time = f_dt.strftime("%I:%M:%S %p")
        
        prev_app, prev_title, prev_text, prev_ts, prev_emb = prev_frame
        
        # Compute embedding drift
        drift = 0.0
        if f_emb and prev_emb:
            try:
                vec_curr = np.frombuffer(f_emb, dtype=np.float64)
                vec_prev = np.frombuffer(prev_emb, dtype=np.float64)
                sim = cosine_similarity(vec_curr, vec_prev)
                drift = 1.0 - sim
            except Exception:
                pass
        
        # Compute text diff
        text_changes = compute_text_diff_summary(prev_text, f_text)
        
        # ─── YouTube micro-actions ───
        if curr_details.get("platform") == "youtube" and prev_details.get("platform") == "youtube":
            curr_yt = curr_details.get("playback_current")
            prev_yt = prev_details.get("playback_current")
            curr_dur = curr_details.get("playback_duration")
            prev_dur = prev_details.get("playback_duration")
            
            # Rewind detection
            if curr_yt is not None and prev_yt is not None and curr_dur == prev_dur:
                time_elapsed = f_ts - prev_ts
                if curr_yt < prev_yt:
                    diff = prev_yt - curr_yt
                    action = f"Rewound video by {format_seconds(diff)} (from {format_seconds(prev_yt)} back to {format_seconds(curr_yt)})"
                    if action != last_emitted_action:
                        points.append({"action": action, "time": f_time, "icon": "⏪"})
                        last_emitted_action = action
                elif curr_yt > prev_yt + time_elapsed + 5:
                    diff = curr_yt - prev_yt
                    action = f"Skipped forward by {format_seconds(diff)} (from {format_seconds(prev_yt)} to {format_seconds(curr_yt)})"
                    if action != last_emitted_action:
                        points.append({"action": action, "time": f_time, "icon": "⏩"})
                        last_emitted_action = action
            
            # Next video detection
            if f_title != prev_title and drift > 0.35:
                new_vid = (f_title or "").replace("- YouTube", "").replace("- Google Chrome", "").strip()
                channels = curr_details.get("channels", [])
                ch_str = f" (channel: {channels[0]})" if channels else ""
                action = f"Switched to next video: \"{new_vid}\"{ch_str}"
                if action != last_emitted_action:
                    points.append({"action": action, "time": f_time, "icon": "⏭️"})
                    last_emitted_action = action
            
            # Comment section interaction
            prev_like = prev_details.get("like_state", "")
            curr_like = curr_details.get("like_state", "")
            if curr_like == "liked" and prev_like != "liked":
                points.append({"action": "Liked a comment or the video", "time": f_time, "icon": "👍"})
            
            if curr_details.get("comment_section_visible") and not prev_details.get("comment_section_visible"):
                points.append({"action": "Scrolled down to the comments section", "time": f_time, "icon": "💬"})
            
            # Subscribe state change
            prev_sub = prev_details.get("subscribe_state", "")
            curr_sub = curr_details.get("subscribe_state", "")
            if curr_sub == "subscribed" and prev_sub != "subscribed":
                points.append({"action": "Subscribed to the channel", "time": f_time, "icon": "🔔"})
        
        # ─── Spreadsheet micro-actions ───
        elif curr_details.get("platform") == "spreadsheet":
            prev_ctx = prev_details.get("active_context", "")
            curr_ctx = curr_details.get("active_context", "")
            if curr_ctx and prev_ctx and curr_ctx != prev_ctx:
                action = f"Switched from \"{prev_ctx}\" sheet to \"{curr_ctx}\" sheet"
                if action != last_emitted_action:
                    points.append({"action": action, "time": f_time, "icon": "📋"})
                    last_emitted_action = action
            
            # Detect new data records appearing (instead of isolated numbers)
            prev_recs = set(prev_details.get("data_records", []))
            curr_recs = set(curr_details.get("data_records", []))
            new_recs = curr_recs - prev_recs
            if new_recs:
                valid_new = [nr for nr in new_recs if not any(difflib.SequenceMatcher(None, nr, pr).ratio() > 0.8 for pr in prev_recs)]
                for rec in valid_new[:2]:
                    action = f"Working on record/task: \"{rec[:60]}\""
                    if action != last_emitted_action:
                        points.append({"action": action, "time": f_time, "icon": "📝"})
                        last_emitted_action = action
            
            # Formula usage
            prev_formulas = prev_details.get("formulas_visible", [])
            curr_formulas = curr_details.get("formulas_visible", [])
            new_formulas = [f for f in curr_formulas if f not in prev_formulas]
            if new_formulas:
                action = f"Using formula: {new_formulas[0]}"
                if action != last_emitted_action:
                    points.append({"action": action, "time": f_time, "icon": "🔢"})
                    last_emitted_action = action
        
        # ─── Code Editor micro-actions ───
        elif curr_details.get("platform") == "code_editor":
            prev_file = prev_details.get("active_file", "")
            curr_file = curr_details.get("active_file", "")
            if curr_file and prev_file and curr_file != prev_file:
                action = f"Switched from \"{prev_file}\" to \"{curr_file}\""
                if action != last_emitted_action:
                    points.append({"action": action, "time": f_time, "icon": "📄"})
                    last_emitted_action = action
            
            # New functions/classes appearing
            prev_syms = set(prev_details.get("visible_symbols", []))
            curr_syms = set(curr_details.get("visible_symbols", []))
            new_syms = curr_syms - prev_syms
            if new_syms and len(new_syms) <= 5:
                sym_str = ", ".join(list(new_syms)[:3])
                action = f"Screen content includes code constructs: {sym_str}"
                if action != last_emitted_action:
                    points.append({"action": action, "time": f_time, "icon": "⚙️"})
                    last_emitted_action = action
            
            # Error appeared
            prev_errs = prev_details.get("visible_errors", [])
            curr_errs = curr_details.get("visible_errors", [])
            new_errs = [e for e in curr_errs if e not in prev_errs]
            if new_errs:
                action = f"Error encountered: {new_errs[0][:80]}"
                if action != last_emitted_action:
                    points.append({"action": action, "time": f_time, "icon": "🔴"})
                    last_emitted_action = action
            elif prev_errs and not curr_errs:
                action = "Errors resolved — code appears clean"
                if action != last_emitted_action:
                    points.append({"action": action, "time": f_time, "icon": "✅"})
                    last_emitted_action = action
            
            # Git operations
            git_ops = curr_details.get("git_operations", [])
            prev_git = prev_details.get("git_operations", [])
            new_git = [g for g in git_ops if g not in prev_git]
            if new_git:
                action = f"Git operation: {new_git[0]}"
                if action != last_emitted_action:
                    points.append({"action": action, "time": f_time, "icon": "🔀"})
                    last_emitted_action = action
            
            # Terminal commands
            cmds = curr_details.get("terminal_commands", [])
            prev_cmds = prev_details.get("terminal_commands", [])
            new_cmds = [c for c in cmds if c not in prev_cmds]
            if new_cmds:
                action = f"Ran command: {new_cmds[0]}"
                if action != last_emitted_action:
                    points.append({"action": action, "time": f_time, "icon": "⌨️"})
                    last_emitted_action = action
            
            # Line range change (scrolling through code)
            prev_lr = prev_details.get("line_range", "")
            curr_lr = curr_details.get("line_range", "")
            if curr_lr and prev_lr and curr_lr != prev_lr:
                action = f"Scrolled to {curr_lr} in {curr_file or 'code'}"
                if action != last_emitted_action:
                    points.append({"action": action, "time": f_time, "icon": "📜"})
                    last_emitted_action = action
        
        # ─── Facebook micro-actions ───
        elif curr_details.get("platform") == "facebook":
            prev_group = prev_details.get("group_or_page", "")
            curr_group = curr_details.get("group_or_page", "")
            if curr_group and prev_group and curr_group != prev_group:
                action = f"Navigated to \"{curr_group}\""
                if action != last_emitted_action:
                    points.append({"action": action, "time": f_time, "icon": "📱"})
                    last_emitted_action = action
            
            # New actions (like, comment, share)
            prev_actions = set(prev_details.get("visible_actions", []))
            curr_actions = set(curr_details.get("visible_actions", []))
            new_actions = curr_actions - prev_actions
            for act in new_actions:
                grp_str = f" in \"{curr_group}\"" if curr_group else ""
                action = f"{act.capitalize()} on a post{grp_str}"
                if action != last_emitted_action:
                    points.append({"action": action, "time": f_time, "icon": "👍"})
                    last_emitted_action = action
            
            # Post content changes
            prev_snippets = prev_details.get("post_snippets", [])
            curr_snippets = curr_details.get("post_snippets", [])
            new_snippets = [s for s in curr_snippets if s not in prev_snippets]
            if new_snippets and drift > 0.15:
                snippet = new_snippets[0][:80]
                action = f"Scrolled to new content: \"{snippet}...\""
                if action != last_emitted_action:
                    points.append({"action": action, "time": f_time, "icon": "📰"})
                    last_emitted_action = action
        
        # ─── Chat micro-actions ───
        elif curr_details.get("platform") == "chat":
            prev_ctx = prev_details.get("chat_context", "")
            curr_ctx = curr_details.get("chat_context", "")
            if curr_ctx and prev_ctx and curr_ctx != prev_ctx:
                plat = curr_details.get("chat_platform", "chat")
                action = f"Switched conversation on {plat} to \"{curr_ctx}\""
                if action != last_emitted_action:
                    points.append({"action": action, "time": f_time, "icon": "🔄"})
                    last_emitted_action = action
            
            # New messages
            prev_msg_count = prev_details.get("visible_message_count", 0)
            curr_msg_count = curr_details.get("visible_message_count", 0)
            if curr_msg_count > prev_msg_count + 2:
                diff_count = curr_msg_count - prev_msg_count
                action = f"~{diff_count} new messages appeared in conversation"
                if action != last_emitted_action:
                    points.append({"action": action, "time": f_time, "icon": "✉️"})
                    last_emitted_action = action
            
            # Visible message content
            msgs = curr_details.get("visible_messages", [])
            prev_msgs = prev_details.get("visible_messages", [])
            new_msgs = [m for m in msgs if m not in prev_msgs]
            if new_msgs:
                action = f"Message exchanged: \"{new_msgs[0][:70]}...\""
                if action != last_emitted_action:
                    points.append({"action": action, "time": f_time, "icon": "💬"})
                    last_emitted_action = action
        
        # ─── Browser/Search micro-actions ───
        elif curr_details.get("platform") == "browser":
            prev_page = prev_details.get("page_title", "")
            curr_page = curr_details.get("page_title", "")
            if curr_page and prev_page and curr_page != prev_page:
                action = f"Navigated to \"{curr_page[:60]}\""
                if action != last_emitted_action:
                    points.append({"action": action, "time": f_time, "icon": "🌐"})
                    last_emitted_action = action
            
            curr_search = curr_details.get("search_query", "")
            prev_search = prev_details.get("search_query", "")
            if curr_search and curr_search != prev_search:
                action = f"Searched for \"{curr_search}\""
                if action != last_emitted_action:
                    points.append({"action": action, "time": f_time, "icon": "🔍"})
                    last_emitted_action = action
        
        # ─── Platform switch within session ───
        if curr_details.get("platform") != prev_details.get("platform") and i > 0:
            prev_plat = prev_details.get("platform", "unknown")
            curr_plat = curr_details.get("platform", "unknown")
            action = f"Context switched from {prev_plat} to {curr_plat}"
            if action != last_emitted_action:
                points.append({"action": action, "time": f_time, "icon": "🔄"})
                last_emitted_action = action
        
        # ─── High embedding drift (significant content change) — generic fallback ───
        if drift > 0.20 and len(points) < 25:
            # If no platform-specific point was generated, add a generic one from text diff
            point_count_before = len(points)
            if text_changes:
                for change in text_changes:
                    if change["type"] == "text_appeared" and change["items"]:
                        item = change["items"][0][:80]
                        action = f"Screen content changed — new text visible: \"{item}\""
                        if action != last_emitted_action and len(points) < 25:
                            points.append({"action": action, "time": f_time, "icon": "📋"})
                            last_emitted_action = action
                            break
        
        prev_details = curr_details
        prev_frame = frame
    
    # ─── 3. Session End Point ───
    last = session_frames[-1]
    l_app, l_title, l_text, l_ts, _ = last
    last_details = extract_frame_details(l_text, l_title, l_app)
    l_dt = datetime.datetime.fromtimestamp(l_ts)
    l_time = l_dt.strftime("%I:%M:%S %p")
    
    # YouTube: add final playback position
    if last_details.get("platform") == "youtube" and last_details.get("playback_current") is not None:
        pts = format_seconds(last_details["playback_current"])
        dur = format_seconds(last_details.get("playback_duration", 0))
        points.append({"action": f"Last seen playback position: {pts} / {dur}", "time": l_time, "icon": "⏸️"})
    
    # Session duration summary
    total_dur = l_ts - session_frames[0][3]
    if total_dur > 0:
        start_time_str = datetime.datetime.fromtimestamp(session_frames[0][3]).strftime("%I:%M:%S %p")
        points.append({"action": f"Session lasted from {start_time_str} to {l_time} (Duration: {format_seconds(total_dur)})", "time": l_time, "icon": "⏱️"})
    
    # ─── 4. Ensure minimum 10 points — fill with frame-level observations ───
    if len(points) < 10 and len(session_frames) > 1:
        # Add observations from text changes between sampled frames
        sample_indices = list(range(0, len(session_frames), max(1, len(session_frames) // (12 - len(points)))))
        for idx in sample_indices:
            if len(points) >= 10:
                break
            if idx == 0:
                continue
            frame = session_frames[idx]
            f_app, f_title, f_text, f_ts, _ = frame
            f_dt = datetime.datetime.fromtimestamp(f_ts)
            f_time_str = f_dt.strftime("%I:%M:%S %p")
            
            details = extract_frame_details(f_text, f_title, f_app)
            
            # Add platform-specific observations
            if details.get("platform") == "youtube":
                if details.get("visible_stats"):
                    points.append({"action": f"Video stats visible: {', '.join(details['visible_stats'][:3])}", "time": f_time_str, "icon": "📊"})
            elif details.get("platform") == "spreadsheet":
                if details.get("data_records"):
                    rec = details["data_records"][0][:70]
                    points.append({"action": f"Current context: \"{rec}...\"", "time": f_time_str, "icon": "📊"})
            elif details.get("platform") == "code_editor":
                if details.get("imports"):
                    imp_str = ", ".join(details["imports"][:4])
                    points.append({"action": f"Imports/dependencies used: {imp_str}", "time": f_time_str, "icon": "📦"})
            elif details.get("platform") == "browser":
                urls = details.get("visible_urls", [])
                if urls:
                    points.append({"action": f"Page URL: {urls[0][:70]}", "time": f_time_str, "icon": "🔗"})
            elif details.get("platform") == "chat":
                msg_count = details.get("visible_message_count", 0)
                if msg_count > 0:
                    points.append({"action": f"~{msg_count} messages visible in chat window", "time": f_time_str, "icon": "✉️"})
            else:
                text_lines = details.get("visible_text_lines", [])
                if text_lines:
                    points.append({"action": f"Screen shows: \"{text_lines[0][:70]}\"", "time": f_time_str, "icon": "📋"})
    
    # If still under 10 points for very short sessions, add observations about the overall session
    if len(points) < 10:
        # Add window title info
        cleaned_titles = []
        for f in session_frames:
            if f[1]:
                t = f[1].replace("- Google Chrome", "").replace("- Mozilla Firefox", "").strip()
                if len(t) > 55: t = t[:52] + "..."
                if t and not any(t in ex or ex in t for ex in cleaned_titles):
                    cleaned_titles.append(t)
        if len(cleaned_titles) > 1:
            points.append({"action": f"Window titles during session: {', '.join(cleaned_titles[:3])}", "time": time_str, "icon": "🪟"})
        
        # Add app info
        unique_apps = list(set((f[0] or "unknown") for f in session_frames if f[0] and f[0].lower() not in ["chrome.exe", "msedge.exe", "explorer.exe", "unknown"]))
        if unique_apps:
            points.append({"action": f"Specific applications detected: {', '.join(unique_apps[:4])}", "time": time_str, "icon": "📱"})
        
        # Add text volume info
        total_chars = sum(len(f[2] or "") for f in session_frames)
        if total_chars > 0:
            points.append({"action": f"Total OCR text captured: ~{total_chars:,} characters across {len(session_frames)} frames", "time": time_str, "icon": "📝"})
    
    # Deduplicate points by action text
    seen_actions = set()
    unique_points = []
    for p in points:
        if p["action"] not in seen_actions:
            seen_actions.add(p["action"])
            unique_points.append(p)
    
    return unique_points


@app.route("/api/activity_sessions")
def api_activity_sessions():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Fetch all entries sorted by timestamp ascending for chronological processing
    results = c.execute(
        "SELECT app, title, text, timestamp, embedding FROM entries ORDER BY timestamp ASC"
    ).fetchall()
    conn.close()
    
    if not results:
        return jsonify([])
        
    # ─── Step 1: Classify each frame and detect transitions ───
    activities = []
    
    for i in range(len(results)):
        row = results[i]
        app_name, title, text, timestamp, embedding_bytes = row
        category, summary = classify_activity(app_name, title, text)
        
        if i > 0:
            prev_row = results[i-1]
            prev_app, prev_title, prev_text, prev_timestamp, prev_embedding_bytes = prev_row
            
            drift = 0.0
            if embedding_bytes and prev_embedding_bytes:
                try:
                    vec_curr = np.frombuffer(embedding_bytes, dtype=np.float64)
                    vec_prev = np.frombuffer(prev_embedding_bytes, dtype=np.float64)
                    sim = cosine_similarity(vec_curr, vec_prev)
                    drift = 1.0 - sim
                except Exception:
                    pass
            
            is_youtube_curr = "youtube" in (title or "").lower() or "youtube" in (app_name or "").lower()
            is_youtube_prev = "youtube" in (prev_title or "").lower() or "youtube" in (prev_app or "").lower()
            
            if is_youtube_curr and is_youtube_prev:
                curr_yt = parse_youtube_time(text)
                prev_yt = parse_youtube_time(prev_text)
                
                if curr_yt and prev_yt:
                    curr_time, curr_dur = curr_yt
                    prev_time, prev_dur = prev_yt
                    
                    if curr_dur == prev_dur:
                        if curr_time < prev_time:
                            diff = prev_time - curr_time
                            category = "YouTube Rewind"
                            summary = f"Rewound YouTube video by {diff} seconds"
                        elif curr_time > prev_time + (timestamp - prev_timestamp) + 5:
                            diff = curr_time - prev_time
                            category = "YouTube Skip"
                            summary = f"Skipped forward in YouTube video by {diff} seconds"
                
                if (title != prev_title) and (drift > 0.35):
                    video_title = title.replace("- YouTube", "").replace("- Google Chrome", "").strip()
                    category = "YouTube Next Video"
                    summary = f"Switched to next video: \"{video_title}\""
                elif title == prev_title:
                    count_prev = (prev_text or "").lower().count("liked")
                    count_curr = (text or "").lower().count("liked")
                    if count_curr > count_prev:
                        category = "YouTube Interaction"
                        summary = "Liked a comment or video on YouTube"
            
            prev_category, _ = classify_activity(prev_app, prev_title, prev_text)
            if prev_category == "Coding" and category == "Debugging":
                summary = "Encountered a compile or runtime error while coding"
            elif prev_category == "Debugging" and category == "Coding":
                summary = "Resolved code errors and resumed active development"
                
        activities.append({
            "category": category,
            "summary": summary,
            "timestamp": timestamp,
            "frame_index": i
        })
    
    # ─── Step 2: Group consecutive activities into sessions ───
    sessions = []
    current_session = None
    
    for act in activities:
        cat = act["category"]
        sum_text = act["summary"]
        ts = act["timestamp"]
        fi = act["frame_index"]
        
        if current_session and current_session["category"] == cat and (ts - current_session["last_timestamp"]) <= 300:
            current_session["end_timestamp"] = ts
            current_session["last_timestamp"] = ts
            current_session["duration_seconds"] = current_session["end_timestamp"] - current_session["start_timestamp"]
            current_session["frame_indices"].append(fi)
            if len(sum_text) > len(current_session["summary"]):
                current_session["summary"] = sum_text
        else:
            if current_session:
                sessions.append(current_session)
            current_session = {
                "category": cat,
                "summary": sum_text,
                "start_timestamp": ts,
                "end_timestamp": ts,
                "last_timestamp": ts,
                "duration_seconds": 0,
                "frame_indices": [fi]
            }
            
    if current_session:
        sessions.append(current_session)
    
    # ─── Step 3: Generate detail points for each session ───
    formatted_sessions = []
    
    for s in reversed(sessions):
        dur_mins = max(1, round(s["duration_seconds"] / 60))
        duration_str = f"{dur_mins} min" if dur_mins == 1 else f"{dur_mins} mins"
        
        dt_start = datetime.datetime.fromtimestamp(s["start_timestamp"])
        dt_end = datetime.datetime.fromtimestamp(s["end_timestamp"])
        
        # Collect the actual frames for this session
        session_frames = [results[idx] for idx in s["frame_indices"]]
        
        # Generate deep detail points
        detail_points = generate_session_detail_points(session_frames)
        
        formatted_sessions.append({
            "category": s["category"],
            "summary": s["summary"],
            "start_time": dt_start.strftime("%I:%M %p"),
            "end_time": dt_end.strftime("%I:%M %p"),
            "date": dt_start.strftime("%Y-%m-%d"),
            "duration": duration_str,
            "duration_seconds": s["duration_seconds"],
            "start_timestamp": s["start_timestamp"],
            "end_timestamp": s["end_timestamp"],
            "frame_count": len(session_frames),
            "detail_points": detail_points
        })
        
    return jsonify(formatted_sessions)


@app.route("/static/<filename>")
def serve_image(filename):
    return send_from_directory(screenshots_path, filename)


if __name__ == "__main__":
    create_db()

    # Start the thread to record screenshots
    t = threading.Thread(target=record_screenshot_thread)
    t.start()

    app.run(port=8082)
