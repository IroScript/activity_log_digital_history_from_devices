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


def get_active_app_name():
    if sys.platform == "win32":
        return get_active_app_name_windows()
    elif sys.platform == "darwin":
        return get_active_app_name_osx()
    else:
        raise NotImplementedError("This platform is not supported")


def get_active_window_title():
    if sys.platform == "win32":
        return get_active_window_title_windows()
    elif sys.platform == "darwin":
        return get_active_window_title_osx()
    else:
        raise NotImplementedError("This platform is not supported")


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
    with mss.mss() as sct:
        monitor_ = sct.monitors[monitor]
        screenshot = np.array(sct.grab(monitor_))
        screenshot = screenshot[:, :, [2, 1, 0]]
        return screenshot

def record_screenshot_thread():
    """
    Thread function to continuously record screenshots and process them.

    This function takes screenshots at regular intervals and compares them with the previous screenshot.
    If the new screenshot is different enough from the previous one, it saves the screenshot, performs OCR on it,
    extracts the text, computes the embedding, and stores the entry in the database.

    Returns:
        None
    """
    last_screenshot = None
    while last_screenshot is None:
        try:
            last_screenshot = take_screenshot()
        except Exception as e:
            print(f"Error taking initial screenshot: {e}. Retrying in 5 seconds...")
            time.sleep(5)

    while True:
        try:
            screenshot = take_screenshot()

            if not is_similar(screenshot, last_screenshot):
                last_screenshot = screenshot
                image = Image.fromarray(screenshot)
                timestamp = int(time.time())
                image.save(
                    os.path.join(screenshots_path, f"{timestamp}.webp"),
                    format="webp",
                    lossless=True,
                )
                result = ocr([screenshot])
                text = ""

                for page in result.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            for word in line.words:
                                text += word.value + " "
                            text += "\n"
                        text += "\n"

                embedding = get_embedding(text)
                active_app_name = get_active_app_name()
                active_window_title = get_active_window_title()

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
        except Exception as e:
            print(f"Error in screenshot recording thread: {e}. Continuing in 3 seconds...")

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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/timeline")
def api_timeline():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    results = c.execute(
        "SELECT id, app, title, text, timestamp FROM entries ORDER BY timestamp DESC LIMIT 1000"
    ).fetchall()
    conn.close()
    
    entries = []
    for row in results:
        entries.append({
            "id": row[0],
            "app": row[1],
            "title": row[2],
            "text": row[3],
            "timestamp": row[4],
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


@app.route("/static/<filename>")
def serve_image(filename):
    return send_from_directory(screenshots_path, filename)


if __name__ == "__main__":
    create_db()

    # Start the thread to record screenshots
    t = threading.Thread(target=record_screenshot_thread)
    t.start()

    app.run(port=8082)
