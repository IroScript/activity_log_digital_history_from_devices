import re

with open(r"c:\Users\Irak\Desktop\AI_Agent\DigitalHistory\openrecall\app.py", "r", encoding="utf-8", errors="replace") as f:
    content = f.read()

# Find where the api_critic route starts and ends
start_idx = content.find('@app.route("/api/ai_critic"')
if start_idx == -1:
    start_idx = content.find('@app.route("/api/ai_critic", methods=["GET", "POST"])')

end_idx = content.find('@app.route("/api/search")')

if start_idx != -1 and end_idx != -1:
    before = content[:start_idx]
    after = content[end_idx:]
    
    new_route = """@app.route("/api/ai_critic", methods=["GET", "POST"])
def api_critic():
    import requests
    import base64
    import json
    
    try:
        # Read API key
        try:
            with open(r"C:\\Users\\Irak\\Desktop\\AI_Agent\\DigitalHistory\\groqapi.txt", "r") as f:
                api_key = f.read().strip()
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
                "SELECT timestamp FROM entries WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp DESC LIMIT ?",
                (start_time, end_time, max_frames)
            ).fetchall()
        else:
            results = c.execute(
                "SELECT timestamp FROM entries ORDER BY timestamp DESC LIMIT ?", (max_frames,)
            ).fetchall()
        conn.close()

        if not results:
            return jsonify({
                "status": "success",
                "critic": "<p>কোনো কার্যক্রম পাওয়া যায়নি। প্রথমে কিছু সময় কম্পিউটারে কাজ করুন যাতে ওপেনরিকল আপনার অ্যাক্টিভিটি ক্যাপচার করতে পারে।</p>"
            })

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze the attached screenshots of my workflow. Please provide a critical analysis of my workflow based on what you see, similar to a strict supervisor who asks critical questions about my actions. Format your response exactly like this in Bengali (HTML format string):\\n\\n<p><strong>১. আপনি কীভাবে চিন্তা করেছেন?</strong><br>[Your answer]</p>\\n<p><strong>২. আপনি কীভাবে কাজটা করলেন?</strong><br>[Your answer]</p>\\n<p><strong>৩. আপনি এখানে কী কী করছেন? আপনি কি এআই বা কোড এডিটরের রিপ্লাইয়ের জন্য অপেক্ষা করছেন?</strong><br><ul><li><strong>যা করছেন:</strong> [Your answer]</li><li><strong>অপেক্ষা করছেন কি না:</strong> [Your answer]</li></ul></p>\\n<hr style=\\"border-color: rgba(255, 0, 255, 0.2); margin: 25px 0;\\">\\n<p style=\\"color: var(--accent-fuchsia); font-weight: bold; text-shadow: var(--glow-fuchsia); text-transform: uppercase; letter-spacing: 1px;\\">স্ক্রিনশটটির ওপর ভিত্তি করে ১০টি ব্যক্তিগত সমালোচনামূলক প্রশ্ন ও উত্তর:</p>\\n<p><strong>প্রশ্ন ১: [Question]</strong><br><strong>উত্তর:</strong> [Answer]</p>\\n... up to 10 questions. Ensure everything is in Bengali. Do NOT use markdown codeblocks for HTML, just return raw HTML."
                    }
                ]
            }
        ]

        # Limit to the actual frames found, reversed so chronological order is maintained
        for row in reversed(results):
            ts = row[0]
            img_path = os.path.join(screenshots_path, f"{ts}.webp")
            if os.path.exists(img_path):
                try:
                    with open(img_path, "rb") as image_file:
                        b64 = base64.b64encode(image_file.read()).decode('utf-8')
                        messages[0]["content"].append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/webp;base64,{b64}"
                            }
                        })
                except Exception as e:
                    pass

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


"""
    with open(r"c:\Users\Irak\Desktop\AI_Agent\DigitalHistory\openrecall\app.py", "w", encoding="utf-8") as f:
        f.write(before + new_route + after)
    print("Fixed app.py")
else:
    print("Could not find the bounds to fix app.py")
