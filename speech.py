# web/views/speech.py
import os, subprocess, tempfile, json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from django.views.decorators.http import require_POST
from django.conf import settings

USE_WHISPER = True  # یا False برای Vosk

@require_POST
@ensure_csrf_cookie
def speech_to_text(request):
    if "audio" not in request.FILES:
        return JsonResponse({"ok": False, "error": "no file"}, status=400)

    up = request.FILES["audio"]   # webm/opus
    with tempfile.TemporaryDirectory() as td:
        in_path  = os.path.join(td, "in.webm")
        out_path = os.path.join(td, "out.wav")
        with open(in_path, "wb") as f:
            for chunk in up.chunks():
                f.write(chunk)

        # تبدیل به WAV تک‌کاناله 16kHz
        # ffmpeg -i in.webm -ac 1 -ar 16000 out.wav
        try:
            subprocess.check_call([
                "ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", "16000", out_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            return JsonResponse({"ok": False, "error": f"ffmpeg failed: {e}"}, status=500)

        text = ""
        try:
            if USE_WHISPER:
                import whisper
                model_name = getattr(settings, "WHISPER_MODEL", "base")
                model = whisper.load_model(model_name)
                # می‌توانی language="fa" هم بدهی اگر زبان مشخص است
                res = model.transcribe(out_path, language="fa")
                text = (res.get("text") or "").strip()
            else:
                from vosk import Model, KaldiRecognizer, SetLogLevel
                SetLogLevel(-1)
                # مسیر مدل فارسی را دانلود و تنظیم کن (مثلا vosk-model-small-fa-0.4)
                model_dir = getattr(settings, "VOSK_MODEL_DIR", "/opt/vosk-model-small-fa")
                if not os.path.isdir(model_dir):
                    return JsonResponse({"ok": False, "error": "Vosk model not found"}, status=500)

                import wave, json as pyjson
                wf = wave.open(out_path, "rb")
                rec = KaldiRecognizer(Model(model_dir), wf.getframerate())
                rec.SetWords(True)
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0: break
                    rec.AcceptWaveform(data)
                final = pyjson.loads(rec.FinalResult())
                text = (final.get("text") or "").strip()
        except Exception as e:
            return JsonResponse({"ok": False, "error": f"stt failed: {e}"}, status=500)

    return JsonResponse({"ok": True, "text": text or ""})
