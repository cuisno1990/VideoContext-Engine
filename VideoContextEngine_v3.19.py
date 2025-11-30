#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VideoContext Engine v3.19
-------------------------
Author: dolphin-creator (https://github.com/dolphin-creator)
Project: VideoContext Engine
License: MIT

Description:
Local-first microservice for video understanding (Scene Detection + Whisper + Qwen3-VL).

-------------------------
INSTALLATION DES DÉPENDANCES (copier/coller) :

# macOS (Apple Silicon / Intel) – CPU (ou Metal) :
#   python -m pip install --upgrade pip
#   pip install fastapi uvicorn[standard] opencv-python yt-dlp pillow numpy openai-whisper huggingface_hub
#   pip install mlx-vlm torchvision

# Windows / Linux – CPU (ou GPU si vous avez déjà PyTorch CUDA) :
#   python -m pip install --upgrade pip
#   pip install fastapi uvicorn[standard] opencv-python yt-dlp pillow numpy openai-whisper huggingface_hub
#   pip install llama-cpp-python

(⚠️ PyTorch est requis par Whisper. Si besoin : pip install torch)

-------------------------
- Prompts techniques (structure JSON, etc.) figés dans le code.
- L'utilisateur ne modifie que des "user prompts" qui s'ajoutent par-dessus.
- 1 seul appel VLM par scène (description + tags en JSON).
- Nombre de keyframes par scène paramétrable (1 à 5), 1 par défaut.
- Audio_features activés par défaut, désactivables.
- Résumé global activé par défaut, désactivable.
- Chronos détaillés (Whisper + VLM + total).
- max_tokens VLM réglables avec valeurs par défaut (220 / 260).
- safe_json_parse robuste (réparation JSON) pour scènes + résumé global.
- Nettoyage du texte brut quand le JSON échoue (on extrait la valeur de "description" / "summary").
- Mode RAM :
    - RAM_MODE = "ram+" (par env VIDEOCONTEXT_RAM_MODE) :
        * précharge VLM par défaut + Whisper small
        * garde tout en RAM
    - RAM_MODE = "ram-":
        * charge/décharge Whisper et VLM à chaque requête
- Swagger patché pour afficher de grands textarea pour les prompts utilisateur.
"""

import os
import gc
import cv2
import time
import shutil
import base64
import asyncio
import platform
import numpy as np
import yt_dlp
import json
import re

from typing import Optional, Dict, Any, List, Union, Literal
from io import BytesIO
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import PlainTextResponse
from fastapi.openapi.utils import get_openapi
from PIL import Image

import whisper

# Optionnel : supprimer le warning HF tokenizers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --- MODE RAM ---

RAM_MODE = os.getenv("VIDEOCONTEXT_RAM_MODE", "ram-").lower()  # "ram+" ou "ram-"
WHISPER_CACHE: Dict[str, Any] = {}

# --- CONFIGURATION GLOBALE ---

PORT_SERVEUR = 7555

DEFAULT_VLM_MODEL_MLX = "mlx-community/Qwen3-VL-2B-Instruct-4bit"
DEFAULT_VLM_REPO_GGUF = "bartowski/Qwen3-VL-2B-Instruct-GGUF"
DEFAULT_VLM_MODEL_GGUF = "qwen3-vl-2b-instruct-q4_k_m.gguf"
DEFAULT_MMPROJ_GGUF = "mmproj-model-f16.gguf"
DEFAULT_WHISPER_MODEL = "small"

DEFAULT_SCENE_THRESHOLD = 0.35
DEFAULT_MIN_DURATION = 2.0
DEFAULT_MAX_DURATION = 60.0
DEFAULT_RESOLUTION = 768
MAX_TOTAL_VIDEO_DURATION = 4 * 60 * 60  # 4h max

DEFAULT_KEYFRAMES_PER_SCENE = 1

DEFAULT_VLM_MAX_TOKENS_SCENE = 220
DEFAULT_VLM_MAX_TOKENS_SUMMARY = 260

if platform.system() == "Darwin":
    ACTIVE_DEFAULT_VLM = DEFAULT_VLM_MODEL_MLX
else:
    ACTIVE_DEFAULT_VLM = DEFAULT_VLM_MODEL_GGUF

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

gpu_lock = asyncio.Lock()

# --- PROMPTS SYSTÈME ---

BASE_VISUAL_PROMPT = """
You are an assistant that analyzes one or several frames from the SAME video scene.

You MUST answer with VALID JSON using EXACTLY this structure:
{
  "description": "short and factual description of what happens in the scene (gestures, posture, general situation)",
  "tags": {
    "people_count": <approximate number of visible people>,
    "place_type": "studio | tv_set | classroom | office | home | outdoor | nature | stage | other",
    "main_action": "short description of the main action",
    "emotional_tone": "calm | neutral | tense | conflictual | joyful | sad | enthusiastic | serious | other",
    "movement_level": "low | medium | high"
  }
}

Rules:
- Do NOT add any text outside of this JSON.
- Do NOT change field names.
- The description and tag values should be in the same language as the user instructions if any, otherwise in French.
""".strip()

BASE_SUMMARY_PROMPT = """
You are summarizing a video based on scene-level notes (audio + visual context).

Write a global summary of the video, answering:
- What is it about (main topic / content)?
- In what context does it happen (place, type of situation)?
- What is the overall tone (calm, tense, professional, intimate, etc.)?

Rules:
- Use clear, natural language.
- Use the same language as the user instructions if any, otherwise French.
- If you return JSON, use exactly: {"summary": "..."} and nothing else.
""".strip()


# --- MOTEUR VLM ---

class VLMProvider:
    def __init__(self):
        self.current_model_path = None
        self.last_load_time: float = 0.0

    def load_model(self, model_path: str):
        raise NotImplementedError

    def describe_scene(
        self,
        images: List[Image.Image],
        prompt: str,
        max_tokens: int,
    ) -> str:
        raise NotImplementedError

    def unload_model(self):
        pass


class MLXEngine(VLMProvider):
    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None
        self.config = None
        self._generate_fn = None

    def load_model(self, model_path: str):
        if self.model is not None and self.current_model_path == model_path:
            print(f"🍎 [MLX] Modèle VLM déjà chargé : {model_path}")
            self.last_load_time = 0.0
            return

        print(f"🍎 [MLX] Chargement VLM : {model_path}")
        t0 = time.time()
        try:
            from mlx_vlm import load, generate
            from mlx_vlm.utils import load_config

            self.model = None
            self.processor = None
            self.config = None
            gc.collect()

            self.model, self.processor = load(model_path, trust_remote_code=True)
            self.config = load_config(model_path)
            self._generate_fn = generate
            self.current_model_path = model_path
            self.last_load_time = time.time() - t0
            print(f"✅ [MLX] VLM chargé en {self.last_load_time:.2f}s")
        except ImportError:
            raise RuntimeError(
                "Erreur: installez 'mlx-vlm' (pip install mlx-vlm torchvision)"
            )

    def describe_scene(
        self,
        images: List[Image.Image],
        prompt: str,
        max_tokens: int,
    ) -> str:
        if not self.model:
            raise RuntimeError("Modèle VLM MLX non chargé")

        from mlx_vlm.prompt_utils import apply_chat_template

        num_images = max(1, len(images))
        formatted_prompt = apply_chat_template(
            self.processor,
            self.config,
            prompt,
            num_images=num_images,
        )

        imgs = images if len(images) > 1 else images[0]

        output = self._generate_fn(
            self.model,
            self.processor,
            formatted_prompt,
            imgs,
            max_tokens=max_tokens,
            verbose=False,
            temp=0.0,
        )
        text = output.text if hasattr(output, "text") else str(output)
        return text.strip()

    def unload_model(self):
        self.model = None
        self.processor = None
        self.config = None
        self.current_model_path = None
        gc.collect()


class LlamaCppEngine(VLMProvider):
    def __init__(self):
        super().__init__()
        self.llm = None
        self.chat_handler = None

    def _download_if_missing(self, filename: str, repo_id: str):
        if not os.path.exists(filename):
            print(f"⬇️ [Auto-Download] {filename}...")
            try:
                from huggingface_hub import hf_hub_download

                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=".",
                )
                print(f"✅ {filename} prêt.")
            except ImportError:
                raise RuntimeError("Erreur: 'huggingface_hub' manquant.")
            except Exception as e:
                print(f"⚠️ Erreur téléchargement auto : {e}")

    def load_model(self, model_path: str):
        if model_path == DEFAULT_VLM_MODEL_GGUF:
            self._download_if_missing(model_path, DEFAULT_VLM_REPO_GGUF)
            self._download_if_missing(DEFAULT_MMPROJ_GGUF, DEFAULT_VLM_REPO_GGUF)

        if self.llm is not None and self.current_model_path == model_path:
            print(f"🐧/🪟 [Llama.cpp] Modèle VLM déjà chargé : {model_path}")
            self.last_load_time = 0.0
            return

        print(f"🐧/🪟 [Llama.cpp] Chargement VLM : {model_path}")
        t0 = time.time()
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler

            self.llm = None
            gc.collect()

            mmproj = DEFAULT_MMPROJ_GGUF if os.path.exists(DEFAULT_MMPROJ_GGUF) else None
            self.chat_handler = (
                Llava15ChatHandler(clip_model_path=mmproj) if mmproj else None
            )

            self.llm = Llama(
                model_path=model_path,
                chat_handler=self.chat_handler,
                n_gpu_layers=-1,
                n_ctx=4096,
                verbose=False,
            )
            self.current_model_path = model_path
            self.last_load_time = time.time() - t0
            print(f"✅ [Llama.cpp] VLM chargé en {self.last_load_time:.2f}s")
        except ImportError:
            raise RuntimeError("Erreur: installez 'llama-cpp-python'")
        except Exception as e:
            raise RuntimeError(f"Erreur chargement VLM llama.cpp : {e}")

    def describe_scene(
        self,
        images: List[Image.Image],
        prompt: str,
        max_tokens: int,
    ) -> str:
        if not self.llm:
            raise RuntimeError("Modèle VLM llama.cpp non chargé")

        contents = []
        for img in images:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            contents.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                }
            )

        contents.append({"type": "text", "text": prompt})

        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": contents,
                }
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return response["choices"][0]["message"]["content"].strip()

    def unload_model(self):
        self.llm = None
        self.chat_handler = None
        self.current_model_path = None
        gc.collect()


def get_vlm_engine() -> VLMProvider:
    return MLXEngine() if platform.system() == "Darwin" else LlamaCppEngine()


# --- UTILS SCÈNES & VIDÉO ---

def compute_hsv_histogram(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [50, 60],
        [0, 180, 0, 256],
    )
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()


def detect_scenes(
    video_path: str,
    threshold: float,
    min_duration: float,
    max_duration: float,
) -> List[Dict[str, float]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Erreur lecture vidéo")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = int(fps)
    scenes = []
    last_hist = None
    start_sec = 0.0
    prev_sec = 0.0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            curr_sec = frame_idx / fps
            hist = compute_hsv_histogram(frame)
            is_visual_change = False

            if last_hist is not None:
                score = cv2.compareHist(last_hist, hist, cv2.HISTCMP_CORREL)
                if (1.0 - score) > threshold:
                    is_visual_change = True

            duration_ok = (curr_sec - start_sec) >= min_duration
            force_cut = (curr_sec - start_sec) >= max_duration

            if (is_visual_change and duration_ok) or force_cut:
                scenes.append({"start": start_sec, "end": prev_sec})
                start_sec = curr_sec

            last_hist = hist
            prev_sec = curr_sec

        frame_idx += 1

    if prev_sec > start_sec:
        scenes.append({"start": start_sec, "end": prev_sec})

    cap.release()
    return scenes


def download_video_from_url(url: str, output_template: str) -> str:
    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "match_filter": yt_dlp.utils.match_filter_func(
            f"duration < {MAX_TOTAL_VIDEO_DURATION}"
        ),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)
        except yt_dlp.utils.DownloadError as e:
            if "video is too long" in str(e).lower():
                raise ValueError("Vidéo trop longue")
            raise e


def sanitize_filename(filename: str) -> str:
    name = os.path.basename(filename)
    safe_name = "".join(
        [c for c in name if c.isalnum() or c in (" ", ".", "_", "-")]
    )
    return safe_name.strip()[:60] or "video_output"


def validate_video_file(file_path: str) -> float:
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise HTTPException(400, "Fichier vidéo invalide")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()

    duration = frames / fps if fps > 0 else 0
    if duration > MAX_TOTAL_VIDEO_DURATION:
        raise HTTPException(
            400, f"Vidéo trop longue ({duration/60:.1f} min)"
        )
    return duration


def sample_keyframes_for_scene(
    cap: cv2.VideoCapture,
    scene: Dict[str, float],
    base_res: int,
    max_frames: int,
) -> List[Dict[str, Any]]:
    start = scene["start"]
    end = scene["end"]
    duration = max(0.001, end - start)

    n = max(1, min(max_frames, 5))

    if n == 1:
        times = [start + 0.5 * duration]
    else:
        times = [start + (k / (n + 1)) * duration for k in range(1, n + 1)]

    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (base_res, base_res))
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append({"time": round(t, 2), "image": pil_img})

    return frames


# --- AUDIO FEATURES ---

def compute_audio_features_for_scene(
    scene: Dict[str, float],
    whisper_segments: List[Dict[str, Any]],
) -> Dict[str, float]:
    start = scene["start"]
    end = scene["end"]
    duration = max(0.001, end - start)

    segs = [
        s
        for s in whisper_segments
        if s["start"] < end and s["end"] > start
    ]

    if not segs:
        return {
            "speech_duration": 0.0,
            "speaking_rate_wpm": 0.0,
            "speech_ratio": 0.0,
            "silence_ratio": 1.0,
        }

    speech_duration = 0.0
    word_count = 0

    for s in segs:
        seg_start = max(start, s["start"])
        seg_end = min(end, s["end"])
        overlap = max(0.0, seg_end - seg_start)
        speech_duration += overlap

        text = s.get("text", "") or ""
        word_count += len(text.strip().split())

    speech_ratio = min(1.0, speech_duration / duration)
    silence_ratio = max(0.0, 1.0 - speech_ratio)

    if speech_duration > 0:
        speaking_rate_wpm = (word_count / speech_duration) * 60.0
    else:
        speaking_rate_wpm = 0.0

    return {
        "speech_duration": round(speech_duration, 3),
        "speaking_rate_wpm": round(speaking_rate_wpm, 2),
        "speech_ratio": round(speech_ratio, 3),
        "silence_ratio": round(silence_ratio, 3),
    }


def analyze_emotion_from_frame(frame_rgb: np.ndarray) -> Dict[str, Any]:
    return {}


# --- JSON ROBUSTE + NETTOYAGE ---

def safe_json_parse(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    if start == -1:
        return {}

    candidate = text[start:].strip()

    try:
        return json.loads(candidate)
    except Exception:
        pass

    for _ in range(3):
        open_curly = candidate.count("{")
        close_curly = candidate.count("}")
        if close_curly < open_curly:
            candidate += "}" * (open_curly - close_curly)

        open_brack = candidate.count("[")
        close_brack = candidate.count("]")
        if close_brack < open_brack:
            candidate += "]" * (open_brack - close_brack)

        last_brace = max(candidate.rfind("}"), candidate.rfind("]"))
        if last_brace != -1:
            candidate = candidate[: last_brace + 1].strip()

        try:
            return json.loads(candidate)
        except Exception:
            continue

    return {}


def _extract_field_loose(raw: str, field: str) -> str:
    """
    Essaie d'extraire la valeur d'un champ JSON (ex: "summary", "description")
    dans un texte potentiellement tronqué et/ou entouré de ```json, {, } etc.
    """
    if not raw:
        return ""

    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)

    m = re.search(r'"%s"\s*:\s*(.*)' % re.escape(field), text, flags=re.DOTALL)
    if not m:
        return ""

    value = m.group(1)
    value = re.sub(r'^[\s`"{]+', "", value)

    for sep in ["```", "\n\n", "\n}", "}\n", "\n]", "]"]:
        pos = value.find(sep)
        if pos != -1:
            value = value[:pos]
            break

    value = value.rstrip("`}\n\r\t ")
    return value.strip()


def clean_raw_visual_text(raw: str) -> str:
    if not raw:
        return ""
    desc = _extract_field_loose(raw, "description")
    if desc:
        return desc
    text = raw.strip().lstrip("{").rstrip("}")
    return text.strip()


def clean_raw_summary_text(raw: str) -> str:
    if not raw:
        return ""
    s = _extract_field_loose(raw, "summary")
    if s:
        return s
    s = _extract_field_loose(raw, "description")
    if s:
        return s
    text = raw.strip().lstrip("{").rstrip("}")
    return text.strip()


def generate_text_report(
    filename: str,
    duration: float,
    segments: List[Dict[str, Any]],
    process_time: float,
    params: Dict[str, Any],
    global_summary: str = "",
) -> str:
    lines = [
        "### CONTEXTE VIDEO (VideoContext v3.19)",
        f"Source : {filename}",
        f"Durée : {duration:.2f}s | Traitement : {process_time:.2f}s",
        f"Config : Res={params['resolution']}px | Seuil={params['threshold']} "
        f"| Min={params['min_duration']}s | Max={params['max_duration']}s "
        f"| Keyframes/scene={params['keyframes_per_scene']} | RAM_MODE={params['ram_mode']}",
        "",
    ]

    if global_summary:
        lines.append("--- RÉSUMÉ GLOBAL ---")
        lines.append(global_summary)
        lines.append("")
        lines.append("-" * 40)
        lines.append("")

    for seg in segments:
        lines.append(f"⏱️ [{seg['start']:.2f} - {seg['end']:.2f}] SCÈNE {seg['scene_id']}")

        if seg.get("audio_transcript"):
            lines.append(f"   🎙️ TEXTE : \"{seg['audio_transcript']}\"")

        af = seg.get("audio_features") or {}
        if af:
            lines.append(
                f"   🔊 AudioFeatures : speech={af.get('speech_ratio', 0):.2f}, "
                f"silence={af.get('silence_ratio', 0):.2f}, "
                f"wpm={af.get('speaking_rate_wpm', 0):.1f}"
            )

        if seg.get("visual_description"):
            lines.append(f"   👀 VISUEL : {seg['visual_description']}")

        vt = seg.get("visual_tags") or {}
        if vt:
            lines.append(
                "   🧩 Tags : "
                f"people={vt.get('people_count', '?')}, "
                f"place={vt.get('place_type', '?')}, "
                f"action={vt.get('main_action', '?')}, "
                f"tone={vt.get('emotional_tone', '?')}, "
                f"movement={vt.get('movement_level', '?')}"
            )

        lines.append("")

    return "\n".join(lines)


# --- API FASTAPI ---

vlm_engine: Optional[VLMProvider] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vlm_engine, WHISPER_CACHE
    vlm_engine = get_vlm_engine()

    if RAM_MODE == "ram+":
        print("🚀 RAM+ mode: préchargement des modèles...")
        try:
            vlm_engine.load_model(ACTIVE_DEFAULT_VLM)
        except Exception as e:
            print(f"⚠️ Impossible de précharger le VLM par défaut: {e}")
        try:
            t0 = time.time()
            WHISPER_CACHE[DEFAULT_WHISPER_MODEL] = whisper.load_model(DEFAULT_WHISPER_MODEL)
            print(f"✅ Whisper '{DEFAULT_WHISPER_MODEL}' préchargé ({time.time()-t0:.2f}s)")
        except Exception as e:
            print(f"⚠️ Impossible de précharger Whisper '{DEFAULT_WHISPER_MODEL}': {e}")

    yield

    if vlm_engine:
        vlm_engine.unload_model()
    for k, m in list(WHISPER_CACHE.items()):
        del WHISPER_CACHE[k]
    WHISPER_CACHE.clear()
    gc.collect()


app = FastAPI(
    title="VideoContext Engine",
    description="Microservice v3.19 (RAM modes + JSON robuste + nettoyage brut)",
    version="3.19",
    lifespan=lifespan,
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    try:
        paths = openapi_schema["paths"]
        for path, methods in paths.items():
            for method, operation in methods.items():
                request_body = operation.get("requestBody", {})
                content = request_body.get("content", {})
                form_schema = content.get(
                    "application/x-www-form-urlencoded", {}
                ).get("schema", {})
                props = form_schema.get("properties", {})

                for field_name in ["visual_user_prompt", "summary_user_prompt"]:
                    if field_name in props:
                        props[field_name]["format"] = "textarea"
    except Exception as e:
        print("⚠️ Error patching OpenAPI for textareas:", e)

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi


@app.post("/api/v1/analyze")
async def analyze_video(
    video_file: Union[UploadFile, str, None] = File(
        None, description="Fichier vidéo local (Optionnel)"
    ),
    video_url: str = Form(
        "", description="URL YouTube ou direct (laisser vide si upload)"
    ),

    visual_user_prompt: str = Form(
        "Décris factuellement ce qui se passe dans la scène à partir des images, "
        "en te concentrant sur les gestes, la posture, l'ambiance et le contexte. avec un maximum de 80 mots",
        max_length=1000,
        description=(
            "Instructions supplémentaires pour la description visuelle et le nombre de mot max "
            "(style, ce qu'il faut mettre en avant). Laisse vide pour utiliser "
            "le comportement par défaut."
        ),
    ),

    summary_user_prompt: str = Form(
        "Résume la vidéo de façon claire et concise en t'appuyant sur l'ensemble "
        "des scènes, comme si tu expliquais la vidéo à quelqu'un qui ne l'a pas vue avec un maximum de 120 mots.",
        max_length=2000,
        description=(
            "Instructions supplémentaires pour le résumé global (ton, niveau de détail...). et le nombre de mot max"
            "Laisse vide pour utiliser le comportement par défaut."
        ),
    ),

    vlm_model: str = Form(
        ACTIVE_DEFAULT_VLM,
        description="Modèle VLM",
    ),
    whisper_model: str = Form(
        DEFAULT_WHISPER_MODEL,
        description="Modèle Whisper",
    ),

    response_format: Literal["json", "text"] = Form(
        "json", description="Format de sortie"
    ),

    vlm_resolution: int = Form(
        DEFAULT_RESOLUTION, ge=128, le=2048, description="Résolution pour le VLM"
    ),
    scene_threshold: float = Form(
        DEFAULT_SCENE_THRESHOLD, ge=0.01, le=1.0, description="Seuil de changement de scène"
    ),
    min_scene_duration: float = Form(
        DEFAULT_MIN_DURATION, ge=0.5, le=60.0, description="Durée minimale de scène (s)"
    ),
    max_scene_duration: float = Form(
        DEFAULT_MAX_DURATION, ge=5.0, le=300.0, description="Durée maximale de scène (s)"
    ),

    keyframes_per_scene: int = Form(
        DEFAULT_KEYFRAMES_PER_SCENE,
        ge=1,
        le=5,
        description="Nombre de keyframes par scène (1 à 5). 1 = plus rapide.",
    ),

    vlm_max_tokens_scene: int = Form(
        DEFAULT_VLM_MAX_TOKENS_SCENE,
        ge=16,
        le=1024,
        description="Nombre max de tokens générés par le VLM pour chaque scène (par défaut 220).",
    ),
    vlm_max_tokens_summary: int = Form(
        DEFAULT_VLM_MAX_TOKENS_SUMMARY,
        ge=16,
        le=2048,
        description="Nombre max de tokens générés par le VLM pour le résumé global (par défaut 260).",
    ),

    skip_audio: bool = Form(False),
    skip_visual: bool = Form(False),
    generate_txt: bool = Form(False),

    enable_audio_features: bool = Form(
        True, description="Calculer les audio_features par scène ? (True par défaut)"
    ),
    generate_summary: bool = Form(
        True, description="Générer un résumé global de la vidéo ? (True par défaut)"
    ),
):
    if isinstance(video_file, str):
        video_file = None

    if not video_file and not video_url.strip():
        raise HTTPException(
            400, "Veuillez fournir soit un fichier vidéo, soit une URL."
        )

    request_start_time = time.time()

    whisper_infer_time = 0.0
    vlm_infer_time = 0.0
    whisper_load_time = 0.0

    target_whisper = whisper_model
    target_vlm = vlm_model
    temp_path = None
    source_name = "Inconnu"

    try:
        if video_url.strip():
            source_name = video_url
            clean_title = sanitize_filename(video_url)
            try:
                temp_path = download_video_from_url(
                    video_url,
                    f"temp_{int(time.time())}_{clean_title}.%(ext)s",
                )
            except ValueError as ve:
                raise HTTPException(400, str(ve))
        elif video_file:
            ext = Path(video_file.filename).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(400, f"Extension interdite: {ext}")
            source_name = video_file.filename
            clean_name = sanitize_filename(video_file.filename)
            temp_path = os.path.abspath(f"temp_{int(time.time())}_{clean_name}")
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(video_file.file, buffer)

        video_duration = validate_video_file(temp_path)
        start_process = time.time()
        final_segments: List[Dict[str, Any]] = []
        global_summary_text = ""

        print("1️⃣ Détection des scènes...")
        scenes_raw = detect_scenes(
            temp_path,
            threshold=scene_threshold,
            min_duration=min_scene_duration,
            max_duration=max_scene_duration,
        )

        if skip_audio and skip_visual:
            for idx, scene in enumerate(scenes_raw):
                final_segments.append(
                    {
                        "scene_id": idx + 1,
                        "start": round(scene["start"], 2),
                        "end": round(scene["end"], 2),
                        "audio_transcript": "",
                        "audio_features": {},
                        "visual_description": "",
                        "visual_tags": {},
                        "emotion": {},
                    }
                )
        else:
            async with gpu_lock:
                print("🔒 GPU Lock acquis.")

                whisper_segments: List[Dict[str, Any]] = []

                if not skip_audio:
                    if RAM_MODE == "ram+":
                        if target_whisper in WHISPER_CACHE:
                            w_model = WHISPER_CACHE[target_whisper]
                            whisper_load_time = 0.0
                            print(f"🎙️ Whisper '{target_whisper}' déjà en RAM (cache).")
                        else:
                            print(f"🎙️ Chargement Whisper '{target_whisper}' en RAM (ram+)...")
                            t0 = time.time()
                            w_model = whisper.load_model(target_whisper)
                            WHISPER_CACHE[target_whisper] = w_model
                            whisper_load_time = time.time() - t0
                            print(f"✅ Whisper '{target_whisper}' chargé en {whisper_load_time:.2f}s (ram+)")
                    else:
                        print(f"🎙️ Chargement Whisper '{target_whisper}' (ram-)...")
                        t0 = time.time()
                        w_model = whisper.load_model(target_whisper)
                        whisper_load_time = time.time() - t0
                        print(f"✅ Whisper '{target_whisper}' chargé en {whisper_load_time:.2f}s (ram-)")

                    t_infer_start = time.time()
                    audio_res = w_model.transcribe(
                        temp_path,
                        condition_on_previous_text=False,
                        temperature=0.0,
                        compression_ratio_threshold=1.8,
                    )
                    t_infer_end = time.time()
                    whisper_infer_time = t_infer_end - t_infer_start
                    print(f"⏱️ Whisper inference (transcribe): {whisper_infer_time:.2f}s")

                    whisper_segments = audio_res.get("segments", []) or []

                    if RAM_MODE == "ram-":
                        del w_model
                        gc.collect()
                else:
                    whisper_segments = []

                if not skip_visual or generate_summary:
                    t0 = time.time()
                    vlm_engine.load_model(target_vlm)
                    t1 = time.time()
                    vlm_load_time = vlm_engine.last_load_time or (t1 - t0)
                else:
                    vlm_load_time = 0.0

                cap = cv2.VideoCapture(temp_path)

                resize_tuple = (vlm_resolution, vlm_resolution)
                first_frame_img = None
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, f_frame = cap.read()
                if ret:
                    f_frame = cv2.resize(f_frame, resize_tuple)
                    first_frame_img = Image.fromarray(
                        cv2.cvtColor(f_frame, cv2.COLOR_BGR2RGB)
                    )

                for idx, scene in enumerate(scenes_raw):
                    scene_id = idx + 1
                    start_s = scene["start"]
                    end_s = scene["end"]

                    audio_parts = []
                    if not skip_audio and whisper_segments:
                        for s in whisper_segments:
                            if s["start"] < end_s and s["end"] > start_s:
                                txt = (s.get("text") or "").strip()
                                if txt:
                                    audio_parts.append(txt)

                    audio_text = " ".join(audio_parts).strip()

                    audio_features = {}
                    if (
                        not skip_audio
                        and whisper_segments
                        and enable_audio_features
                    ):
                        audio_features = compute_audio_features_for_scene(
                            scene, whisper_segments
                        )

                    visual_description = ""
                    visual_tags: Dict[str, Any] = {}
                    emotion_info_scene: Dict[str, Any] = {}

                    if not skip_visual:
                        keyframes = sample_keyframes_for_scene(
                            cap,
                            scene,
                            base_res=vlm_resolution,
                            max_frames=keyframes_per_scene,
                        )

                        if keyframes:
                            images = [kf["image"] for kf in keyframes]

                            full_visual_prompt = BASE_VISUAL_PROMPT
                            if visual_user_prompt.strip():
                                full_visual_prompt += (
                                    "\n\nAdditional user instructions (follow them if possible):\n"
                                    + visual_user_prompt.strip()
                                )

                            t_vlm_start = time.time()
                            raw = vlm_engine.describe_scene(
                                images,
                                full_visual_prompt,
                                max_tokens=vlm_max_tokens_scene,
                            )
                            t_vlm_end = time.time()
                            vlm_infer_time += t_vlm_end - t_vlm_start

                            data = safe_json_parse(raw)

                            if data and isinstance(data, dict) and "description" in data:
                                visual_description = (data.get("description") or "").strip()
                                tags = data.get("tags") or {}
                                if isinstance(tags, dict):
                                    visual_tags = tags
                                else:
                                    visual_tags = {}
                            else:
                                visual_description = clean_raw_visual_text(raw)
                                visual_tags = {}

                    final_segments.append(
                        {
                            "scene_id": scene_id,
                            "start": round(start_s, 2),
                            "end": round(end_s, 2),
                            "audio_transcript": audio_text,
                            "audio_features": audio_features,
                            "visual_description": visual_description,
                            "visual_tags": visual_tags,
                            "emotion": emotion_info_scene,
                        }
                    )

                if generate_summary:
                    print("🧠 Génération du résumé global...")
                    context_lines = []
                    for s in final_segments:
                        parts = []
                        if s.get("audio_transcript"):
                            parts.append(f"Audio: {s['audio_transcript']}")
                        vt = s.get("visual_tags") or {}
                        if vt:
                            parts.append(
                                "Visuel: "
                                f"people={vt.get('people_count', '?')}, "
                                f"place={vt.get('place_type', '?')}, "
                                f"action={vt.get('main_action', '?')}, "
                                f"tone={vt.get('emotional_tone', '?')}, "
                                f"movement={vt.get('movement_level', '?')}"
                            )
                        elif s.get("visual_description"):
                            parts.append(f"Visuel: {s['visual_description']}")

                        if parts:
                            context_lines.append(
                                f"- Scène {s['scene_id']} "
                                f"({s['start']:.1f}-{s['end']:.1f}s): "
                                + " | ".join(parts)
                            )

                    context_log = "\n".join(context_lines)

                    full_summary_prompt = (
                        "Voici un ensemble de notes sur les scènes d'une vidéo.\n"
                        "Pour chaque scène, tu as une description audio (ce qui est dit) "
                        "et des éléments de contexte visuel (gestes, lieu, ambiance, ton).\n\n"
                        f"{context_log}\n\n"
                        f"{BASE_SUMMARY_PROMPT}"
                    )

                    if summary_user_prompt.strip():
                        full_summary_prompt += (
                            "\n\nUser instructions (apply if possible):\n"
                            + summary_user_prompt.strip()
                        )

                    if not skip_visual and first_frame_img is not None:
                        t_vlm_start = time.time()
                        raw_summary = vlm_engine.describe_scene(
                            [first_frame_img],
                            full_summary_prompt,
                            max_tokens=vlm_max_tokens_summary,
                        )
                        t_vlm_end = time.time()
                        vlm_infer_time += t_vlm_end - t_vlm_start

                        data_sum = safe_json_parse(raw_summary)
                        if data_sum and isinstance(data_sum, dict) and (
                            "summary" in data_sum or "description" in data_sum
                        ):
                            global_summary_text = (
                                data_sum.get("summary")
                                or data_sum.get("description")
                                or ""
                            ).strip()
                        else:
                            global_summary_text = clean_raw_summary_text(raw_summary)
                    else:
                        global_summary_text = context_log[:2000]

                cap.release()
                print("🔓 GPU Lock relâché.")

                if RAM_MODE == "ram-":
                    print("🧹 RAM- mode: déchargement du VLM après traitement.")
                    vlm_engine.unload_model()
                    gc.collect()

        process_duration = time.time() - start_process
        total_request_time = time.time() - request_start_time

        vlm_load_time = getattr(vlm_engine, "last_load_time", 0.0) if vlm_engine else 0.0

        print("===== PROFIL TEMPS (v3.19) =====")
        print(f"⏱️ Total traitement (process_time): {process_duration:.2f}s")
        print(f"⏱️ Total requête (incl. I/O):      {total_request_time:.2f}s")
        print(f"⏱️ Whisper: load={whisper_load_time:.2f}s, infer={whisper_infer_time:.2f}s")
        print(f"⏱️ VLM    : load={vlm_load_time:.2f}s, infer={vlm_infer_time:.2f}s")
        print(f"RAM_MODE = {RAM_MODE}")
        print("================================")

        params = {
            "threshold": scene_threshold,
            "min_duration": min_scene_duration,
            "max_duration": max_scene_duration,
            "vlm": target_vlm,
            "whisper": target_whisper,
            "resolution": vlm_resolution,
            "keyframes_per_scene": keyframes_per_scene,
            "vlm_max_tokens_scene": vlm_max_tokens_scene,
            "vlm_max_tokens_summary": vlm_max_tokens_summary,
            "ram_mode": RAM_MODE,
        }

        text_report_content = generate_text_report(
            source_name,
            video_duration,
            final_segments,
            process_duration,
            params,
            global_summary_text,
        )

        txt_output_name = None
        if generate_txt:
            safe_name = sanitize_filename(source_name)
            txt_output_name = f"{safe_name}_context.txt"
            txt_path = os.path.abspath(txt_output_name)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text_report_content)

        timings = {
            "total_process_time": round(process_duration, 3),
            "total_request_time": round(total_request_time, 3),
            "whisper": {
                "model": target_whisper,
                "load_time": round(whisper_load_time, 3),
                "inference_time": round(whisper_infer_time, 3),
            },
            "vlm": {
                "model": target_vlm,
                "load_time": round(vlm_load_time, 3),
                "inference_time": round(vlm_infer_time, 3),
            },
            "ram_mode": RAM_MODE,
        }

        if response_format == "text":
            download_name = f"{sanitize_filename(source_name)}_context.txt"
            return PlainTextResponse(
                content=text_report_content,
                headers={
                    "Content-Disposition": f'attachment; filename=\"{download_name}\"'
                },
            )
        else:
            return {
                "meta": {
                    "source": source_name,
                    "duration": round(video_duration, 2),
                    "process_time": round(process_duration, 2),
                    "global_summary": global_summary_text,
                    "scene_count": len(final_segments),
                    "models": {
                        "vlm": target_vlm,
                        "whisper": target_whisper,
                    },
                    "skipped": {
                        "audio": skip_audio,
                        "visual": skip_visual,
                    },
                    "params": {
                        "keyframes_per_scene": keyframes_per_scene,
                        "enable_audio_features": enable_audio_features,
                        "generate_summary": generate_summary,
                        "vlm_max_tokens_scene": vlm_max_tokens_scene,
                        "vlm_max_tokens_summary": vlm_max_tokens_summary,
                        "ram_mode": RAM_MODE,
                    },
                    "timings": timings,
                },
                "segments": final_segments,
                "txt_filename": txt_output_name,
            }

    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Erreur interne: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT_SERVEUR)