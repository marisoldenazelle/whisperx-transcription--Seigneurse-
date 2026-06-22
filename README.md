# whisperx-transcription--Seigneuse-
# 🎙️ WhisperX — Pipeline de Transcription & Diarisation (Français)

> **Contribution individuelle au projet [InfoVox Tracker – CDingo](https://github.com/PoidsPlume/CDingo)**  
> Hackathon HN 2026 · École Nationale des Chartes – PSL · Équipe de 5 · 5 jours

---

## 🗂️ Contexte

Ce repo contient **ma contribution personnelle** au hackathon [InfoVox Tracker](https://github.com/PoidsPlume/CDingo), organisé par l'ENC–PSL en 2026.

**CDingo** est un pipeline de diarisation automatique appliqué à des flux audio d'une chaîne d'information en continu : segmentation des prises de parole, extraction d'embeddings vocaux (pyannote.audio), et identification des locuteurs par matching temporel sur un corpus média audiovisuel.

**Mon rôle dans l'équipe :** mise en place du pipeline de transcription et de diarisation avec WhisperX, la brique qui transforme le flux audio brut en segments horodatés attribués à chaque locuteur.  

---

## ✨ Ce que fait ce script

| Étape | Outil | Résultat |
|-------|-------|----------|
| Transcription | `faster-whisper` (backend WhisperX) | Texte brut + timestamps |
| Alignement | `wav2vec2` | Recalage au millième de seconde |
| Diarisation | `pyannote/speaker-diarization-3.1` | Attribution par locuteur (SPEAKER_00, SPEAKER_01…) |
| Output | Python natif | Segments `[start – end] Locuteur : texte` |

---

## 🛠️ Prérequis

### 1. Accès aux modèles Pyannote (obligatoire)

La diarisation nécessite d'accepter les conditions d'utilisation de deux modèles sur Hugging Face :

- [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0)

### 2. Token Hugging Face

Générez un token **"Read"** dans vos [paramètres Hugging Face](https://huggingface.co/settings/tokens) et remplacez `VOTRE_TOKEN_ICI` dans le script.

### 3. GPU recommandé

Sur **Google Colab** : sélectionnez `Exécution > Modifier le type d'exécution > T4 GPU`.  
Sans GPU, utilisez `compute_type = "int8"`.

---

## 🚀 Installation

```bash
pip install whisperx huggingface_hub
pip install git+https://github.com/m-bain/whisperX.git --upgrade
```

---

## 📖 Utilisation

1. Placez votre fichier audio (`audio.mp3` ou autre) dans votre répertoire de travail.
2. Remplacez `YOUR_HF_TOKEN` par votre token Hugging Face.
3. Lancez le script.

```python
import whisperx
import torch
import whisperx.asr
from faster_whisper import WhisperModel
from huggingface_hub import login

# --- CONFIGURATION ---
device = "cuda" if torch.cuda.is_available() else "cpu"
audio_file = "ton_fichier.mp3"   # ← à adapter
batch_size = 16
compute_type = "float16"          # ou "int8" sans GPU performant
YOUR_HF_TOKEN = "VOTRE_TOKEN_ICI"

# --- CONNEXION HUGGING FACE ---
login(token=YOUR_HF_TOKEN)

# --- PATCH DE COMPATIBILITÉ ---
# Évite les RecursionError liées aux versions récentes de Python / Torch / faster-whisper
if not hasattr(WhisperModel, "_is_patched"):
    class SafeOptions:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
        def __getattr__(self, name): return None
    whisperx.asr.TranscriptionOptions = SafeOptions
    original_get_prompt = WhisperModel.get_prompt
    def patched_get_prompt(self, tokenizer, previous_tokens, **kwargs):
        kwargs.pop('hotwords', None)
        return original_get_prompt(self, tokenizer, previous_tokens, **kwargs)
    WhisperModel.get_prompt = patched_get_prompt
    WhisperModel._is_patched = True

# --- 1. TRANSCRIPTION ---
model = whisperx.load_model("base", device, compute_type=compute_type)
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)

# --- 2. ALIGNEMENT TEMPOREL ---
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device)

# --- 3. DIARISATION ---
from whisperx.diarize import DiarizationPipeline
diarize_model = DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)
diarize_segments = diarize_model(audio)
result = whisperx.assign_word_speakers(diarize_segments, result)

# --- 4. AFFICHAGE ---
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment.get('speaker', 'Inconnu')}: {segment['text']}")
```

---

## ⚠️ Note sur le patch de compatibilité

Le bloc `_is_patched` en début de script est **essentiel** : il corrige un conflit entre les versions récentes de `faster-whisper` et l'architecture interne de `whisperx`, qui provoque des `RecursionError` sur la fonction `get_prompt`. Sans ce patch, le script plante silencieusement sur certaines configurations.

---

## 🔗 Projet complet

Ce script s'intègre dans le pipeline global **InfoVox Tracker** :  
👉 [github.com/PoidsPlume/CDingo](https://github.com/PoidsPlume/CDingo)

---

## ⚖️ Licence

MIT — libre d'utilisation et de modification.
