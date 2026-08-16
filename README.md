# ⚙️ SWARAAG — Backend Analysis API

<div align="center">
  <p><b>FastAPI & Python ethnomusicological audio-processing engine for Indian Classical & Assamese Borgeet vocal music.</b></p>
</div>

---

## 🚀 Setup & Run

### 1. Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
*Note: Ensure `ffmpeg` is installed on your system (required by `yt-dlp` for audio extraction):*
- **macOS:** `brew install ffmpeg`
- **Ubuntu/Debian:** `sudo apt install ffmpeg`
- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### 2. Start Uvicorn Server
```bash
uvicorn app.main:app --reload --port 8000
```
Visit **`http://localhost:8000/docs`** for interactive API documentation (Swagger UI).

---

## 📡 API Endpoints

| Method | Endpoint | Payload / Description |
|---|---|---|
| `GET` | `/health` | Server status check |
| `POST` | `/analyze` | `{ "source": "https://youtube.com/watch?v=..." }` |
| `POST` | `/analyze/upload` | Multipart file upload (`.wav`, `.mp3`, `.m4a`, `.flac`) |
| `GET` | `/dataset` | Paginated Borgeet audio dataset with filtering |

---

## 🧩 Pipeline Architecture (`app/pipeline/`)

| Module | Responsibility |
|---|---|
| `audio_input.py` | `yt-dlp` download, audio loading, resampling, and normalization |
| `pitch_extraction.py` | High-precision $F_0$ contour extraction via `librosa` pYIN & `CREPE` GPU |
| `tonic_detection.py` | Acoustic tonic ($Sa$) detection via drone/vocal stem separation & voiced-frame thresholding |
| `note_mapping.py` | Cents-relative swara histogram (12-bin & fine 48-bin) and bigram transition matrices |
| `raga_db.py` | Authoritative Borgeet raga definitions (*Gauri*, *Mallar*, *Dhanashri*, *Asowari*, *Suhai*, etc.) |
| `raga_matcher.py` | Cosine similarity & transition-aware matching weighted by Vadi, Samvadi, and Vivadi penalties |
| `tempo.py` | Beat grid and BPM estimation via `librosa` |
| `analyze.py` | Orchestrates pipeline stages and formats JSON schema for frontend rendering |

---

## 📜 Musicological Grounding

The backend database (`raga_db.py`) is curated directly from traditional Sattriya and Borgeet literature (*Shodhganga thesis*, *Goswami 1997*, *Bhuyan 2003*, *Mahanta 2009*):
- Models pentatonic (*Audav-Audav*) Borgeet ragas like **Gauri** and **Mallar** ($S-R-m-P-n$) without Hindustani scale assumptions.
- Integrates traditional Vadi (dominant) and Samvadi (sub-dominant) hierarchies directly into candidate scoring vectors.
