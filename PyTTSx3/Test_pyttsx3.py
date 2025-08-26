# -*- coding: utf-8 -*-
import os
import time
import shutil
import psutil
import pandas as pd
import soundfile as sf
import numpy as np
import subprocess
from pathlib import Path
from threading import Thread


CSV_FILE = r"E:\Trae Code\TP_FINAL_IA_TTS\frases.csv"   # debe tener columna 'texto'
OUTPUT_DIR = "salida_pyttsx3"
VOICE_MATCH = "es-MX"   # ejemplo: "Spanish", "Español", "Helena", "Sabina", "es-MX", etc.
RATE_WPM = 170            # velocidad (palabras por minuto aprox.)
VOLUME = 1.0              # 0.0 a 1.0
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Utilidades
# -------------------------
def find_ffmpeg():
    """Devuelve ruta a ffmpeg si está disponible."""
    return shutil.which("ffmpeg.exe") or shutil.which("ffmpeg")

def export_mp3(ffmpeg_bin, in_wav, out_mp3, bitrate="128k"):
    """Convierte WAV -> MP3 usando ffmpeg."""
    cmd = [ffmpeg_bin, "-y", "-i", str(in_wav), "-b:a", bitrate, str(out_mp3)]
    creationflags = 0x08000000 if os.name == "nt" else 0
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creationflags)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg falló al convertir {in_wav} -> {out_mp3}")

def wav_info(path: Path):
    """Retorna (samplerate, channels, duration_s) o (None, None, 0.0) si falla."""
    try:
        info = sf.info(str(path))
        dur = info.frames / info.samplerate if info.samplerate else 0.0
        return info.samplerate, info.channels, float(dur)
    except Exception:
        return None, None, 0.0

def list_and_pick_voice(engine, match: str = None):
    """
    Lista voces disponibles y elige una por 'match' (subcadena en name/id/language).
    Si no hay match, intenta detectar español por language; si no, devuelve None (voz por defecto).
    """
    voices = engine.getProperty("voices")
    chosen = None
    normalized = (match or "").lower().strip()

    # 1) Coincidencia por nombre/ID/language
    if normalized:
        for v in voices:
            name = (getattr(v, "name", "") or "").lower()
            vid  = (getattr(v, "id", "") or "").lower()
            langs = []
            try:
                langs = [str(x, errors="ignore").lower() if isinstance(x, bytes) else str(x).lower()
                         for x in (getattr(v, "languages", []) or [])]
            except Exception:
                pass
            lang_str = " ".join(langs)

            if (normalized in name) or (normalized in vid) or (normalized in lang_str):
                chosen = v.id
                break

    # 2) Si no hubo match, intenta detectar español por languages
    if chosen is None:
        for v in voices:
            langs = []
            try:
                langs = [str(x, errors="ignore").lower() if isinstance(x, bytes) else str(x).lower()
                         for x in (getattr(v, "languages", []) or [])]
            except Exception:
                pass
            if any(("es" in ln or "spa" in ln) for ln in langs):
                chosen = v.id
                break

    # 3) Fallback: voz por defecto (None)
    return chosen

def synthesize_with_pyttsx3(text: str, wav_path: Path, voice_match: str = None, rate: int = 170, volume: float = 1.0):
    """
    Sintetiza con pyttsx3 guardando WAV, midiendo latencia, CPU promedio y pico de RAM.
    Retorna: (latency_s, cpu_avg_pct, peak_ram_mb)
    """
    import pyttsx3

    # La síntesis se hace en un hilo aparte (pyttsx3/SAPI bloquea runAndWait).
    state = {"done": False, "err": None}

    def worker():
        try:
            engine = pyttsx3.init()  # en Windows usa SAPI5
            # Selección de voz
            vid = list_and_pick_voice(engine, voice_match)
            if vid:
                engine.setProperty("voice", vid)
            engine.setProperty("rate", rate)
            engine.setProperty("volume", float(max(0.0, min(1.0, volume))))
            # Encolar y ejecutar
            engine.save_to_file(text, str(wav_path))
            engine.runAndWait()
        except Exception as e:
            state["err"] = str(e)
        finally:
            state["done"] = True

    start = time.perf_counter()
    t = Thread(target=worker, daemon=True)
    t.start()

    p = psutil.Process(os.getpid())
    p.cpu_percent(interval=None)  # prime
    cpu_samples = []
    peak_rss = 0

    # Muestreo mientras corre la síntesis
    while not state["done"]:
        try:
            cpu_samples.append(p.cpu_percent(interval=0.05))
            rss = p.memory_info().rss
            peak_rss = max(peak_rss, rss)
        except psutil.NoSuchProcess:
            break

    latency = time.perf_counter() - start
    if state["err"]:
        raise RuntimeError(state["err"])

    cpu_avg = float(np.mean(cpu_samples)) if cpu_samples else 0.0
    peak_ram_mb = peak_rss / (1024 * 1024)
    return latency, cpu_avg, peak_ram_mb

# -------------------------
# Main
# -------------------------
def main():
    # Cargar frases
    df = pd.read_csv(CSV_FILE)
    if "texto" not in df.columns:
        raise ValueError("El CSV debe tener una columna llamada 'texto'.")
    frases = [str(t) for t in df["texto"].dropna().tolist()]
    print(f"[INFO] Total frases: {len(frases)}")

    ffmpeg_bin = find_ffmpeg()
    if not ffmpeg_bin:
        print("[WARN] ffmpeg no encontrado: se omitirá MP3 (solo WAV).")

    resultados = []
    for i, frase in enumerate(frases, start=1):
        wav_path = Path(OUTPUT_DIR) / f"frase_{i}.wav"
        mp3_path = Path(OUTPUT_DIR) / f"frase_{i}.mp3"
        print(f"[{i}] Sintetizando (pyttsx3): {frase[:60]}...")

        try:
            latency, cpu_avg, peak_ram = synthesize_with_pyttsx3(
                frase, wav_path, VOICE_MATCH, RATE_WPM, VOLUME
            )

            # Info de WAV
            sr, ch, dur = wav_info(wav_path)
            if (dur or 0.0) <= 0.0:
                raise RuntimeError(f"No se pudo leer WAV generado: {wav_path.name}")

            rtf = latency / dur if dur > 0 else None

            # MP3 opcional
            mp3_done = False
            if ffmpeg_bin:
                try:
                    export_mp3(ffmpeg_bin, wav_path, mp3_path)
                    mp3_done = True
                except Exception as e:
                    print(f"   [WARN] Falló conversión MP3: {e}")

            print(f"   -> Latencia={latency:.3f}s | Duración={dur:.3f}s | RTF={rtf:.2f} | "
                  f"CPU~{cpu_avg:.1f}% | RAM^ {peak_ram:.1f}MB | SR={sr}Hz | CH={ch}")

            resultados.append({
                "idx": i,
                "frase": frase,
                "chars": len(frase),
                "wav_path": str(wav_path),
                "mp3_path": str(mp3_path) if mp3_done else "",
                "latencia_s": round(latency, 4),
                "duracion_audio_s": round(dur, 4),
                "rtf": round(rtf, 4) if rtf is not None else None,
                "cpu_promedio_%": round(cpu_avg, 2),
                "ram_max_MB": round(peak_ram, 2),
                "samplerate": sr,
                "channels": ch,
                "error": ""
            })

        except Exception as e:
            print(f"[ERR] {i} -> {e}")
            resultados.append({
                "idx": i,
                "frase": frase,
                "chars": len(frase),
                "wav_path": "",
                "mp3_path": "",
                "latencia_s": None,
                "duracion_audio_s": None,
                "rtf": None,
                "cpu_promedio_%": None,
                "ram_max_MB": None,
                "samplerate": None,
                "channels": None,
                "error": str(e)
            })

    # Guardar resultados
    res_path = Path(OUTPUT_DIR) / "resultados_pyttsx3.csv"
    res_df = pd.DataFrame(resultados)
    res_df.to_csv(res_path, index=False, encoding="utf-8")
    print(f"\n[OK] Resultados guardados en {res_path}")

    # Resumen global
    ok_df = res_df[res_df["latencia_s"].notna()].copy()
    if not ok_df.empty:
        print("\n[MÉTRICAS GLOBALES]")
        print(f"- Frases OK: {len(ok_df)}/{len(res_df)}")
        print(f"- Latencia promedio (s): {ok_df['latencia_s'].mean():.3f}")
        print(f"- RTF promedio: {ok_df['rtf'].mean():.3f}")
        print(f"- CPU promedio (%): {ok_df['cpu_promedio_%'].mean():.1f}")
        print(f"- Pico RAM promedio (MB): {ok_df['ram_max_MB'].mean():.1f}")
        if ok_df['samplerate'].notna().any():
            print(f"- Sample rate dominante: {int(ok_df['samplerate'].mode().iat[0])} Hz")
    else:
        print("[WARN] No hubo síntesis exitosa.")

if __name__ == "__main__":
    main()
