# -*- coding: utf-8 -*-
"""
Test / benchmark offline para MaryTTS (Python 3.8).
- Lee CSV con columna 'texto'
- Genera WAV (y MP3 si ffmpeg está disponible)
- Mide latencia cliente, RTF (latencia/duración), muestrea CPU y pico RAM del proceso Mary (si se detecta)
- Opcional: arranca MaryTTS con gradlew si no está corriendo y se provee MARY_ROOT
"""
import os
import time
import shutil
import subprocess
from pathlib import Path

import requests
import pandas as pd
import psutil
import soundfile as sf
import numpy as np

# ---------------- CONFIGURACIÓN: EDITA ESTO ----------------
CSV_FILE = r"E:\Trae Code\TP_FINAL_IA_TTS\frases.csv"   # debe tener columna 'texto'
OUTPUT_DIR = "salida_marytts"
MARY_URL_BASE = "http://localhost:59125"               # base
MARY_PROCESS_ENDPOINT = "/process"
VOICE = "cmu-slt-hsmm"     # cambialo por la voz que tengas instalada (VER nota abajo)
LOCALE = "en_US"           # si tenés voz en es_AR u otra, ponela
MARY_ROOT = r"E:\Trae Code\TP_FINAL_IA_TTS\MaryTTS\marytts"  # ruta al repo/gradle (opcional)
AUTO_START_MARY = True     # si True: intentará arrancar gradle en MARY_ROOT si server no responde
START_TIMEOUT = 90         # segundos para esperar que Mary arranque
FFMPEG_BITRATE = "128k"
# ----------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_ffmpeg():
    return shutil.which("ffmpeg.exe") or shutil.which("ffmpeg")


def export_mp3(ffmpeg_bin, in_wav, out_mp3, bitrate="128k"):
    cmd = [ffmpeg_bin, "-y", "-i", str(in_wav), "-b:a", bitrate, str(out_mp3)]
    creationflags = 0x08000000 if os.name == "nt" else 0
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creationflags)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg falló al convertir {in_wav} -> {out_mp3}: {proc.stderr.decode(errors='ignore')}")


def wav_info(path: Path):
    try:
        info = sf.info(str(path))
        dur = info.frames / info.samplerate if info.samplerate else 0.0
        return int(info.samplerate), int(info.channels), float(dur)
    except Exception:
        return None, None, 0.0


def is_mary_up(base_url: str):
    try:
        r = requests.get(base_url + "/", timeout=2)
        return r.status_code in (200, 403, 404)  # la root puede devolver HTML (200) o 403/404 en algunas builds
    except Exception:
        return False


def start_mary(mary_root: str, timeout: int = 60):
    """
    Arranca MaryTTS usando gradlew (en mary_root). Devuelve el Popen si lo arranca, o None.
    """
    if not mary_root:
        raise RuntimeError("MARY_ROOT no está configurado.")
    gradlew = "gradlew.bat" if os.name == "nt" else "./gradlew"
    cmd = [gradlew, "run"]
    creationflags = 0x08000000 if os.name == "nt" else 0
    proc = subprocess.Popen(
        cmd,
        cwd=mary_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
        text=True,
        shell=False
    )
    # Esperar hasta que HTTP responda o timeout
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        if is_mary_up(MARY_URL_BASE):
            return proc
        time.sleep(0.5)
    # Si llegó acá, no arrancó a tiempo
    # intentar leer algo de stdout/stderr para el log (no bloqueante)
    try:
        out, err = proc.communicate(timeout=1)
    except Exception:
        out, err = "", ""
    proc.terminate()
    raise RuntimeError(f"No arrancó MaryTTS en {timeout}s. stdout:\n{out}\nstderr:\n{err}")


def find_mary_java_process():
    """Busca procesos java con 'mary' o 'marytts' en cmdline; devuelve psutil.Process o None."""
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            name = (p.info.get("name") or "").lower()
            cmd = " ".join(p.info.get("cmdline") or []).lower()
            if "java" in name and ("mary" in cmd or "marytts" in cmd or "org.mary" in cmd):
                return p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


def synthesize_with_marytts(text: str, wav_path: Path, voice: str, locale: str, mary_proc: psutil.Process = None, timeout=60):
    """
    Llama al endpoint /process de MaryTTS (streaming) y escribe WAV en wav_path.
    Mide latencia cliente en segundos, CPU promedio (%) y pico RAM (MB) sobre mary_proc si está (o None).
    """
    params = {
        "INPUT_TEXT": text,
        "INPUT_TYPE": "TEXT",
        "LOCALE": locale,
        "VOICE": voice,
        "OUTPUT_TYPE": "AUDIO",
        "AUDIO": "WAVE_FILE",
    }
    url = MARY_URL_BASE + MARY_PROCESS_ENDPOINT
    cpu_samples = []
    peak_rss = 0

    # Preparar muestreo CPU si detectamos proceso
    p_mary = mary_proc
    try:
        if p_mary and not p_mary.is_running():
            p_mary = None
    except Exception:
        p_mary = None

    # iniciar petición
    start = time.perf_counter()
    with requests.get(url, params=params, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(wav_path, "wb") as f:
            # leer chunks e ir muestreando recursos
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                # muestreo
                try:
                    if p_mary:
                        try:
                            rss = p_mary.memory_info().rss
                            peak_rss = max(peak_rss, rss)
                            cpu_samples.append(p_mary.cpu_percent(interval=0.05))
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            p_mary = None
                    else:
                        # si no hay process mary, muestreamos CPU del sistema como referencia
                        cpu_samples.append(psutil.cpu_percent(interval=0.05))
                except Exception:
                    pass

    latency = time.perf_counter() - start
    cpu_avg = float(np.mean(cpu_samples)) if cpu_samples else 0.0
    peak_ram_mb = peak_rss / (1024 * 1024) if peak_rss else None
    return latency, cpu_avg, peak_ram_mb


def main():
    # prereqs
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        raise SystemExit(f"No se pudo leer CSV {CSV_FILE}: {e}")

    if "texto" not in df.columns:
        raise SystemExit("El CSV debe tener una columna llamada 'texto'")

    frases = [str(t) for t in df["texto"].dropna().tolist()]
    print(f"[INFO] Frases a sintetizar: {len(frases)}")

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("[WARN] ffmpeg no encontrado: se omitirá conversión a MP3 (solo WAV).")

    # Comprobar Mary
    mary_proc_popen = None
    started_mary = False
    if is_mary_up(MARY_URL_BASE):
        print("[INFO] MaryTTS responde en", MARY_URL_BASE)
    else:
        if AUTO_START_MARY and MARY_ROOT:
            print("[INFO] MaryTTS no responde. Intentando arrancar con gradlew en:", MARY_ROOT)
            mary_proc_popen = start_mary(MARY_ROOT, timeout=START_TIMEOUT)
            started_mary = True
            print("[OK] MaryTTS arrancado (pendiente).")
        else:
            raise SystemExit("MaryTTS no responde en {}. Arrancalo manualmente o configura MARY_ROOT y AUTO_START_MARY.".format(MARY_URL_BASE))

    # Detectar proceso Java/Mary para muestrear recursos
    mary_ps = find_mary_java_process()
    if mary_ps:
        print("[INFO] Se detectó proceso Mary/Java (pid={}): {}".format(mary_ps.pid, mary_ps.name()))
    elif mary_proc_popen:
        try:
            mary_ps = psutil.Process(mary_proc_popen.pid)
            print("[INFO] Usando Popen pid={} como proceso Mary".format(mary_ps.pid))
        except Exception:
            mary_ps = None

    # Iterar frases
    resultados = []
    for i, frase in enumerate(frases, start=1):
        wav_path = Path(OUTPUT_DIR) / f"frase_{i}.wav"
        mp3_path = Path(OUTPUT_DIR) / f"frase_{i}.mp3"
        print(f"[{i}] Sintetizando: {frase[:70]}...")

        try:
            latency, cpu_avg, peak_ram = synthesize_with_marytts(frase, wav_path, VOICE, LOCALE, mary_proc=mary_ps, timeout=120)

            sr, ch, dur = wav_info(wav_path)
            if (dur or 0.0) <= 0.0:
                raise RuntimeError("WAV vacío o no legible")

            rtf = latency / dur if dur > 0 else None

            mp3_done = False
            if ffmpeg:
                try:
                    export_mp3(ffmpeg, wav_path, mp3_path, bitrate=FFMPEG_BITRATE)
                    mp3_done = True
                except Exception as e:
                    print("   [WARN] No pude crear MP3:", e)

            print(f"   -> Latencia={latency:.3f}s | Dur={dur:.3f}s | RTF={rtf:.3f} | CPU~{cpu_avg:.1f}% | RAM^ {peak_ram or 0:.1f}MB | SR={sr}Hz")

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
                "ram_java_max_MB": round(peak_ram, 2) if peak_ram else None,
                "samplerate": sr, "channels": ch, "error": ""
            })

        except Exception as e:
            print("[ERR] ", e)
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
                "ram_java_max_MB": None,
                "samplerate": None,
                "channels": None,
                "error": str(e)
            })

    # Guardar CSV de resultados
    res_path = Path(OUTPUT_DIR) / "resultados_marytts.csv"
    pd.DataFrame(resultados).to_csv(res_path, index=False, encoding="utf-8")
    print("[OK] Resultados guardados en:", res_path)

    # Si el script arrancó Mary, avisar al usuario (no lo mata automáticamente)
    if started_mary:
        print("\n[NOTA] Este script arrancó MaryTTS en background. No lo maté automáticamente.")
        print("Si querés cerrarlo: busca el proceso Java/gradle y terminálo, o cerrá la terminal donde se ejecutó gradle.")
    print("Fin.")

if __name__ == "__main__":
    main()
