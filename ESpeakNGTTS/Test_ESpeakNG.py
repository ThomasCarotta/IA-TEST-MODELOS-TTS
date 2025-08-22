# -*- coding: utf-8 -*-
import os
import time
import shutil
import psutil
import pandas as pd
import soundfile as sf
import subprocess
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
# Si lo tenés en el PATH, dejá ESPEAK_BIN = None y lo detecta solo.
ESPEAK_BIN = r"C:\Program Files\eSpeak NG\espeak-ng.exe" 
CSV_FILE   = r"E:\Trae Code\TP_FINAL_IA_TTS\frases.csv"
OUTPUT_DIR = "salida_espeak"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parámetros de voz eSpeak NG
ESPEAK_VOICE = "es"      # "es" español, podés probar "es-la", etc. según voces instaladas
ESPEAK_SPEED = 170       # palabras por minuto
ESPEAK_PITCH = 50        # 0-99
ESPEAK_AMPL  = 150       # 0-200 (volumen)
# -------------------------

def find_espeak():
    """Intenta ubicar espeak-ng/espeak en el PATH si ESPEAK_BIN no está definido."""
    if ESPEAK_BIN and Path(ESPEAK_BIN).exists():
        return ESPEAK_BIN
    for name in ("espeak-ng.exe", "espeak-ng", "espeak.exe", "espeak"):
        exe = shutil.which(name)
        if exe:
            return exe
    raise FileNotFoundError(
        "No se encontró 'espeak-ng' en PATH. "
        "Definí ESPEAK_BIN con la ruta absoluta al ejecutable."
    )

def find_ffmpeg():
    """Retorna ruta a ffmpeg o None."""
    return shutil.which("ffmpeg.exe") or shutil.which("ffmpeg")

def export_mp3(ffmpeg_bin, in_wav, out_mp3, bitrate="128k"):
    """Convierte WAV -> MP3 con ffmpeg."""
    cmd = [ffmpeg_bin, "-y", "-i", str(in_wav), "-b:a", bitrate, str(out_mp3)]
    creationflags = 0x08000000 if os.name == "nt" else 0
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creationflags)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg falló al convertir {in_wav} -> {out_mp3}")

def synthesize_with_espeak(text, wav_path, voice="es", speed=170, pitch=50, ampl=150, espeak_bin=None):
    """
    Ejecuta eSpeak NG por CLI, mide latencia/CPU/RAM.
    Retorna: latency_s, cpu_avg_pct, peak_ram_mb, stderr_text
    """
    if not espeak_bin:
        espeak_bin = find_espeak()

    # eSpeak NG recibe el texto como argumento final (no por stdin)
    # -v voz, -s speed (wpm), -p pitch (0-99), -a amplitude (0-200), -w salida.wav
    cmd = [
        espeak_bin,
        "-v", voice,
        "-s", str(speed),
        "-p", str(pitch),
        "-a", str(ampl),
        "-w", str(wav_path),
        text
    ]
    creationflags = 0x08000000 if os.name == "nt" else 0

    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creationflags
    )

    # Medición de recursos del proceso hijo
    p_ps = psutil.Process(proc.pid)
    p_ps.cpu_percent(interval=None)  # prime
    peak_rss = 0
    cpu_samples = []

    while proc.poll() is None:
        try:
            rss = p_ps.memory_info().rss
            peak_rss = max(peak_rss, rss)
            cpu_samples.append(p_ps.cpu_percent(interval=0.05))
        except psutil.NoSuchProcess:
            break

    stderr_text = ""
    try:
        if proc.stderr:
            stderr_text = proc.stderr.read().strip()
    except Exception:
        pass

    latency = time.perf_counter() - start
    cpu_avg = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    peak_ram_mb = peak_rss / (1024 * 1024)

    return latency, cpu_avg, peak_ram_mb, stderr_text

def wav_info(path):
    """Devuelve (samplerate, channels, dur_s) o (None, None, 0.0) si no existe."""
    p = Path(path)
    if not p.exists():
        return None, None, 0.0
    info = sf.info(str(p))
    dur = info.frames / info.samplerate if info.samplerate else 0.0
    return info.samplerate, info.channels, dur

def main():
    espeak = find_espeak()
    print(f"[INFO] Usando eSpeak NG: {espeak}")
    print(f"[INFO] Voz='{ESPEAK_VOICE}' | Velocidad={ESPEAK_SPEED} wpm | Pitch={ESPEAK_PITCH} | Ampl={ESPEAK_AMPL}")

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
        print(f"[{i}] Sintetizando: {frase[:60]}...")

        try:
            latency, cpu_avg, peak_ram, stderr = synthesize_with_espeak(
                frase, wav_path,
                voice=ESPEAK_VOICE, speed=ESPEAK_SPEED, pitch=ESPEAK_PITCH, ampl=ESPEAK_AMPL,
                espeak_bin=espeak
            )

            sr, ch, dur = wav_info(wav_path)
            if dur <= 0:
                raise RuntimeError(f"No se pudo leer WAV generado: {wav_path.name}. STDERR: {stderr}")

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
                "engine": "espeak-ng",
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
                "voice": ESPEAK_VOICE,
                "speed_wpm": ESPEAK_SPEED,
                "pitch": ESPEAK_PITCH,
                "amplitude": ESPEAK_AMPL,
                "error": stderr
            })

        except Exception as e:
            print(f"[ERR] {i} -> {e}")
            resultados.append({
                "engine": "espeak-ng",
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
                "voice": ESPEAK_VOICE,
                "speed_wpm": ESPEAK_SPEED,
                "pitch": ESPEAK_PITCH,
                "amplitude": ESPEAK_AMPL,
                "error": str(e)
            })

    # Guardar resultados
    res_path = Path(OUTPUT_DIR) / "resultados_espeak.csv"
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
