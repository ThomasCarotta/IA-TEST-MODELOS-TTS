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
# CONFIG (editá tus rutas)
# -------------------------
PIPER_BIN = r"E:\Trae Code\TP_FINAL_IA_TTS\ThomasTTS\Scripts\piper.exe"
MODEL_PATH = r"E:\Trae Code\TP_FINAL_IA_TTS\modelos\es_AR-daniela-high.onnx"
CSV_FILE = r"E:\Trae Code\TP_FINAL_IA_TTS\frases.csv"
OUTPUT_DIR = "salida_piper"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Utilidades
# -------------------------
def check_prereqs():
    if not Path(PIPER_BIN).exists():
        raise FileNotFoundError(f"No se encontró Piper en: {PIPER_BIN}")
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"No existe el modelo: {MODEL_PATH}")
    sidecar = MODEL_PATH + ".json"  # p. ej. es_AR-daniela-high.onnx.json
    if not Path(sidecar).exists():
        raise FileNotFoundError(f"Falta el JSON del modelo: {sidecar}")
    return True

def find_ffmpeg():
    # retorna ruta a ffmpeg o None
    f = shutil.which("ffmpeg.exe") or shutil.which("ffmpeg")
    return f

def export_mp3(ffmpeg_bin, in_wav, out_mp3, bitrate="128k"):
    cmd = [ffmpeg_bin, "-y", "-i", str(in_wav), "-b:a", bitrate, str(out_mp3)]
    creationflags = 0x08000000 if os.name == "nt" else 0  # NO console window en Windows
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creationflags)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg falló al convertir {in_wav} -> {out_mp3}")

def synthesize_with_piper(text, wav_path, model_path):
    """
    Ejecuta Piper via CLI, mide latencia/CPU/RAM.
    Retorna: latency_s, cpu_avg_pct, peak_ram_mb, stderr
    """
    cmd = [PIPER_BIN, "--model", model_path, "--output_file", str(wav_path)]
    creationflags = 0x08000000 if os.name == "nt" else 0

    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creationflags
    )
    p_ps = psutil.Process(proc.pid)
    p_ps.cpu_percent(interval=None)
    peak_rss = 0
    cpu_samples = []

    # Enviar texto
    try:
        proc.stdin.write(text + "\n")
        proc.stdin.flush()
        proc.stdin.close()
    except Exception:
        pass

    # Muestreo de recursos
    while proc.poll() is None:
        try:
            rss = p_ps.memory_info().rss
            peak_rss = max(peak_rss, rss)
            cpu_samples.append(p_ps.cpu_percent(interval=0.05))
        except psutil.NoSuchProcess:
            break

    # Leer stderr tras finalizar
    stderr_output = ""
    try:
        if proc.stderr:
            stderr_output = proc.stderr.read()
    except Exception:
        pass

    latency = time.perf_counter() - start
    cpu_avg = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    peak_ram_mb = peak_rss / (1024 * 1024)
    return latency, cpu_avg, peak_ram_mb, stderr_output.strip()

def wav_info(path):
    if not Path(path).exists():
        return None, None, 0.0
    info = sf.info(str(path))
    dur = info.frames / info.samplerate if info.samplerate else 0.0
    return info.samplerate, info.channels, dur

def generar_plantilla_mos(resultados_ok_df, out_csv="mos_template.csv", evaluadores=10):
    """
    Crea plantilla MOS (10 evaluadores por defecto) solo para outputs OK.
    Columnas: file, texto, evaluator_id, mos_1_5, naturalness_1_5, clarity_1_5, comments
    """
    filas = []
    for _, r in resultados_ok_df.iterrows():
        for i in range(evaluadores):
            filas.append({
                "file": r["wav_path"],
                "texto": r["frase"],
                "evaluator_id": f"eval_{i+1}",
                "mos_1_5": "",
                "naturalness_1_5": "",
                "clarity_1_5": "",
                "comments": ""
            })
    mos_df = pd.DataFrame(filas)
    out_path = Path(OUTPUT_DIR) / out_csv
    mos_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[MOS] Plantilla creada: {out_path}")

# -------------------------
# Main
# -------------------------
def main():
    check_prereqs()

    df = pd.read_csv(CSV_FILE)
    if "texto" not in df.columns:
        raise ValueError("El CSV debe tener una columna llamada 'texto'.")

    frases = [str(t) for t in df["texto"].dropna().tolist()]
    print(f"[INFO] Total frases: {len(frases)}")

    ffmpeg_bin = find_ffmpeg()
    if not ffmpeg_bin:
        print("[WARN] ffmpeg no encontrado: se omitirá MP3 (solo WAV). "
              "Para compatibilidad completa WAV/MP3, instalá ffmpeg.")  # Requisito de compatibilidad. 

    resultados = []
    for i, frase in enumerate(frases, start=1):
        wav_path = Path(OUTPUT_DIR) / f"frase_{i}.wav"
        mp3_path = Path(OUTPUT_DIR) / f"frase_{i}.mp3"
        print(f"[{i}] Sintetizando: {frase[:60]}...")

        try:
            latency, cpu_avg, peak_ram, stderr = synthesize_with_piper(frase, wav_path, MODEL_PATH)
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
    res_path = Path(OUTPUT_DIR) / "resultados_piper.csv"
    res_df = pd.DataFrame(resultados)
    res_df.to_csv(res_path, index=False, encoding="utf-8")
    print(f"\n[OK] Resultados guardados en {res_path}")

    # Resumen global (sólo OK)
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

        # Plantilla MOS (10 evaluadores, como exige la consigna)
        generar_plantilla_mos(ok_df, "mos_template.csv", evaluadores=10)
    else:
        print("[WARN] No hubo síntesis exitosa; no se generó MOS template.")

if __name__ == "__main__":
    main()
