# -*- coding: utf-8 -*-
import os
import time
import shutil
import psutil
import pandas as pd
import soundfile as sf
import subprocess
import torch
import numpy as np
from pathlib import Path

# -------------------------
CSV_FILE = r"E:\Trae Code\TP_FINAL_IA_TTS\frases.csv"
OUTPUT_DIR = "salida_f5_tts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
def check_prereqs():
    """Verifica que las dependencias estén instaladas"""
    try:
        import torch
        import soundfile
        import transformers
        print("[INFO] Todas las dependencias están instaladas")
        return True
    except ImportError as e:
        raise ImportError(f"Dependencia faltante: {e}")

def find_ffmpeg():
    f = shutil.which("ffmpeg.exe") or shutil.which("ffmpeg")
    return f

def export_mp3(ffmpeg_bin, in_wav, out_mp3, bitrate="128k"):
    cmd = [ffmpeg_bin, "-y", "-i", str(in_wav), "-b:a", bitrate, str(out_mp3)]
    creationflags = 0x08000000 if os.name == 'nt' else 0
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creationflags)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg falló al convertir {in_wav} -> {out_mp3}")

def synthesize_with_tts(text, wav_path):
    """
    Usa un modelo TTS REAL que existe en Hugging Face
    """
    try:
        from transformers import pipeline
        
        # Modelos REALES que existen en Hugging Face para español:
        MODEL_CHOICES = [
            "facebook/mms-tts-spa",  # Modelo MULTILINGÜE de Facebook que SÍ existe
            "microsoft/speecht5_tts",  # Modelo de Microsoft que SÍ existe
            "espnet/kan-bayashi_jsut_tts_train_raw_char_tacotron_train.loss.best",  # Modelo japonés pero con soporte multilingüe
        ]
        
        # Usar el primer modelo disponible
        model_name = MODEL_CHOICES[0]
        
        # Inicializar el pipeline (solo una vez)
        if not hasattr(synthesize_with_tts, 'tts_pipeline'):
            print(f"[INFO] Cargando modelo: {model_name}")
            synthesize_with_tts.tts_pipeline = pipeline(
                "text-to-speech", 
                model=model_name,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        
        tts_pipeline = synthesize_with_tts.tts_pipeline
        
        start = time.perf_counter()
        
        # Monitoreo de recursos
        process = psutil.Process()
        process.cpu_percent(interval=None)
        peak_rss = process.memory_info().rss
        cpu_samples = []
        
        # Sintetizar audio - método compatible con modelos reales
        if "mms" in model_name:
            # Para modelos MMS de Facebook
            audio_output = tts_pipeline(text, lang="spa")
        else:
            # Para otros modelos
            audio_output = tts_pipeline(text)
        
        # Monitorear recursos
        cpu_samples.append(process.cpu_percent(interval=0.05))
        peak_rss = max(peak_rss, process.memory_info().rss)
        
        # Extraer datos de audio
        sampling_rate = audio_output["sampling_rate"]
        audio_array = audio_output["audio"]
        
        # Convertir a numpy array
        if isinstance(audio_array, torch.Tensor):
            audio_data = audio_array.cpu().numpy()
        else:
            audio_data = np.array(audio_array)
        
        # Guardar archivo WAV
        sf.write(str(wav_path), audio_data.T, sampling_rate)  # Transponer si es necesario
        
        latency = time.perf_counter() - start
        
        # Calcular métricas de recursos
        cpu_avg = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
        peak_ram_mb = peak_rss / (1024 * 1024)
        
        return latency, cpu_avg, peak_ram_mb, ""
        
    except Exception as e:
        return 0, 0, 0, f"Error en TTS: {str(e)}"

def wav_info(path):
    if not Path(path).exists():
        return None, None, 0.0
    try:
        info = sf.info(str(path))
        dur = info.frames / info.samplerate if info.samplerate else 0.0
        return info.samplerate, info.channels, dur
    except:
        return None, None, 0.0

# -------------------------
def main():
    check_prereqs()

    df = pd.read_csv(CSV_FILE)
    if "texto" not in df.columns:
        raise ValueError("El CSV debe tener una columna llamada 'texto'.")

    frases = [str(t) for t in df["texto"].dropna().tolist()]
    print(f"[INFO] Total frases: {len(frases)}")
    print(f"[INFO] Dispositivo: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    ffmpeg_bin = find_ffmpeg()
    if not ffmpeg_bin:
        print("[WARN] ffmpeg no encontrado: se omitirá MP3 (solo WAV).")

    resultados = []
    for i, frase in enumerate(frases, start=1):
        wav_path = Path(OUTPUT_DIR) / f"frase_{i}.wav"
        mp3_path = Path(OUTPUT_DIR) / f"frase_{i}.mp3"
        print(f"[{i}] Sintetizando: {frase[:60]}...")

        try:
            latency, cpu_avg, peak_ram, error_msg = synthesize_with_tts(frase, wav_path)
            
            if error_msg:
                raise RuntimeError(error_msg)
            
            sr, ch, dur = wav_info(wav_path)
            if dur <= 0:
                raise RuntimeError(f"No se pudo generar WAV: {wav_path.name}")

            rtf = latency / dur if dur > 0 else None

            # Exportar a MP3
            mp3_done = False
            if ffmpeg_bin and dur > 0:
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
                "error": error_msg
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
    res_path = Path(OUTPUT_DIR) / "resultados_tts.csv"
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

if __name__ == "__main__":
    main()