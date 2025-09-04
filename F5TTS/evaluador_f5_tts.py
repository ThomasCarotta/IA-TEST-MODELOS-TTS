# -*- coding: utf-8 -*-
# evaluador_f5_tts_corregido.py
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

# ------------------------- CONFIGURACIÓN -------------------------
CSV_FILE = r"frases.csv"
OUTPUT_DIR = "salida_f5_tts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------- FUNCIONES -------------------------
def check_prereqs():
    """Verifica que todas las dependencias estén instaladas"""
    try:
        import torch
        import soundfile
        import transformers
        print("[INFO] Todas las dependencias están instaladas")
        return True
    except ImportError as e:
        raise ImportError(f"Dependencia faltante: {e}")

def find_ffmpeg():
    """Busca ffmpeg en el sistema"""
    ffmpeg_path = shutil.which("ffmpeg.exe") or shutil.which("ffmpeg")
    if not ffmpeg_path:
        print("[WARN] ffmpeg no encontrado en el sistema")
    return ffmpeg_path

def export_mp3(ffmpeg_bin, wav_path, mp3_path, bitrate="128k"):
    """Convierte WAV a MP3 usando ffmpeg"""
    try:
        cmd = [
            ffmpeg_bin, "-y", "-i", str(wav_path), 
            "-b:a", bitrate, "-ac", "1", str(mp3_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception as e:
        print(f"   [WARN] Error convirtiendo a MP3: {e}")
        return False

def initialize_tts():
    """Inicializa el modelo TTS para español"""
    try:
        from transformers import VitsModel, AutoTokenizer
        import torch
        
        # Modelo específico para español que funciona
        model_name = "facebook/mms-tts-spa"
        
        print(f"[INFO] Cargando modelo: {model_name}")
        
        # Cargar modelo y tokenizer correctamente
        model = VitsModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"No se pudo inicializar TTS: {e}")

def synthesize_text(model, tokenizer, text, wav_path):
    """Sintetiza texto a audio"""
    try:
        start_time = time.perf_counter()
        
        # Monitoreo de recursos
        process = psutil.Process()
        process.cpu_percent(interval=None)
        cpu_samples = []
        peak_ram = process.memory_info().rss
        
        # Tokenizar el texto
        inputs = tokenizer(text, return_tensors="pt")
        
        # Sintetizar audio
        with torch.no_grad():
            output = model(**inputs)
            waveform = output.waveform[0].numpy()
        
        # Monitorear recursos
        cpu_samples.append(process.cpu_percent(interval=0.1))
        peak_ram = max(peak_ram, process.memory_info().rss)
        
        # Guardar archivo WAV
        sampling_rate = model.config.sampling_rate
        sf.write(str(wav_path), waveform, sampling_rate)
        
        latency = time.perf_counter() - start_time
        cpu_avg = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
        peak_ram_mb = peak_ram / (1024 * 1024)
        
        return latency, cpu_avg, peak_ram_mb, ""
        
    except Exception as e:
        return 0.0, 0.0, 0.0, str(e)

def get_audio_info(wav_path):
    """Obtiene información del archivo de audio"""
    try:
        if not os.path.exists(wav_path):
            return None, None, 0.0
        
        info = sf.info(str(wav_path))
        duration = info.duration
        samplerate = info.samplerate
        channels = info.channels
        
        return samplerate, channels, duration
    except:
        return None, None, 0.0

# ------------------------- PROGRAMA PRINCIPAL -------------------------
def main():
    print("=== EVALUACIÓN TTS PARA ESPAÑOL ===")
    
    # Verificar dependencias
    check_prereqs()
    
    # Cargar frases desde CSV
    try:
        df = pd.read_csv(CSV_FILE)
        if "texto" not in df.columns:
            raise ValueError("El CSV debe tener una columna 'texto'")
        
        frases = df["texto"].dropna().astype(str).tolist()
        print(f"[INFO] {len(frases)} frases cargadas desde el CSV")
        
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el CSV: {e}")
        return
    
    # Buscar ffmpeg
    ffmpeg_bin = find_ffmpeg()
    
    # Inicializar modelo TTS
    try:
        model, tokenizer = initialize_tts()
        print(f"[INFO] Modelo cargado correctamente - Sampling rate: {model.config.sampling_rate}Hz")
    except Exception as e:
        print(f"[ERROR] {e}")
        return
    
    # Procesar cada frase
    resultados = []
    print("\n=== COMIENZA LA SÍNTESIS ===")
    
    for i, frase in enumerate(frases, 1):
        print(f"[{i:2d}] Sintetizando: {frase[:60]}{'...' if len(frase) > 60 else ''}")
        
        # Configurar paths de salida
        wav_path = Path(OUTPUT_DIR) / f"frase_{i}.wav"
        mp3_path = Path(OUTPUT_DIR) / f"frase_{i}.mp3"
        
        try:
            # Sintetizar audio
            latency, cpu_avg, peak_ram, error_msg = synthesize_text(model, tokenizer, frase, wav_path)
            
            if error_msg:
                raise RuntimeError(error_msg)
            
            # Obtener información del audio
            samplerate, channels, duration = get_audio_info(wav_path)
            
            if duration <= 0:
                raise RuntimeError("Audio con duración inválida")
            
            # Calcular RTF
            rtf = latency / duration if duration > 0 else None
            
            # Convertir a MP3
            mp3_done = False
            if ffmpeg_bin and os.path.exists(wav_path):
                mp3_done = export_mp3(ffmpeg_bin, wav_path, mp3_path)
            
            # Mostrar métricas
            print(f"   ✓ Latencia: {latency:.3f}s | Duración: {duration:.3f}s | RTF: {rtf:.3f}")
            print(f"   ✓ CPU: {cpu_avg:.1f}% | RAM: {peak_ram:.1f}MB | SR: {samplerate}Hz")
            
            # Guardar resultados
            resultados.append({
                "idx": i,
                "frase": frase,
                "chars": len(frase),
                "wav_path": str(wav_path),
                "mp3_path": str(mp3_path) if mp3_done else "",
                "latencia_s": round(latency, 4),
                "duracion_audio_s": round(duration, 4),
                "rtf": round(rtf, 4) if rtf is not None else None,
                "cpu_promedio_%": round(cpu_avg, 2),
                "ram_java_max_MB": round(peak_ram, 2),
                "samplerate": samplerate,
                "channels": channels,
                "error": ""
            })
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
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
    
    # Guardar resultados en CSV
    resultados_df = pd.DataFrame(resultados)
    csv_output_path = Path(OUTPUT_DIR) / "resultados_f5_tts.csv"
    resultados_df.to_csv(csv_output_path, index=False, encoding="utf-8")
    
    # Mostrar resumen
    successful = resultados_df[resultados_df["latencia_s"].notna()]
    print(f"\n=== RESUMEN ===")
    print(f"Frases procesadas: {len(resultados)}")
    print(f"Frases exitosas: {len(successful)}")
    print(f"Frases con error: {len(resultados) - len(successful)}")
    print(f"Resultados guardados en: {csv_output_path}")

if __name__ == "__main__":
    main()