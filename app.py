from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import whisperx
from datetime import datetime
import logging
import traceback
import uvicorn
import os
import requests
import uuid
from typing import Optional
import concurrent.futures
import json
from queue import Queue
import gc
import torch
import subprocess
from pii_masking import mask_transcript

os.environ['TORCH_USE_CUDA_DSA'] = '1'

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

device = "cuda"
batch_size = 8
compute_type = "int8"
asr_options = {"suppress_numerals": True}
language_in_use = "en"
YOUR_HF_TOKEN = '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'

# Load models
model = whisperx.load_model("deepdml/faster-whisper-large-v3-turbo-ct2", device, compute_type=compute_type, asr_options=asr_options, language=language_in_use)
model_a, metadata = whisperx.load_align_model(language_code=language_in_use, device=device)
diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)

# Configuration for thread pool
max_concurrent_tasks = 1
executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_tasks)
task_queue = Queue()

def is_stereo(wav_file_path: str) -> bool:
    try:
        # Use ffprobe to get audio stream information
        command = [
            'ffprobe', '-v', 'error', '-select_streams', 'a:0', 
            '-show_entries', 'stream=channels', '-of', 'default=nw=1:nk=1', wav_file_path
        ]
        # Run the subprocess and capture the output
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Get the number of channels from the output
        channels = int(process.stdout.strip())
        
        # If the number of channels is 2, it's a stereo recording
        return channels == 2
    except subprocess.SubprocessError as e:
        logger.error("Failed to determine if audio is stereo: %s", e)
    except ValueError as e:
        logger.error("Error parsing the number of channels: %s", e)
    return False

def split_stereo(wav_file_path: str):
    base_filename, extension = os.path.splitext(os.path.basename(wav_file_path))
    left_output = f'/tmp/{base_filename}_left{extension}'
    right_output = f'/tmp/{base_filename}_right{extension}'
    command = [
        'ffmpeg', '-i', wav_file_path, '-filter_complex',
        '[0:a]channelsplit=channel_layout=stereo[left][right]',
        '-map', '[left]', left_output, '-map', '[right]', right_output
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return left_output, right_output

def assign_speaker(words, speaker_id):
    for word in words:
        word['speaker'] = speaker_id

def merge_words_to_text(data):
    result, current_text, current_words, current_speaker = [], "", [], None
    current_start, current_end = None, None

    for word in data['words']:
        if not current_words or word['speaker'] == current_speaker:
            # Start time is the start time of the first word in the segment
            if current_start is None:
                current_start = word['start']
            current_text += word['word'] + " "
            current_words.append(word)
            current_end = word['end']  # End time is continuously updated to the last word's end time
            current_speaker = word['speaker']
        else:
            # Append merged data
            result.append({
                'text': current_text.strip(),
                'speaker': current_speaker,
                'words': current_words,
                'start': current_start,
                'end': current_end
            })
            # Reset for the new segment
            current_text = word['word'] + " "
            current_words = [word]
            current_start = word['start']
            current_end = word['end']
            current_speaker = word['speaker']

    # Add the last segment if any
    if current_words:
        result.append({
            'text': current_text.strip(),
            'speaker': current_speaker,
            'words': current_words,
            'start': current_start,
            'end': current_end
        })
    
    return result


def merge_segments(output1, output2):
    words1 = [word for segment in output1['segments'] for word in segment['words']]
    words2 = [word for segment in output2['segments'] for word in segment['words']]
    assign_speaker(words1, "SPEAKER_00")
    assign_speaker(words2, "SPEAKER_01")
    merged_words = words1 + words2
    merged_segments = {'words': sorted(merged_words, key=lambda x: x['start'])}
    return merged_segments

def process_transcription(audio_path: str):
    logger.info("loading the audio")
    audio_data = whisperx.load_audio(audio_path)
    logger.info("audio loaded")
    result = model.transcribe(audio_data, batch_size=batch_size)
    logger.info("transcription complete")
    return whisperx.align(result["segments"], model_a, metadata, audio_data, device, return_char_alignments=False)

def get_gpu_metrics():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            gpu_data = []
            for line in result.stdout.strip().split('\n'):
                metrics = line.split(',')
                gpu_data.append({
                    'Utilisation': float(metrics[0]),
                    'Temperature': float(metrics[1]),
                    'MemoryUsed': int(metrics[2]),
                    'MemoryTotal': int(metrics[3])
                })
            return gpu_data
        logger.error(f"nvidia-smi error: {result.stderr}")
    except Exception as e:
        logger.error("Error fetching GPU metrics: %s", e)

def transcribe_audio_worker(audio_path, request_id, webhook_url, mask, language_code, use_diarization_model):
    global model, model_a, metadata, diarize_model, language_in_use

    try:
        logger.info("Started processing for request %s", request_id)

        # Transcription logic
        final_result = {}

        is_audio_stereo = is_stereo(audio_path)
        logger.info("Stereo type found as %s", is_audio_stereo)
        
        if not use_diarization_model and is_audio_stereo:
            logger.info("Processing stereo audio for request %s", request_id)
            left_path, right_path = split_stereo(audio_path)
            logger.info("Split the audio into 2 parts")
            output_left = process_transcription(left_path)
            logger.info("transcription of left part complete")
            output_right = process_transcription(right_path)
            logger.info("transcription of right part complete")
            merged_segments = merge_segments(output_left, output_right)
            logger.info("merging the left and right")
            final_result = {'segments': merge_words_to_text(merged_segments)}
            logger.info("---------------- all done --------------")

        else:
            logger.info('Processing mono audio for request')
            if language_code != language_in_use:
                model = whisperx.load_model("small", device, compute_type=compute_type, asr_options=asr_options, language=language_code)
                language_in_use = language_code
            
            audio_data = whisperx.load_audio(audio_path)
            result = model.transcribe(audio_data, batch_size=batch_size)

            if language_in_use == 'en':
                result_transcribe = whisperx.align(result["segments"], model_a, metadata, audio_data, device, return_char_alignments=False)
            else:
                result_transcribe = result

            diarize_result = diarize_model(audio_data)
            final_result = whisperx.assign_word_speakers(diarize_result, result_transcribe)
            # Clear audio_data from memory
            del audio_data

        try : 
            if mask:
                mask_transcript(final_result['segments'])
        except Exception as e:
            logger.error("Error processing transcript attributes: %s", e)
            
        result_return = {'transcription': final_result['segments'], 'requestID': request_id}
        process_transcription_attributes(result_return)
        if webhook_url:
            requests.post(webhook_url, json=result_return)

        result_file_path = f'/tmp/{request_id}.txt'
        with open(result_file_path, 'w') as result_file:
            json.dump(result_return, result_file)

        logger.info("Completed processing for request %s", request_id)
        
    except Exception as e:
        logger.error("Error processing audio for request %s: %s", request_id, e)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        task_queue.get()  # Mark task as done
        task_queue.task_done()

def process_transcription_attributes(result_return):
    try:
        for segment in result_return["transcription"]:
            if "speaker" not in segment:
                segment["speaker"] = "SPEAKER_00"
            previous_word = None
            for word in segment["words"]:
                if "speaker" not in word:
                    word["speaker"] = segment["speaker"]
                if "start" not in word or "end" not in word or "score" not in word:
                    if previous_word:
                        word.update({"start": previous_word["end"], "end": previous_word["end"], "score": 0.99})
                    else:
                        next_word = next((w for w in segment["words"] if "start" in w and "end" in w), None)
                        if next_word:
                            word.update({"start": next_word["start"], "end": next_word["start"], "score": 0.99})
                previous_word = word
    except Exception as e:
        logger.error("Error processing transcript attributes: %s", e)

@app.get('/gpu/metrics')
def get_gpu_metrics_route():
    return get_gpu_metrics()

@app.post('/transcribe')
async def transcribe_audio(audio: UploadFile = File(...), 
                           webhook_url: Optional[str] = Form(None),
                           mask: Optional[bool] = Form(False),
                           language_code: Optional[str] = Form("en"),
                           use_diarization_model: Optional[bool] = Form(True),
                           no_of_participants: Optional[int] = Form(2)):
    try:
        logger.info("Received file %s for transcription", audio.filename)
        logger.info("UseDiarizationModel is set to %s", use_diarization_model)

        request_id = str(uuid.uuid4())
        temp_audio_path = f'/tmp/{request_id}_{audio.filename}'
        with open(temp_audio_path, 'wb') as f:
            f.write(await audio.read())

        logger.info("Temp file created at %s for file %s", datetime.utcnow(), audio.filename)

        # Submit task to executor
        task_queue.put(request_id)

        task_args = (temp_audio_path, request_id, webhook_url, mask, language_code, use_diarization_model)
        executor.submit(transcribe_audio_worker, *task_args)

        return {"message": "Audio processing started, result will be sent to webhook URL if provided", "requestID": request_id}

    except Exception as e:
        logger.error("Error receiving audio file: %s", e)
        raise HTTPException(status_code=500, detail=f"Error receiving audio: {str(e)}")

@app.post('/update_batch_size')
async def update_batch_size(new_batch_size: int):
    global batch_size
    batch_size = new_batch_size
    logger.info(f"Batch size updated to {batch_size}")
    return {"message": f"Batch size updated to {batch_size}"}

@app.post('/update_concurrent_tasks')
async def update_concurrent_tasks(new_max_concurrent_tasks: int):
    global executor
    executor.shutdown(wait=False)
    executor = ThreadPoolExecutor(max_workers=new_max_concurrent_tasks)
    logger.info(f"Max concurrent tasks updated to {new_max_concurrent_tasks}")
    return {"message": f"Max concurrent tasks updated to {new_max_concurrent_tasks}"}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
