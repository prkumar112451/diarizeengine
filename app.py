from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import whisperx
from datetime import datetime
import logging
import uvicorn
import os
import requests
import uuid
from typing import Optional
import concurrent.futures
import subprocess
from queue import Queue
import json
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
        command = [
            'ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream=channels', '-of', 'default=nw=1:nk=1', wav_file_path
        ]
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        channels = int(process.stdout.strip())
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

def assign_speaker_to_segments(segments, speaker_id):
    for segment in segments:
        segment['speaker'] = speaker_id
        for word in segment.get('words', []):
            word['speaker'] = speaker_id

def merge_segments(output1, output2):
    assign_speaker_to_segments(output1['segments'], "SPEAKER_00")
    assign_speaker_to_segments(output2['segments'], "SPEAKER_01")
    all_segments = sorted(output1['segments'] + output2['segments'], key=lambda s: s['start'])
    return all_segments

def improve_transcription(transcription):
    new_segments = []
    
    for segment in transcription:
        words = segment["words"]
        current_segment = {"start": words[0]["start"], "end": None, "text": "", "words": []}
        
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]
            
            current_segment["words"].append(current_word)
            current_segment["text"] += current_word["word"] + " "
            
            # Check if the gap between current and next word is more than 2 seconds
            if next_word["start"] - current_word["end"] > 2.0:
                current_segment["end"] = current_word["end"]
                new_segments.append(current_segment)
                
                # Start a new segment
                current_segment = {"start": next_word["start"], "end": None, "text": "", "words": []}
        
        # Add the last word and finalize the segment
        current_segment["words"].append(words[-1])
        current_segment["text"] += words[-1]["word"]
        current_segment["end"] = words[-1]["end"]
        new_segments.append(current_segment)
    
    return new_segments

def process_transcription(audio_path: str):
    logger.info("Loading the audio")
    audio_data = whisperx.load_audio(audio_path)
    logger.info("Audio loaded")
    result = model.transcribe(audio_data, batch_size=batch_size)
    logger.info("Transcription complete")
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio_data, device, return_char_alignments=False)
    improved_segments = improve_transcription(aligned_result['segments'])

    # Return the improved segments wrapped in a dictionary to maintain structure
    return {'segments': improved_segments}
    
def transcribe_audio_worker(audio_path, request_id, webhook_url, mask, language_code, use_diarization_model):
    global model, model_a, metadata, diarize_model, language_in_use

    try:
        logger.info("Started processing for request %s", request_id)

        is_audio_stereo = is_stereo(audio_path)
        logger.info("Stereo type found as %s", is_audio_stereo)

        if not use_diarization_model and is_audio_stereo:
            logger.info("Processing stereo audio for request %s", request_id)
            left_path, right_path = split_stereo(audio_path)
            logger.info("Split the audio into 2 parts")
            output_left = process_transcription(left_path)
            output_right = process_transcription(right_path)
            merged_segments = merge_segments(output_left, output_right)
            final_result = {'segments': merged_segments}
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

        try:
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
        task_queue.get()
        task_queue.task_done()

def process_transcription_attributes(result_return):
    try:
        for segment in result_return["transcription"]:
            if "speaker" not in segment:
                segment["speaker"] = "SPEAKER_00"
            for word in segment["words"]:
                if "speaker" not in word:
                    word["speaker"] = segment["speaker"]
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
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=new_max_concurrent_tasks)
    logger.info(f"Max concurrent tasks updated to {new_max_concurrent_tasks}")
    return {"message": f"Max concurrent tasks updated to {new_max_concurrent_tasks}"}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
