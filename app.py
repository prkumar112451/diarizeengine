from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Depends, Header
import whisperx
from datetime import datetime
import logging
import traceback
import uvicorn
import os
import requests
from multiprocessing import set_start_method
import uuid
from typing import Optional
import concurrent.futures
import base64
import os
from pii_masking import mask_transcript
import json
from queue import Queue
import gc
import torch

os.environ['TORCH_USE_CUDA_DSA'] = '1'

app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

logger = logging.getLogger(__name__)

device = "cuda"
batch_size = 8
compute_type = "int8"
YOUR_HF_TOKEN = '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'
asr_options = {
    "suppress_numerals": True  # Set suppress_numerals to True here
}
model = whisperx.load_model("small", device, compute_type=compute_type, asr_options=asr_options)
model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device="cuda:0")

# Configuration for thread pool
max_concurrent_tasks = 1  # You can change this as needed
executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_tasks)
task_queue = Queue()
SECRET_KEY = 'your_secret_key_here'


# Define transcribe_audio_worker function
def transcribe_audio_worker(temp_audio_path, request_id, webhook_url, mask):
    try:
        result_return = {}
        # Diarize segments
        device = "cuda"
        compute_type = "int8"
        audio_data = whisperx.load_audio(temp_audio_path)
        # Initialize results
        result_transcribe = None
        diarize_result = None

        logger.info("about to start tasks %s for file %s", datetime.utcnow(), request_id)

        # Perform transcription        
        result = model.transcribe(audio_data, batch_size=batch_size)
        result_transcribe = whisperx.align(result["segments"], model_a, metadata, audio_data, device, return_char_alignments=False)

        # Perform diarization
        diarize_result = diarize_model(audio_data)

        logger.info("both tasks completed at %s for file %s", datetime.utcnow(), request_id)
        #logger.info("transcribe result after alignment %s for file %s", result_transcribe["segments"], request_id)
        #logger.info("diarization result %s for file %s", diarize_result, request_id)

        # Assign word speakers
        final_result = whisperx.assign_word_speakers(diarize_result, result_transcribe)

        # Process transcript for response
        logger.info("result ready to send at %s for file %s", datetime.utcnow(), request_id)
        if mask:
            try:
                mask_transcript(final_result['segments'])
            except Exception as e:
                logger.error("Error masking transcript: %s", e)
                traceback.print_exc()
            
        result_return['transcription'] = final_result['segments']
        result_return['requestID'] = request_id

        # Post-process to fill missing word attributes and assign speakers        
        try:
            for segment in result_return["transcription"]:
                # Assign SPEAKER_00 if no speaker is assigned at the segment level
                if "speaker" not in segment:
                    segment["speaker"] = "SPEAKER_00"
                
                previous_word = None
                for word in segment["words"]:
                    # Assign speaker from segment level if word level speaker is missing
                    if "speaker" not in word:
                        word["speaker"] = segment["speaker"]
    
                    # Fill missing word attributes
                    if "start" not in word or "end" not in word or "score" not in word:
                        if previous_word:
                            word["start"] = word["end"] = previous_word["end"]
                            word["score"] = 0.99
                        else:
                            # If no previous word, use the next word's start and speaker
                            next_word = next((w for w in segment["words"] if "start" in w and "end" in w), None)
                            if next_word:
                                word["start"] = word["end"] = next_word["start"]
                                word["score"] = 0.99
                    previous_word = word
        except Exception as e:
            logger.error("Error masking transcript: %s", e)
            traceback.print_exc()

        # Save result to a text file
        result_file_path = f'/tmp/{request_id}.txt'
        with open(result_file_path, 'w') as result_file:
            json.dump(result_return, result_file)

        logger.info("Processing completed for request %s", request_id)
        if webhook_url is not None:
            requests.post(webhook_url, json=result_return)
    except Exception as e:
        logger.error("Error processing audio: %s", e)
        traceback.print_exc()
    finally:
        # Mark task as done in the queue
        task_queue.get()
        task_queue.task_done()

        # Clear memory
        del audio_data
        del result
        del result_transcribe
        del diarize_result
        del final_result
        torch.cuda.empty_cache()
        gc.collect()
        

# Protected endpoint - requires authentication
@app.post('/transcribe')
async def transcribe_audio(audio: UploadFile = File(...), 
                           webhook_url: Optional[str] = Form(None),
                           mask: Optional[bool] = Form(False)):
    try:
        logger.info("Processing root request at %s for file %s", datetime.utcnow(), audio.filename)
        logger.info(f"Received webhook_url: {webhook_url}")
        request_id = str(uuid.uuid4())
        # Save audio file temporarily
        temp_audio_path = f'/tmp/{audio.filename}'
        with open(temp_audio_path, 'wb') as f:
            content = await audio.read()
            f.write(content)

        logger.info("temp file created at %s for file %s", datetime.utcnow(), audio.filename)

        # Log the current queue size before submitting the task
        queue_size_before = task_queue.qsize()
        logger.info(f"Queue size before adding the task: {queue_size_before}")
        
        # Put task in the queue and submit to the thread pool
        task_queue.put(request_id)

        # Submit the task to the thread pool
        task_args = (temp_audio_path, request_id, webhook_url, mask)
        executor.submit(transcribe_audio_worker, *task_args)

        # Log the current queue size after submitting the task
        queue_size_after = task_queue.qsize()
        logger.info(f"Queue size after adding the task: {queue_size_after}")
        
        # Return a response immediately, while the background task processes the audio
        return {"message": "Audio processing started, result will be sent to webhook URL", "requestID": request_id}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

# Endpoint to update batch size without authentication
@app.post('/update_batch_size')
async def update_batch_size(new_batch_size: int):
    global batch_size
    batch_size = new_batch_size
    logger.info(f"Batch size updated to {batch_size}")
    return {"message": f"Batch size updated to {batch_size}"}

# Endpoint to update concurrent tasks without authentication
@app.post('/update_concurrent_tasks')
async def update_concurrent_tasks(new_max_concurrent_tasks: int):
    global max_concurrent_tasks
    max_concurrent_tasks = new_max_concurrent_tasks
    global executor
    executor.shutdown(wait=False)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_tasks)
    logger.info(f"Max concurrent tasks updated to {max_concurrent_tasks}")
    return {"message": f"Max concurrent tasks updated to {max_concurrent_tasks}"}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0")
