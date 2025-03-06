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
import gc
import torch
from pii_masking import mask_transcript
import bisect
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import librosa
from whisperx_pyannote import Pyannote
import soundfile as sf
import noisereduce as nr
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

#diarization config
vad_onset = 0.5
vad_offset = 0.3
chunk_size = 30

# Load models
model = whisperx.load_model("small", device, compute_type=compute_type, asr_options=asr_options, language=language_in_use)
model_a, metadata = whisperx.load_align_model(language_code=language_in_use, device=device)
diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)
#vad_model = load_silero_vad()
vad_model = Pyannote(device=device, vad_onset=vad_onset, vad_offset=vad_offset)

# Configuration for thread pool
max_concurrent_tasks = 1
executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_tasks)
task_queue = Queue()

def denoise_audio(audio_path):
    try:
        # Check if GPU is available
        if torch.cuda.is_available():
            logger.info("GPU is available, using it for noise reduction.")
            use_gpu = True
        else:
            logger.info("GPU is not available, using CPU for noise reduction.")
            use_gpu = False
        
        # Read the audio file
        data, rate = sf.read(audio_path)
        
        # Reduce noise with GPU if available, otherwise use CPU
        reduced_noise = nr.reduce_noise(
            y=data, sr=rate, n_std_thresh_stationary=1.5, 
            stationary=True, use_torch=use_gpu
        )
        
        # Generate the new file path by replacing '.wav' with '_denoise.wav'
        denoise_file_path = audio_path.replace('.wav', '_denoise.wav')
        
        # Write the denoised audio to the new file
        sf.write(denoise_file_path, reduced_noise, rate)
        
        logger.info(f"Denoised audio saved to: {denoise_file_path}")
        return denoise_file_path

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return audio_path
        
def diarization(audio_path):
    waveform, sample_rate = librosa.load(audio_path, sr=16000)
    audio = {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": sample_rate}    
    vad_segments = vad_model(audio)
    vad_segments = Pyannote.merge_chunks(
        vad_segments, chunk_size, onset=vad_onset)
    result = convert_pyannote_to_silero_format(vad_segments)
    return result
    
def convert_pyannote_to_silero_format(pyannote_output):
    # This function converts each 'segments' key in the pyannote output
    # into the format expected for silero VAD outputs.
    silero_format_segments = []

    for chunk in pyannote_output:
        # Extract each sub-segment within 'segments'
        for seg in chunk['segments']:
            segment_dict = {
                'start': seg[0],  # Start of the sub-segment
                'end': seg[1]    # End of the sub-segment
            }
            silero_format_segments.append(segment_dict)

    return silero_format_segments
    
def remove_duplicate_segments_withspeaker_withtext(segments):
    filtered_segments = []
    for i, segment in enumerate(segments):
        if i == 0 or not (
            segment['text'] == segments[i - 1]['text'] and 
            segment['speaker'] == segments[i - 1]['speaker']
        ):
            filtered_segments.append(segment)
    return filtered_segments

def remove_duplicate_segments_withtext(segments):
    filtered_segments = []
    for i, segment in enumerate(segments):
        if i == 0 or not (
            segment['text'] == segments[i - 1]['text']
        ):
            filtered_segments.append(segment)
    return filtered_segments

def process_segment_words(segments):
    from collections import Counter

    def clean_segment(segment):
        # Split words from the 'text' field and count occurrences
        word_counts = Counter(word.lower() for word in segment['text'].split())
        
        # Identify words repeated more than 5 times
        repeated_words = {word for word, count in word_counts.items() if count > 5}

        # Initialize processed text and words array
        processed_text = []
        processed_words = []
        seen_repeated_words = set()

        # Process the 'text' and 'words' arrays
        for word_info in segment['words']:
            word_lower = word_info['word'].lower()

            if word_lower in repeated_words:
                if word_lower not in seen_repeated_words:
                    # Keep the first instance of repeated word
                    seen_repeated_words.add(word_lower)
                    processed_text.append(word_info['word'])
                    processed_words.append(word_info)
            else:
                # Keep non-repeated words
                processed_text.append(word_info['word'])
                processed_words.append(word_info)

        # Update segment with processed text and words
        segment['text'] = ' '.join(processed_text)
        segment['words'] = processed_words

        return segment

    # Process each segment
    return [clean_segment(segment) for segment in segments]

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
        try:
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
        
        except Exception as e:
            print(f"An error occurred while processing the segment: {e}")
            # You can add any additional error handling logic here
    
    return new_segments

def process_transcription(audio_path: str):
    try:
        logger.info("Loading the audio")
        audio_data = whisperx.load_audio(audio_path)
        logger.info("Audio loaded")
        
        # Transcribe the audio
        result = model.transcribe(audio_data, batch_size=batch_size)
        logger.info("Transcription complete")
        
        # Align transcription with the audio data
        aligned_result = whisperx.align(result["segments"], model_a, metadata, audio_data, device, return_char_alignments=False)
        
        # Improve segments and return the result
        try:
            logger.info("stage 1")
            aligned_result['segments'] = process_segment_words(aligned_result['segments'])
        except Exception as e:
            logger.error("Error processing subprocessing: %s", e) 

        try:
            logger.info("stage 2")
            aligned_result['segments'] = remove_duplicate_segments(aligned_result['segments'])
        except Exception as e:
            logger.error("Error processing subprocessing: %s", e) 

        try:
            logger.info("stage 3")
            aligned_result['segments'] = improve_transcription(aligned_result['segments'])
        except Exception as e:
            logger.error("Error processing subprocessing: %s", e) 

        try:
            logger.info("stage 4")
            aligned_result['segments'] = adjust_segment_times(aligned_result['segments'])
        except Exception as e:
            logger.error("Error processing subprocessing: %s", e) 
        
        return {'segments': aligned_result['segments']}
    
    except Exception as e:
        logger.error("Error processing transcription: %s", e)  # Updated error message
        return {'segments': []}  # Return an empty result to avoid breaking the pipeline
    
    finally:
        # Delete variables if they were created
        if 'audio_data' in locals():
            del audio_data
        if 'result' in locals():
            del result
        if 'aligned_result' in locals():
            del aligned_result
        
        # Clear GPU memory and run garbage collection
        torch.cuda.empty_cache()
        gc.collect()

def get_vad_range(audio_path):
    vad_model = load_silero_vad()
    wav = read_audio(audio_path)
    speech_timestamps = get_speech_timestamps(
      wav,
      vad_model,
      return_seconds=True,  # Return speech timestamps in seconds (default is samples)
    )
    return speech_timestamps
    
def find_vad_interval(vad_times, time_point):
    """
    Uses binary search to find the closest VAD interval just smaller than time_point.
    """
    # Convert time_point to float, in case it's a string
    time_point = float(time_point)
    
    # Extract only the 'start' times from vad_times for binary search
    start_times = [float(vad['start']) for vad in vad_times]
    
    # Perform binary search to find the closest interval
    try:
        idx = bisect.bisect_right(start_times, time_point) - 1
        print(f"Closest interval index: {idx}")
    except Exception as e:
        print(f"Error during binary search: {e}")
    
    return idx if idx >= 0 else None



def assign_speaker_to_word(word_start, word_end, vad_left, vad_right):
    """
    Assign a speaker (SPEAKER_01 or SPEAKER_02) to a word based on its time range using binary search.
    If no interval fully contains the word, the speaker with the closest interval is assigned.
    """
    # Helper function to compute the distance between word and VAD interval
    def compute_distance(vad_start, vad_end, word_start, word_end):
        if word_end < vad_start:  # word is before the vad interval
            return vad_start - word_end
        elif word_start > vad_end:  # word is after the vad interval
            return word_start - vad_end
        else:  # word overlaps with the vad interval
            return 0

    try:
        # Find the closest VAD intervals for left (SPEAKER_01)
        left_start_idx = find_vad_interval(vad_left, word_start)
        left_end_idx = find_vad_interval(vad_left, word_end)

        left_closest_distance = float('inf')
        left_speaker = None

        # Iterate through VAD intervals between these indices for SPEAKER_01
        if left_start_idx is not None and left_end_idx is not None:
            for i in range(left_start_idx, left_end_idx + 1):
                vad_start = vad_left[i]['start']
                vad_end = vad_left[i]['end']
                if vad_start <= word_end and word_start <= vad_end:
                    return "SPEAKER_00"
                else:
                    # Compute the distance if not overlapping
                    distance = compute_distance(vad_start, vad_end, word_start, word_end)
                    if distance < left_closest_distance:
                        left_closest_distance = distance
                        left_speaker = "SPEAKER_00"
    except Exception as e:
        print(f"Error processing SPEAKER_01 VAD intervals: {e}")

    try:
        # Find the closest VAD intervals for right (SPEAKER_02)
        right_start_idx = find_vad_interval(vad_right, word_start)
        right_end_idx = find_vad_interval(vad_right, word_end)

        right_closest_distance = float('inf')
        right_speaker = None

        # Iterate through VAD intervals between these indices for SPEAKER_02
        if right_start_idx is not None and right_end_idx is not None:
            for i in range(right_start_idx, right_end_idx + 1):
                vad_start = vad_right[i]['start']
                vad_end = vad_right[i]['end']
                if vad_start <= word_end and word_start <= vad_end:
                    return "SPEAKER_01"
                else:
                    # Compute the distance if not overlapping
                    distance = compute_distance(vad_start, vad_end, word_start, word_end)
                    if distance < right_closest_distance:
                        right_closest_distance = distance
                        right_speaker = "SPEAKER_01"
    except Exception as e:
        print(f"Error processing SPEAKER_02 VAD intervals: {e}")

    # If no interval contains the word, assign the speaker with the closest interval
    try:
        if left_closest_distance < right_closest_distance:
            return left_speaker if left_speaker else "UNKNOWN"
        else:
            return right_speaker if right_speaker else "UNKNOWN"
    except Exception as e:
        print(f"Error determining closest speaker: {e}")
        return "UNKNOWN"

def update_speaker_labels(full_transcription, vad_left, vad_right):
    """
    Updates the transcription with speaker labels and breaks segments based on speaker change
    or large time gaps (> 2 seconds), optimized with binary search on VAD intervals.
    Each word in the segment retains its own speaker label.
    """
    updated_transcription = []
    current_segment = []
    current_speaker = None

    try:
        for idx, segment in enumerate(full_transcription):
            words = segment['words']

            for word_idx, word_info in enumerate(words):
                word_start = word_info['start']
                word_end = word_info['end']
                word_text = word_info['word']
                
                # Assign speaker to the current word based on the VAD results using binary search
                word_speaker = assign_speaker_to_word(word_start, word_end, vad_left, vad_right)
                
                # Add speaker information to the word
                word_info['speaker'] = word_speaker
                
                # If the current segment is empty, start a new one
                if not current_segment:
                    current_segment.append(word_info)
                    current_speaker = word_speaker
                    continue

                # Calculate the time gap between this word and the last word in the current segment
                last_word_end = current_segment[-1]['end']
                time_gap = word_start - last_word_end

                # Check for segment break conditions (speaker change or time gap)
                if time_gap > 2.0 or word_speaker != current_speaker:
                    # Finalize the current segment
                    updated_transcription.append({
                        'start': current_segment[0]['start'],
                        'end': current_segment[-1]['end'],
                        'text': " ".join([word['word'] for word in current_segment]),
                        'words': current_segment,  # Retain the words array with speaker labels
                        'speaker': current_speaker
                    })
                    # Start a new segment
                    current_segment = [word_info]
                    current_speaker = word_speaker
                else:
                    # Continue adding to the current segment
                    current_segment.append(word_info)

        # Don't forget to add the last segment
        if current_segment:
            updated_transcription.append({
                'start': current_segment[0]['start'],
                'end': current_segment[-1]['end'],
                'text': " ".join([word['word'] for word in current_segment]),
                'words': current_segment,  # Retain the words array with speaker labels
                'speaker': current_speaker
            })

    except Exception as e:
        print(f"Error occurred: {e}")
    
    return updated_transcription
    
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
            logger.info(f'left path is {left_path}')
            left_path = denoise_audio(left_path)
            logger.info(f'denoise left path is {left_path}')
            right_path = denoise_audio(right_path)
            
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
                final_result['segments'] = process_segment_words(final_result['segments'])
                final_result['segments'] = remove_duplicate_segments_withspeaker_withtext(final_result['segments'])
            except Exception as e:
                logger.error("Error cleaning up segments: %s", e)
            
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
        result_return = {'transcription': [], 'requestID': request_id}
        if webhook_url:
            requests.post(webhook_url, json=result_return)

        result_file_path = f'/tmp/{request_id}.txt'
        with open(result_file_path, 'w') as result_file:
            json.dump(result_return, result_file)
            raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        torch.cuda.empty_cache()
        gc.collect()

        del audio_data
        del result
        del result_transcribe
        del diarize_result
        del final_result

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
