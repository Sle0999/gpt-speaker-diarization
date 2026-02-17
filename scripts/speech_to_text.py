import io
import logging
import os
import subprocess
import uuid

import auditok
from openai import OpenAI
import soundfile as sf

from .openai_decorator import retry_on_openai_errors
from .utils import get_project_root


class Whisper:
    """
    This class serves as a wrapper for the OpenAI Whisper API to facilitate chatbot responses.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini-transcribe",
        whisper_sample_rate: int = 16000,
        max_chunk_seconds: int = 30,
    ):
        """
        Initialize the Whisper chatbot instance.

        :param model_name: The name of the OpenAI Whisper model to use.
        :param whisper_sample_rate: The sample rate for audio processing.
        """
        self.model_name = model_name
        self.whisper_sample_rate = whisper_sample_rate
        self.max_chunk_seconds = int(os.getenv("MAX_CHUNK_SECONDS", max_chunk_seconds))
        self.client = OpenAI()

    def vad_audiotok(self, audio_content):
        """
        Perform voice activity detection using the audiotok package.

        :param audio_content: Bytes of audio data.
        :return: Chunks containing speech detected in the audio.
        """
        audio_regions = auditok.split(
            audio_content,
            sr=self.whisper_sample_rate,
            ch=1,
            sw=2,
            min_dur=0.5,
            max_dur=self.max_chunk_seconds,
            max_silence=0.3,
            energy_threshold=30
        )
        return audio_regions

    def _convert_to_wav_mono_16k(self, filepath: str) -> str:
        """
        Convert an audio file to WAV mono 16 kHz format.

        :param filepath: Path to the source audio file.
        :return: Path to the converted WAV file.
        """
        root_path = get_project_root()
        resources_path = f"{root_path}/resources/audios"
        temp_wav_path = f"{resources_path}/{str(uuid.uuid4())}.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                filepath,
                "-ar",
                str(self.whisper_sample_rate),
                "-ac",
                "1",
                "-y",
                temp_wav_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return temp_wav_path

    def audio_process(self, wav_path, is_byte=False):
        """
        Process audio data, performing voice activity detection and segmenting the audio.

        :param wav_path: Path to the audio file or audio bytes.
        :param is_byte: Boolean flag indicating if the input is audio bytes.
        :return: Segmented audio chunks containing detected speech.
        """
        if not is_byte:
            with open(wav_path, 'rb') as f:
                wav_bytes = f.read()
            wav, sr = sf.read(wav_path)
        else:
            wav_bytes = wav_path
            wav, sr = sf.read(io.BytesIO(wav_bytes))
        audio_regions = self.vad_audiotok(wav_bytes)
        wav_segments = []
        for r in audio_regions:
            start = r.meta.start
            end = r.meta.end
            segment = wav[int(start * sr):int(end * sr)]
            wav_segments.append(segment)
        return wav_segments

    @retry_on_openai_errors(max_retry=7)
    def transcribe(self, audio_file):
        """
        Transcribe the provided audio using the OpenAI API.

        :param audio_file: Path to the audio file or audio bytes.
        :return: Transcription text from the audio.
        """
        root_path = get_project_root()
        resources_path = f"{root_path}/resources/audios"
        temp_wav_path = f"{resources_path}/{str(uuid.uuid4())}.wav"
        with sf.SoundFile(
            temp_wav_path,
            'wb',
            samplerate=self.whisper_sample_rate,
            channels=1,
        ) as f:
            f.write(audio_file)

        with open(temp_wav_path, 'rb') as audio_stream:
            response = self.client.audio.transcriptions.create(
                model=self.model_name,
                file=audio_stream,
            )
        os.remove(temp_wav_path)
        return response.text

    @retry_on_openai_errors(max_retry=7)
    def transcribe_raw(self, audio_file):
        """
        Transcribe the provided audio using the OpenAI API without saving a temporary file.

        :param audio_file: Path to the audio file or audio bytes.
        :return: Transcription text from the audio.
        """
        with open(audio_file, 'rb') as audio_stream:
            response = self.client.audio.transcriptions.create(
                model=self.model_name,
                file=audio_stream,
            )
        return response.text

    def transcribe_chunked(self, filepath: str) -> str:
        """
        Transcribe an audio file in chunks and return a merged transcript.

        Falls back to a single transcription call if chunking fails.

        :param filepath: Path to the source audio file.
        :return: Full transcript text.
        """
        temp_wav_path = None
        try:
            temp_wav_path = self._convert_to_wav_mono_16k(filepath)
            wav_segments = self.audio_process(temp_wav_path)

            if not wav_segments:
                logging.warning("No audio chunks detected, falling back to single transcription call.")
                return self.transcribe_raw(filepath)

            logging.info(
                "Starting chunked transcription with %s chunks (max chunk seconds: %s).",
                len(wav_segments),
                self.max_chunk_seconds,
            )
            transcript_chunks = []
            for index, segment in enumerate(wav_segments, start=1):
                logging.info("Transcribing chunk %s/%s.", index, len(wav_segments))
                transcript_chunks.append(self.transcribe(segment))

            return " ".join(chunk.strip() for chunk in transcript_chunks if chunk).strip()
        except Exception as e:
            logging.warning(
                "Chunked transcription failed (%s). Falling back to single transcription call.",
                e,
            )
            return self.transcribe_raw(filepath)
        finally:
            if temp_wav_path and os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)


if __name__ == "__main__":
    # Example usage
    wh = Whisper()
    with open("./audios/0_edited.wav", "rb") as f:
        audio_content = f.read()
    print(type(audio_content))
    segments = wh.audio_process("./audios/0_edited.wav")
