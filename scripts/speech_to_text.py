import io
import os
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
    ):
        """
        Initialize the Whisper chatbot instance.

        :param model_name: The name of the OpenAI Whisper model to use.
        :param whisper_sample_rate: The sample rate for audio processing.
        """
        self.model_name = model_name
        self.whisper_sample_rate = whisper_sample_rate
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
            max_dur=30,
            max_silence=0.3,
            energy_threshold=30
        )
        return audio_regions

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


if __name__ == "__main__":
    # Example usage
    wh = Whisper()
    with open("./audios/0_edited.wav", "rb") as f:
        audio_content = f.read()
    print(type(audio_content))
    segments = wh.audio_process("./audios/0_edited.wav")
