import importlib
import sys
import types
import unittest
from unittest.mock import patch

from fastapi import HTTPException


# Provide lightweight fakes for app dependencies so tests run offline.
fake_speech_to_text = types.ModuleType("scripts.speech_to_text")


class _FakeWhisper:
    def __init__(self):
        self.default_chunk_seconds = 120

    @staticmethod
    def _clamp_chunk_seconds(value):
        return max(10, min(600, int(value)))

    def transcribe_single_pass(self, _path):
        return ""

    def transcribe_chunked(self, _path, chunk_seconds=None):
        return ""


fake_speech_to_text.Whisper = _FakeWhisper
sys.modules["scripts.speech_to_text"] = fake_speech_to_text

fake_text_analysis = types.ModuleType("scripts.text_analysis")


class _FakeAI:
    def extract_dialogue(self, _text):
        return []


fake_text_analysis.AI = _FakeAI
sys.modules["scripts.text_analysis"] = fake_text_analysis

fake_video_manager = types.ModuleType("scripts.video_manager")


class _FakeVideoDownloader:
    def download_video(self, _video_id):
        return "/tmp/video.wav"


fake_video_manager.VideoDownloader = _FakeVideoDownloader
sys.modules["scripts.video_manager"] = fake_video_manager

fake_utils = types.ModuleType("scripts.utils")
fake_utils.write_audio = lambda _data, _filename: "/tmp/audio.wav"
sys.modules["scripts.utils"] = fake_utils

app = importlib.import_module("scripts.app")


class SpeakerDiarizationChunkingTests(unittest.TestCase):
    @patch("scripts.app.write_audio", return_value="/tmp/audio.wav")
    @patch("scripts.app.openai_services.extract_dialogue", return_value=[{"speaker": "A", "text": "hello"}])
    @patch("scripts.app.whisper_api.transcribe_chunked")
    @patch("scripts.app.whisper_api.transcribe_single_pass", return_value="full transcript")
    def test_process_audio_uses_single_pass_when_chunk_seconds_zero(
        self,
        mock_single_pass,
        mock_chunked,
        _mock_dialogue,
        _mock_write_audio,
    ):
        response = app.process_audio(b"audio", "clip.wav", chunk_seconds=0)

        mock_single_pass.assert_called_once_with("/tmp/audio.wav")
        mock_chunked.assert_not_called()
        self.assertEqual(response["transcript"], "full transcript")
        self.assertIn("diarization_result", response)

    @patch("scripts.app.write_audio", return_value="/tmp/audio.wav")
    @patch("scripts.app.openai_services.extract_dialogue", return_value=[{"speaker": "A", "text": "hello"}])
    @patch("scripts.app.whisper_api.transcribe_chunked", return_value="chunked transcript")
    @patch("scripts.app.whisper_api.transcribe_single_pass")
    def test_process_audio_uses_chunked_when_chunk_seconds_positive(
        self,
        mock_single_pass,
        mock_chunked,
        _mock_dialogue,
        _mock_write_audio,
    ):
        response = app.process_audio(b"audio", "clip.wav", chunk_seconds=120)

        mock_chunked.assert_called_once_with("/tmp/audio.wav", chunk_seconds=120)
        mock_single_pass.assert_not_called()
        self.assertEqual(response["transcript"], "chunked transcript")
        self.assertIn("diarization_result", response)

    def test_parse_chunk_seconds_rejects_non_integer(self):
        with self.assertRaises(HTTPException) as ctx:
            app.parse_chunk_seconds("not-an-int")

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("must be an integer", ctx.exception.detail)


if __name__ == "__main__":
    unittest.main()
