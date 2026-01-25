"""
Streaming Speech-to-Text Service for the AI Meeting Assistant.

Supports multiple STT providers:
- Deepgram (recommended for lowest latency)
- Google Cloud Speech-to-Text
- Azure Speech Services

All providers stream results in real-time with speaker diarization when available.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Optional, AsyncIterator
import structlog

from src.config import config, STTProvider
from src.context.models import TranscriptChunk

logger = structlog.get_logger()


class BaseSTTService(ABC):
    """Abstract base class for speech-to-text services."""
    
    def __init__(self, on_transcript: Optional[Callable[[TranscriptChunk], None]] = None):
        self.on_transcript = on_transcript
        self._running = False
    
    @abstractmethod
    async def start(self):
        """Start the STT service."""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop the STT service."""
        pass
    
    @abstractmethod
    async def process_audio(self, audio_data: bytes):
        """Process an audio chunk."""
        pass
    
    def _emit_transcript(self, chunk: TranscriptChunk):
        """Emit a transcript chunk to the callback."""
        if self.on_transcript:
            self.on_transcript(chunk)
        logger.debug(
            "transcript_emitted",
            text=chunk.text[:50],
            speaker=chunk.speaker,
            is_final=chunk.is_final
        )


class DeepgramSTTService(BaseSTTService):
    """
    Deepgram streaming speech-to-text service.
    
    Features:
    - ~200ms latency
    - Real-time streaming via WebSocket
    - Speaker diarization
    - Punctuation and formatting
    """
    
    def __init__(self, on_transcript: Optional[Callable[[TranscriptChunk], None]] = None):
        super().__init__(on_transcript)
        self._client = None
        self._connection = None
        self._api_key = config.stt.deepgram_api_key
    
    async def start(self):
        """Start the Deepgram streaming connection."""
        if not self._api_key:
            logger.error("deepgram_api_key_not_set")
            return
        
        try:
            from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
            
            self._client = DeepgramClient(self._api_key)
            
            # Configure live transcription options
            options = LiveOptions(
                model="nova-2",
                language="en",
                smart_format=True,
                punctuate=True,
                diarize=True,
                interim_results=True,
                utterance_end_ms=1000,
                vad_events=True,
            )
            
            self._connection = self._client.listen.asynclive.v("1")
            
            # Set up event handlers
            self._connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
            self._connection.on(LiveTranscriptionEvents.Error, self._on_error)
            
            # Start the connection
            await self._connection.start(options)
            self._running = True
            
            logger.info("deepgram_started")
            
        except ImportError:
            logger.error("deepgram_sdk_not_installed")
        except Exception as e:
            logger.error("deepgram_start_failed", error=str(e))
    
    async def stop(self):
        """Stop the Deepgram connection."""
        self._running = False
        if self._connection:
            await self._connection.finish()
        logger.info("deepgram_stopped")
    
    async def process_audio(self, audio_data: bytes):
        """Send audio data to Deepgram for transcription."""
        if self._connection and self._running:
            await self._connection.send(audio_data)
    
    def _on_transcript(self, *args, **kwargs):
        """Handle transcript events from Deepgram."""
        # Deepgram SDK passes result as first positional arg or keyword
        result = args[1] if len(args) > 1 else kwargs.get("result")
        
        if not result or not result.channel:
            return
        
        for alt in result.channel.alternatives:
            if not alt.transcript:
                continue
            
            # Get speaker label if available
            speaker = None
            if alt.words and len(alt.words) > 0:
                first_word = alt.words[0]
                if hasattr(first_word, 'speaker'):
                    speaker = f"Speaker {first_word.speaker}"
            
            chunk = TranscriptChunk(
                text=alt.transcript,
                timestamp=datetime.now(),
                speaker=speaker,
                is_final=result.is_final,
                confidence=alt.confidence if hasattr(alt, 'confidence') else 1.0
            )
            
            self._emit_transcript(chunk)
    
    def _on_error(self, *args, **kwargs):
        """Handle error events from Deepgram."""
        error = args[1] if len(args) > 1 else kwargs.get("error")
        logger.error("deepgram_error", error=str(error))


class GoogleSTTService(BaseSTTService):
    """
    Google Cloud Speech-to-Text streaming service.
    
    Features:
    - Real-time streaming
    - Multiple language support
    - Speaker diarization
    """
    
    def __init__(self, on_transcript: Optional[Callable[[TranscriptChunk], None]] = None):
        super().__init__(on_transcript)
        self._client = None
        self._streaming_config = None
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._process_task = None
    
    async def start(self):
        """Start the Google STT service."""
        try:
            from google.cloud import speech
            
            self._client = speech.SpeechAsyncClient()
            
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US",
                enable_automatic_punctuation=True,
                enable_speaker_diarization=True,
                diarization_speaker_count=4,
            )
            
            self._streaming_config = speech.StreamingRecognitionConfig(
                config=config,
                interim_results=True,
            )
            
            self._running = True
            self._process_task = asyncio.create_task(self._process_stream())
            
            logger.info("google_stt_started")
            
        except ImportError:
            logger.error("google_cloud_speech_not_installed")
        except Exception as e:
            logger.error("google_stt_start_failed", error=str(e))
    
    async def stop(self):
        """Stop the Google STT service."""
        self._running = False
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
        logger.info("google_stt_stopped")
    
    async def process_audio(self, audio_data: bytes):
        """Queue audio data for processing."""
        await self._audio_queue.put(audio_data)
    
    async def _process_stream(self):
        """Process audio stream and emit transcripts."""
        from google.cloud import speech
        
        async def audio_generator():
            # First request with config
            yield speech.StreamingRecognizeRequest(
                streaming_config=self._streaming_config
            )
            
            # Subsequent requests with audio
            while self._running:
                try:
                    audio_data = await asyncio.wait_for(
                        self._audio_queue.get(), 
                        timeout=0.1
                    )
                    yield speech.StreamingRecognizeRequest(audio_content=audio_data)
                except asyncio.TimeoutError:
                    continue
        
        try:
            responses = await self._client.streaming_recognize(
                requests=audio_generator()
            )
            
            async for response in responses:
                for result in response.results:
                    if not result.alternatives:
                        continue
                    
                    alt = result.alternatives[0]
                    
                    # Get speaker label
                    speaker = None
                    if result.result_end_time:
                        # Google returns speaker info in word_info
                        for word in alt.words:
                            if word.speaker_tag:
                                speaker = f"Speaker {word.speaker_tag}"
                                break
                    
                    chunk = TranscriptChunk(
                        text=alt.transcript,
                        timestamp=datetime.now(),
                        speaker=speaker,
                        is_final=result.is_final,
                        confidence=alt.confidence
                    )
                    
                    self._emit_transcript(chunk)
                    
        except Exception as e:
            logger.error("google_stt_stream_error", error=str(e))


class AzureSTTService(BaseSTTService):
    """
    Azure Speech Services streaming transcription.
    
    Features:
    - Real-time streaming
    - Speaker recognition
    - Custom speech models
    """
    
    def __init__(self, on_transcript: Optional[Callable[[TranscriptChunk], None]] = None):
        super().__init__(on_transcript)
        self._speech_recognizer = None
        self._audio_stream = None
    
    async def start(self):
        """Start the Azure STT service."""
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            speech_config = speechsdk.SpeechConfig(
                subscription=config.stt.azure_speech_key,
                region=config.stt.azure_speech_region
            )
            speech_config.speech_recognition_language = "en-US"
            speech_config.enable_dictation()
            
            # Create push stream for audio input
            self._audio_stream = speechsdk.audio.PushAudioInputStream()
            audio_config = speechsdk.audio.AudioConfig(stream=self._audio_stream)
            
            self._speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config
            )
            
            # Set up event handlers
            self._speech_recognizer.recognizing.connect(self._on_recognizing)
            self._speech_recognizer.recognized.connect(self._on_recognized)
            self._speech_recognizer.canceled.connect(self._on_canceled)
            
            # Start continuous recognition
            self._speech_recognizer.start_continuous_recognition_async()
            self._running = True
            
            logger.info("azure_stt_started")
            
        except ImportError:
            logger.error("azure_speech_sdk_not_installed")
        except Exception as e:
            logger.error("azure_stt_start_failed", error=str(e))
    
    async def stop(self):
        """Stop the Azure STT service."""
        self._running = False
        if self._speech_recognizer:
            self._speech_recognizer.stop_continuous_recognition_async()
        if self._audio_stream:
            self._audio_stream.close()
        logger.info("azure_stt_stopped")
    
    async def process_audio(self, audio_data: bytes):
        """Push audio data to the Azure stream."""
        if self._audio_stream and self._running:
            self._audio_stream.write(audio_data)
    
    def _on_recognizing(self, evt):
        """Handle interim results."""
        chunk = TranscriptChunk(
            text=evt.result.text,
            timestamp=datetime.now(),
            speaker=None,  # Azure requires separate speaker recognition
            is_final=False,
            confidence=1.0
        )
        self._emit_transcript(chunk)
    
    def _on_recognized(self, evt):
        """Handle final results."""
        if evt.result.text:
            chunk = TranscriptChunk(
                text=evt.result.text,
                timestamp=datetime.now(),
                speaker=None,
                is_final=True,
                confidence=1.0
            )
            self._emit_transcript(chunk)
    
    def _on_canceled(self, evt):
        """Handle cancellation/errors."""
        logger.error("azure_stt_canceled", reason=evt.reason)


class MockSTTService(BaseSTTService):
    """
    Mock STT service for testing without real audio.
    
    Accepts text input and emits it as transcript chunks.
    """
    
    def __init__(self, on_transcript: Optional[Callable[[TranscriptChunk], None]] = None):
        super().__init__(on_transcript)
    
    async def start(self):
        self._running = True
        logger.info("mock_stt_started")
    
    async def stop(self):
        self._running = False
        logger.info("mock_stt_stopped")
    
    async def process_audio(self, audio_data: bytes):
        """No-op for mock service."""
        pass
    
    async def inject_text(
        self, 
        text: str, 
        speaker: Optional[str] = None,
        is_final: bool = True
    ):
        """Inject text as if it was transcribed."""
        chunk = TranscriptChunk(
            text=text,
            timestamp=datetime.now(),
            speaker=speaker,
            is_final=is_final,
            confidence=1.0
        )
        self._emit_transcript(chunk)


def create_stt_service(
    on_transcript: Optional[Callable[[TranscriptChunk], None]] = None
) -> BaseSTTService:
    """
    Factory function to create the appropriate STT service based on config.
    
    Args:
        on_transcript: Callback for transcript chunks
    
    Returns:
        Configured STT service instance
    """
    provider = config.stt.provider
    
    if provider == STTProvider.DEEPGRAM:
        return DeepgramSTTService(on_transcript)
    elif provider == STTProvider.GOOGLE:
        return GoogleSTTService(on_transcript)
    elif provider == STTProvider.AZURE:
        return AzureSTTService(on_transcript)
    else:
        logger.warning("unknown_stt_provider", provider=provider, using="mock")
        return MockSTTService(on_transcript)
