#!/usr/bin/env python3
"""
Voice UI Orchestrator - Voice and Audio Interaction System
Enables voice chat, audio responses, and multi-modal interaction with the agentic system
Supports speech-to-text, text-to-speech, and conversational AI
"""

import asyncio
import json
import yaml
import logging
import subprocess
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict
import aiofiles
import aiohttp
from datetime import datetime, timedelta
import websockets
import uuid
import wave
import pyaudio
import threading
import queue
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceConfiguration(BaseModel):
    """Configuration for voice interaction"""
    speech_to_text_provider: str = "openai_whisper"
    text_to_speech_provider: str = "elevenlabs"
    voice_id: str = "default"
    language: str = "en-US"
    sample_rate: int = 16000
    chunk_size: int = 1024
    audio_format: int = pyaudio.paInt16
    channels: int = 1
    wake_word: str = "hey assistant"
    continuous_listening: bool = True
    noise_reduction: bool = True
    voice_activity_detection: bool = True

class VoiceMessage(BaseModel):
    """Voice message data structure"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    audio_data: Optional[bytes] = None
    transcribed_text: Optional[str] = None
    response_text: Optional[str] = None
    response_audio: Optional[bytes] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time: float = 0.0
    confidence_score: float = 0.0

class VoiceUIOrchestrator:
    """
    Comprehensive voice and audio interaction orchestrator
    """
    
    def __init__(self):
        self.base_dir = Path("foundation_data")
        self.voice_dir = self.base_dir / "voice_ui"
        self.audio_cache_dir = self.voice_dir / "audio_cache"
        self.voice_models_dir = self.voice_dir / "models"
        
        # Voice configuration
        self.voice_config = VoiceConfiguration()
        
        # Audio components
        self.audio_interface = None
        self.speech_recognizer = None
        self.text_to_speech = None
        self.wake_word_detector = None
        
        # Processing queues
        self.audio_input_queue = asyncio.Queue()
        self.speech_processing_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        
        # State management
        self.is_listening = False
        self.is_speaking = False
        self.conversation_context = []
        self.active_sessions = {}\n        
        # WebSocket connections for real-time audio
        self.websocket_connections = {}\n        
        self._initialize_directories()
        
    def _initialize_directories(self):
        """Initialize voice UI directories"""
        directories = [
            self.voice_dir,
            self.audio_cache_dir,
            self.voice_models_dir,
            self.voice_dir / "recordings",
            self.voice_dir / "responses",
            self.voice_dir / "training_data"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    async def initialize(self):
        """Initialize the voice UI orchestrator"""
        logger.info("🎤 Initializing Voice UI Orchestrator...")
        
        # Setup audio interface
        await self._setup_audio_interface()
        
        # Initialize speech recognition
        await self._initialize_speech_recognition()
        
        # Initialize text-to-speech
        await self._initialize_text_to_speech()
        
        # Setup wake word detection
        await self._setup_wake_word_detection()
        
        # Start processing loops
        await self._start_processing_loops()
        
        logger.info("✅ Voice UI Orchestrator initialized")
        
    async def _setup_audio_interface(self):
        """Setup PyAudio interface"""
        try:
            # Install required packages
            subprocess.run(["pip3", "install", "pyaudio", "wave", "numpy"], check=True, timeout=120)
            
            import pyaudio
            self.audio_interface = pyaudio.PyAudio()
            
            # List available audio devices
            logger.info("Available audio devices:")
            for i in range(self.audio_interface.get_device_count()):
                info = self.audio_interface.get_device_info_by_index(i)
                logger.info(f"  Device {i}: {info['name']} - {info['maxInputChannels']} input channels")
                
        except Exception as e:
            logger.error(f"Failed to setup audio interface: {e}")
            
    async def _initialize_speech_recognition(self):
        """Initialize speech recognition components"""
        try:
            if self.voice_config.speech_to_text_provider == "openai_whisper":
                await self._setup_whisper()
            elif self.voice_config.speech_to_text_provider == "google_speech":
                await self._setup_google_speech()
            else:
                await self._setup_whisper()  # Default to Whisper
                
        except Exception as e:
            logger.error(f"Failed to initialize speech recognition: {e}")
            
    async def _setup_whisper(self):
        """Setup OpenAI Whisper for speech recognition"""
        try:
            subprocess.run(["pip3", "install", "openai-whisper"], check=True, timeout=180)
            import whisper
            
            # Load Whisper model
            model_size = "base"  # Options: tiny, base, small, medium, large
            self.speech_recognizer = whisper.load_model(model_size)
            
            logger.info(f"✅ Whisper model '{model_size}' loaded")
            
        except Exception as e:
            logger.error(f"Failed to setup Whisper: {e}")
            
    async def _setup_google_speech(self):
        """Setup Google Speech Recognition"""
        try:
            subprocess.run(["pip3", "install", "google-cloud-speech"], check=True, timeout=120)
            from google.cloud import speech
            
            self.speech_recognizer = speech.SpeechClient()
            logger.info("✅ Google Speech Recognition initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup Google Speech: {e}")
            
    async def _initialize_text_to_speech(self):
        """Initialize text-to-speech components"""
        try:
            if self.voice_config.text_to_speech_provider == "elevenlabs":
                await self._setup_elevenlabs()
            elif self.voice_config.text_to_speech_provider == "google_tts":
                await self._setup_google_tts()
            elif self.voice_config.text_to_speech_provider == "azure_tts":
                await self._setup_azure_tts()
            else:
                await self._setup_system_tts()  # Fallback to system TTS
                
        except Exception as e:
            logger.error(f"Failed to initialize text-to-speech: {e}")
            
    async def _setup_elevenlabs(self):
        """Setup ElevenLabs TTS"""
        try:
            subprocess.run(["pip3", "install", "elevenlabs"], check=True, timeout=120)
            
            # Configure ElevenLabs
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if api_key:
                from elevenlabs import set_api_key, voices, generate
                set_api_key(api_key)
                
                # Get available voices
                voice_list = voices()
                logger.info(f"✅ ElevenLabs TTS initialized with {len(voice_list)} voices")
                
                self.text_to_speech = {
                    "provider": "elevenlabs",
                    "voices": voice_list,
                    "generate_func": generate
                }
            else:
                logger.warning("ElevenLabs API key not found, falling back to system TTS")
                await self._setup_system_tts()
                
        except Exception as e:
            logger.error(f"Failed to setup ElevenLabs: {e}")
            await self._setup_system_tts()
            
    async def _setup_google_tts(self):
        """Setup Google Cloud TTS"""
        try:
            subprocess.run(["pip3", "install", "google-cloud-texttospeech"], check=True, timeout=120)
            from google.cloud import texttospeech
            
            self.text_to_speech = {
                "provider": "google_tts",
                "client": texttospeech.TextToSpeechClient()
            }
            
            logger.info("✅ Google Cloud TTS initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup Google TTS: {e}")
            await self._setup_system_tts()
            
    async def _setup_azure_tts(self):
        """Setup Azure Cognitive Services TTS"""
        try:
            subprocess.run(["pip3", "install", "azure-cognitiveservices-speech"], check=True, timeout=120)
            import azure.cognitiveservices.speech as speechsdk
            
            speech_key = os.getenv("AZURE_SPEECH_KEY")
            service_region = os.getenv("AZURE_SPEECH_REGION", "westus")
            
            if speech_key:
                speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
                self.text_to_speech = {
                    "provider": "azure_tts",
                    "config": speech_config
                }
                
                logger.info("✅ Azure TTS initialized")
            else:
                logger.warning("Azure Speech key not found, falling back to system TTS")
                await self._setup_system_tts()
                
        except Exception as e:
            logger.error(f"Failed to setup Azure TTS: {e}")
            await self._setup_system_tts()
            
    async def _setup_system_tts(self):
        """Setup system-level TTS as fallback"""
        try:
            subprocess.run(["pip3", "install", "pyttsx3"], check=True, timeout=120)
            import pyttsx3
            
            engine = pyttsx3.init()
            
            # Configure voice properties
            voices = engine.getProperty('voices')
            if voices:
                engine.setProperty('voice', voices[0].id)
                
            engine.setProperty('rate', 150)  # Speed of speech
            engine.setProperty('volume', 0.9)  # Volume level
            
            self.text_to_speech = {
                "provider": "system_tts",
                "engine": engine
            }
            
            logger.info("✅ System TTS initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup system TTS: {e}")
            
    async def _setup_wake_word_detection(self):
        """Setup wake word detection"""
        try:
            subprocess.run(["pip3", "install", "pvporcupine"], check=True, timeout=120)
            
            # Configure wake word detection
            self.wake_word_detector = {
                "enabled": True,
                "sensitivity": 0.5,
                "keywords": [self.voice_config.wake_word]
            }
            
            logger.info(f"✅ Wake word detection setup for '{self.voice_config.wake_word}'")
            
        except Exception as e:
            logger.error(f"Failed to setup wake word detection: {e}")
            self.wake_word_detector = {"enabled": False}
            
    async def _start_processing_loops(self):
        """Start all voice processing loops"""
        logger.info("🔄 Starting voice processing loops...")
        
        # Start audio input loop
        asyncio.create_task(self._audio_input_loop())
        
        # Start speech processing loop
        asyncio.create_task(self._speech_processing_loop())
        
        # Start response generation loop
        asyncio.create_task(self._response_generation_loop())
        
        # Start continuous listening if enabled
        if self.voice_config.continuous_listening:
            asyncio.create_task(self._continuous_listening_loop())
            
    async def _audio_input_loop(self):
        """Continuous audio input processing"""
        while True:
            try:
                if self.is_listening and not self.is_speaking:
                    # Record audio chunk
                    audio_chunk = await self._record_audio_chunk()
                    
                    if audio_chunk:
                        # Add to processing queue
                        await self.audio_input_queue.put({
                            "audio_data": audio_chunk,
                            "timestamp": datetime.now(),
                            "user_id": "default"
                        })
                        
                await asyncio.sleep(0.1)  # Small delay to prevent overwhelming
                
            except Exception as e:
                logger.error(f"Error in audio input loop: {e}")
                await asyncio.sleep(1)
                
    async def _record_audio_chunk(self) -> Optional[bytes]:
        """Record a chunk of audio"""
        try:
            if not self.audio_interface:
                return None
                
            stream = self.audio_interface.open(
                format=self.voice_config.audio_format,
                channels=self.voice_config.channels,
                rate=self.voice_config.sample_rate,
                input=True,
                frames_per_buffer=self.voice_config.chunk_size
            )
            
            # Record for 1 second
            frames = []
            for _ in range(0, int(self.voice_config.sample_rate / self.voice_config.chunk_size)):
                data = stream.read(self.voice_config.chunk_size)
                frames.append(data)
                
            stream.stop_stream()
            stream.close()
            
            # Convert to bytes
            audio_data = b''.join(frames)
            return audio_data
            
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            return None
            
    async def _speech_processing_loop(self):
        """Process speech from audio input"""
        while True:
            try:
                # Get audio from queue
                if not self.audio_input_queue.empty():
                    audio_data = await self.audio_input_queue.get()
                    
                    # Process speech to text
                    transcription = await self._transcribe_audio(audio_data["audio_data"])
                    
                    if transcription and transcription.strip():
                        # Create voice message
                        voice_message = VoiceMessage(
                            user_id=audio_data["user_id"],
                            audio_data=audio_data["audio_data"],
                            transcribed_text=transcription,
                            timestamp=audio_data["timestamp"]
                        )
                        
                        # Add to speech processing queue
                        await self.speech_processing_queue.put(voice_message)
                        
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in speech processing loop: {e}")
                await asyncio.sleep(1)
                
    async def _transcribe_audio(self, audio_data: bytes) -> Optional[str]:
        """Transcribe audio to text"""
        try:
            if not self.speech_recognizer:
                return None
                
            # Save audio to temporary file
            temp_audio_file = self.audio_cache_dir / f"temp_{uuid.uuid4()}.wav"
            
            # Convert bytes to WAV file
            with wave.open(str(temp_audio_file), 'wb') as wav_file:
                wav_file.setnchannels(self.voice_config.channels)
                wav_file.setsampwidth(self.audio_interface.get_sample_size(self.voice_config.audio_format))
                wav_file.setframerate(self.voice_config.sample_rate)
                wav_file.writeframes(audio_data)
                
            if self.voice_config.speech_to_text_provider == "openai_whisper":
                # Use Whisper for transcription
                result = self.speech_recognizer.transcribe(str(temp_audio_file))
                transcription = result["text"]
            else:
                # Use other provider
                transcription = "Transcription not implemented for this provider"
                
            # Clean up temp file
            temp_audio_file.unlink()
            
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None
            
    async def _response_generation_loop(self):
        """Generate responses to processed speech"""
        while True:
            try:
                # Get processed speech from queue
                if not self.speech_processing_queue.empty():
                    voice_message = await self.speech_processing_queue.get()
                    
                    # Generate text response
                    response_text = await self._generate_text_response(voice_message.transcribed_text)
                    
                    if response_text:
                        # Convert to audio
                        response_audio = await self._generate_audio_response(response_text)
                        
                        # Update voice message
                        voice_message.response_text = response_text
                        voice_message.response_audio = response_audio
                        
                        # Add to response queue
                        await self.response_queue.put(voice_message)
                        
                        # Play audio response
                        await self._play_audio_response(response_audio)
                        
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in response generation loop: {e}")
                await asyncio.sleep(1)
                
    async def _generate_text_response(self, transcribed_text: str) -> str:
        """Generate text response using AI"""
        try:
            # This would integrate with the main orchestrator's LLM capabilities
            # For now, return a simple response
            
            # Add to conversation context
            self.conversation_context.append({
                "role": "user",
                "content": transcribed_text,
                "timestamp": datetime.now()
            })
            
            # Generate response based on context
            if "hello" in transcribed_text.lower():
                response = "Hello! How can I assist you today?"
            elif "status" in transcribed_text.lower():
                response = "All systems are running normally. The orchestrator is active and ready to help."
            elif "projects" in transcribed_text.lower():
                response = "I can help you manage projects. Would you like to see active projects or create a new one?"
            else:
                response = f"I heard you say: {transcribed_text}. How can I help you with that?"
                
            # Add response to context
            self.conversation_context.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now()
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating text response: {e}")
            return "I'm sorry, I couldn't process that request."
            
    async def _generate_audio_response(self, text: str) -> Optional[bytes]:
        """Generate audio from text response"""
        try:
            if not self.text_to_speech:
                return None
                
            provider = self.text_to_speech["provider"]
            
            if provider == "elevenlabs":
                from elevenlabs import generate
                
                audio = generate(
                    text=text,
                    voice=self.voice_config.voice_id,
                    model="eleven_monolingual_v1"
                )
                return audio
                
            elif provider == "google_tts":
                from google.cloud import texttospeech
                
                synthesis_input = texttospeech.SynthesisInput(text=text)
                voice = texttospeech.VoiceSelectionParams(
                    language_code=self.voice_config.language,
                    ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3
                )
                
                response = self.text_to_speech["client"].synthesize_speech(
                    input=synthesis_input, voice=voice, audio_config=audio_config
                )
                
                return response.audio_content
                
            elif provider == "system_tts":
                # Save to file and read back
                temp_file = self.audio_cache_dir / f"response_{uuid.uuid4()}.wav"
                
                engine = self.text_to_speech["engine"]
                engine.save_to_file(text, str(temp_file))
                engine.runAndWait()
                
                # Read file content
                with open(temp_file, 'rb') as f:
                    audio_data = f.read()
                    
                # Clean up
                temp_file.unlink()
                
                return audio_data
                
        except Exception as e:
            logger.error(f"Error generating audio response: {e}")
            return None
            
    async def _play_audio_response(self, audio_data: Optional[bytes]):
        """Play audio response"""
        try:
            if not audio_data or not self.audio_interface:
                return
                
            self.is_speaking = True
            
            # Save audio to temporary file
            temp_file = self.audio_cache_dir / f"play_{uuid.uuid4()}.wav"
            
            with open(temp_file, 'wb') as f:
                f.write(audio_data)
                
            # Play using system command (platform-specific)
            if sys.platform == "darwin":  # macOS
                subprocess.run(["afplay", str(temp_file)], check=True)
            elif sys.platform.startswith("linux"):
                subprocess.run(["aplay", str(temp_file)], check=True)
            elif sys.platform.startswith("win"):
                subprocess.run(["start", str(temp_file)], shell=True, check=True)
                
            # Clean up
            temp_file.unlink()
            
            self.is_speaking = False
            
        except Exception as e:
            logger.error(f"Error playing audio response: {e}")
            self.is_speaking = False
            
    async def _continuous_listening_loop(self):
        """Continuous listening with wake word detection"""
        while True:
            try:
                if not self.is_speaking:
                    # Check for wake word or start listening
                    if self.wake_word_detector["enabled"]:
                        # Implement wake word detection
                        wake_detected = await self._detect_wake_word()
                        if wake_detected:
                            self.is_listening = True
                            logger.info("Wake word detected, starting to listen...")
                    else:
                        # Always listening mode
                        self.is_listening = True
                        
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in continuous listening loop: {e}")
                await asyncio.sleep(2)
                
    async def _detect_wake_word(self) -> bool:
        """Detect wake word in audio stream"""
        try:
            # This is a simplified implementation
            # In a real system, you'd use a proper wake word detection library
            
            # Record a short audio clip
            audio_chunk = await self._record_audio_chunk()
            
            if audio_chunk:
                # Transcribe and check for wake word
                transcription = await self._transcribe_audio(audio_chunk)
                
                if transcription and self.voice_config.wake_word.lower() in transcription.lower():
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error detecting wake word: {e}")
            return False
            
    async def start_voice_session(self, user_id: str = "default") -> str:
        """Start a new voice interaction session"""
        session_id = str(uuid.uuid4())
        
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "start_time": datetime.now(),
            "messages": [],
            "is_active": True
        }
        
        logger.info(f"Started voice session {session_id} for user {user_id}")
        
        # Start listening
        self.is_listening = True
        
        return session_id
        
    async def stop_voice_session(self, session_id: str):
        """Stop a voice interaction session"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["is_active"] = False
            self.active_sessions[session_id]["end_time"] = datetime.now()
            
            logger.info(f"Stopped voice session {session_id}")
            
        # Stop listening if no active sessions
        active_sessions = [s for s in self.active_sessions.values() if s["is_active"]]
        if not active_sessions:
            self.is_listening = False
            
    async def process_voice_command(self, command: str, user_id: str = "default") -> Dict[str, Any]:
        """Process a voice command and return response"""
        try:
            # Generate response
            response_text = await self._generate_text_response(command)
            
            # Generate audio
            response_audio = await self._generate_audio_response(response_text)
            
            # Create voice message
            voice_message = VoiceMessage(
                user_id=user_id,
                transcribed_text=command,
                response_text=response_text,
                response_audio=response_audio
            )
            
            # Play response
            await self._play_audio_response(response_audio)
            
            return {
                "message_id": voice_message.id,
                "transcribed_text": command,
                "response_text": response_text,
                "has_audio": response_audio is not None,
                "timestamp": voice_message.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing voice command: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    async def get_voice_status(self) -> Dict[str, Any]:
        """Get current voice UI status"""
        return {
            "is_listening": self.is_listening,
            "is_speaking": self.is_speaking,
            "active_sessions": len([s for s in self.active_sessions.values() if s["is_active"]]),
            "total_sessions": len(self.active_sessions),
            "voice_config": self.voice_config.dict(),
            "audio_interface_available": self.audio_interface is not None,
            "speech_recognition_available": self.speech_recognizer is not None,
            "text_to_speech_available": self.text_to_speech is not None,
            "wake_word_detection": self.wake_word_detector["enabled"] if self.wake_word_detector else False
        }

async def main():
    """Main execution function"""
    voice_orchestrator = VoiceUIOrchestrator()
    await voice_orchestrator.initialize()
    
    # Start a voice session
    session_id = await voice_orchestrator.start_voice_session()
    
    # Example voice command processing
    result = await voice_orchestrator.process_voice_command(
        "Hello, show me the current system status"
    )
    
    print("Voice Command Result:")
    print(json.dumps(result, indent=2, default=str))
    
    # Get status
    status = await voice_orchestrator.get_voice_status()
    print("\nVoice UI Status:")
    print(json.dumps(status, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(main())