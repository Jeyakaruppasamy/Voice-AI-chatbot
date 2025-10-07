"""
Voice AI Agent Chatbot
A production-ready voice-enabled AI chatbot with speech recognition and synthesis.
"""

import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Data class for chat messages."""
    role: str
    content: str
    timestamp: datetime
    audio_path: Optional[str] = None


class AudioProcessor(ABC):
    """Abstract base class for audio processing."""
    
    @abstractmethod
    def process(self, *args, **kwargs):
        """Process audio input/output."""
        pass


class SpeechRecognizer(AudioProcessor):
    """Handles speech-to-text conversion."""
    
    def __init__(self, language: str = "en-US"):
        self.recognizer = sr.Recognizer()
        self.language = language
        
    def process(self, audio_source: str = "microphone") -> Optional[str]:
        """
        Convert speech to text.
        
        Args:
            audio_source: Source of audio input
            
        Returns:
            Transcribed text or None if recognition fails
        """
        try:
            with sr.Microphone() as source:
                logger.info("Listening...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
            text = self.recognizer.recognize_google(audio, language=self.language)
            logger.info(f"Recognized: {text}")
            return text
            
        except sr.WaitTimeoutError:
            logger.warning("Listening timed out")
            return None
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            return None
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return None


class TextToSpeech(AudioProcessor):
    """Handles text-to-speech conversion."""
    
    def __init__(self, language: str = "en"):
        self.language = language
        self.temp_dir = Path(tempfile.gettempdir()) / "voice_ai_agent"
        self.temp_dir.mkdir(exist_ok=True)
        
    def process(self, text: str) -> Optional[str]:
        """
        Convert text to speech.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Path to generated audio file or None if conversion fails
        """
        try:
            tts = gTTS(text=text, lang=self.language, slow=False)
            audio_path = self.temp_dir / f"response_{datetime.now().timestamp()}.mp3"
            tts.save(str(audio_path))
            logger.info(f"Generated audio: {audio_path}")
            return str(audio_path)
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None


class AIAgent:
    """AI agent for generating responses."""
    
    def __init__(self):
        self.context: List[Dict[str, str]] = []
        self.max_context_length = 10
        
    def generate_response(self, user_input: str) -> str:
        """
        Generate AI response based on user input.
        
        Args:
            user_input: User's message
            
        Returns:
            AI-generated response
        """
        # Add user input to context
        self.context.append({"role": "user", "content": user_input})
        
        # Keep context manageable
        if len(self.context) > self.max_context_length:
            self.context = self.context[-self.max_context_length:]
        
        # Simple rule-based response (replace with actual AI model)
        response = self._rule_based_response(user_input.lower())
        
        # Add response to context
        self.context.append({"role": "assistant", "content": response})
        
        return response
    
    def _rule_based_response(self, text: str) -> str:
        """
        Generate rule-based response.
        Replace this with actual ML/DL model for production.
        """
        if any(greeting in text for greeting in ["hello", "hi", "hey"]):
            return "Hello! How can I assist you today?"
        elif "how are you" in text:
            return "I'm functioning well, thank you! How can I help you?"
        elif any(word in text for word in ["bye", "goodbye", "see you"]):
            return "Goodbye! Have a great day!"
        elif "name" in text:
            return "I'm your AI voice assistant. What would you like to know?"
        elif "help" in text:
            return "I can help you with conversations, answer questions, and assist with various tasks. What do you need?"
        elif "?" in text:
            return f"That's an interesting question about '{text[:50]}...'. Let me think about that."
        else:
            return "I understand. Could you tell me more about what you need?"


class ChatSession:
    """Manages chat session state and history."""
    
    def __init__(self):
        self.messages: List[Message] = []
        
    def add_message(self, role: str, content: str, audio_path: Optional[str] = None):
        """Add a message to the chat history."""
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now(),
            audio_path=audio_path
        )
        self.messages.append(message)
        
    def get_messages(self) -> List[Message]:
        """Retrieve all messages."""
        return self.messages
    
    def clear(self):
        """Clear chat history."""
        self.messages.clear()


class VoiceAIChatbot:
    """Main application controller."""
    
    def __init__(self):
        self.speech_recognizer = SpeechRecognizer()
        self.tts = TextToSpeech()
        self.ai_agent = AIAgent()
        self.session = ChatSession()
        
    def process_voice_input(self) -> Optional[str]:
        """Process voice input and return transcribed text."""
        return self.speech_recognizer.process()
    
    def generate_response(self, user_input: str) -> tuple[str, Optional[str]]:
        """
        Generate AI response and audio.
        
        Returns:
            Tuple of (response_text, audio_path)
        """
        response = self.ai_agent.generate_response(user_input)
        audio_path = self.tts.process(response)
        return response, audio_path
    
    def add_to_history(self, role: str, content: str, audio_path: Optional[str] = None):
        """Add message to session history."""
        self.session.add_message(role, content, audio_path)


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = VoiceAIChatbot()
    if 'is_listening' not in st.session_state:
        st.session_state.is_listening = False


def render_chat_message(message: Message):
    """Render a chat message with audio playback."""
    with st.chat_message(message.role):
        st.write(message.content)
        st.caption(message.timestamp.strftime("%I:%M %p"))
        if message.audio_path and os.path.exists(message.audio_path):
            st.audio(message.audio_path)


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Voice AI Agent",
        page_icon="🎙️",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        
        st.subheader("Voice Settings")
        voice_enabled = st.toggle("Enable Voice Output", value=True)
        
        st.subheader("Session Management")
        if st.button("Clear Chat History", type="secondary"):
            st.session_state.chatbot.session.clear()
            st.rerun()
        
        st.divider()
        st.subheader("About")
        st.info("""
        **Voice AI Agent Chatbot**
        
        Features:
        - 🎤 Speech Recognition
        - 🤖 AI-Powered Responses
        - 🔊 Text-to-Speech
        - 💬 Chat History
        """)
    
    # Main content
    st.title("🎙️ Voice AI Agent Chatbot")
    st.markdown("Speak or type to interact with your AI assistant")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chatbot.session.get_messages():
            render_chat_message(message)
    
    # Input area
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.chat_input("Type your message here...")
    
    with col2:
        voice_button = st.button("🎤 Speak", use_container_width=True, type="primary")
    
    # Handle voice input
    if voice_button:
        with st.spinner("Listening..."):
            transcribed_text = st.session_state.chatbot.process_voice_input()
            
            if transcribed_text:
                user_input = transcribed_text
                st.success(f"You said: {transcribed_text}")
            else:
                st.error("Could not recognize speech. Please try again.")
    
    # Process user input
    if user_input:
        # Add user message
        st.session_state.chatbot.add_to_history("user", user_input)
        
        # Generate AI response
        with st.spinner("Thinking..."):
            response, audio_path = st.session_state.chatbot.generate_response(user_input)
            
            # Add assistant message
            if voice_enabled:
                st.session_state.chatbot.add_to_history("assistant", response, audio_path)
            else:
                st.session_state.chatbot.add_to_history("assistant", response)
        
        st.rerun()


if __name__ == "__main__":
    main()