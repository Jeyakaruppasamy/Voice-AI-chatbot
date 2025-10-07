# Voice-AI-chatbot
A production-ready voice-enabled AI chatbot built with Python, featuring speech recognition, AI-powered responses, and text-to-speech capabilities. Built with clean architecture and solid software engineering principles.
✨ Features

🎤 Speech Recognition: Convert voice to text using Google Speech API
🤖 AI-Powered Responses: Intelligent conversation handling (extensible for ML/DL models)
🔊 Text-to-Speech: Natural voice responses using gTTS
💬 Chat History: Persistent conversation tracking with timestamps
🎨 Modern UI: Clean, intuitive Streamlit interface
⚙️ Configurable Settings: Toggle voice output and manage sessions
📝 Comprehensive Logging: Track all operations for debugging

🏗️ Architecture
Built with professional software engineering principles:

Separation of Concerns: Modular design with distinct components
Abstract Base Classes: Extensible audio processing framework
Type Safety: Full type hints throughout
Error Handling: Graceful failure management
Logging: Comprehensive operation tracking
Clean Code: Well-documented, maintainable codebase

Component Structure
VoiceAIChatbot/
├── SpeechRecognizer      # Speech-to-text processing
├── TextToSpeech          # Text-to-speech synthesis
├── AIAgent               # Response generation (ML/DL ready)
├── ChatSession           # Session and history management
└── VoiceAIChatbot        # Main application controller
🚀 Quick Start
Prerequisites

Python 3.8 or higher
Microphone access
Internet connection (for speech recognition)

Installation

Clone the repository

bashgit clone https://github.com/yourusername/voice-ai-chatbot.git
cd voice-ai-chatbot

Create a virtual environment (recommended)

bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies

bashpip install -r requirements.txt
Required Dependencies
txtstreamlit>=1.28.0
SpeechRecognition>=3.10.0
gTTS>=2.4.0
PyAudio>=0.2.13
Platform-Specific Setup
macOS:
bashbrew install portaudio
pip install pyaudio
Linux (Ubuntu/Debian):
bashsudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
Windows:
bashpip install pipwin
pipwin install pyaudio
💻 Usage
Running the Application
bashstreamlit run app.py
The application will open in your default browser at http://localhost:8501
Basic Operations

Voice Input: Click the 🎤 Speak button and speak clearly
Text Input: Type in the chat input box at the bottom
Listen to Responses: AI responses are automatically played (if voice output is enabled)
Manage History: Use sidebar to clear chat history

Configuration
Access settings in the sidebar:

Toggle voice output on/off
Clear chat history
View application information

🧠 Integrating ML/DL Models
The architecture is designed for easy ML/DL integration. Replace the _rule_based_response method in the AIAgent class:
Example: Hugging Face Transformers
pythonfrom transformers import pipeline

class AIAgent:
    def __init__(self):
        self.context: List[Dict[str, str]] = []
        self.model = pipeline("text-generation", model="gpt2")
        
    def generate_response(self, user_input: str) -> str:
        response = self.model(user_input, max_length=100)[0]['generated_text']
        return response
Example: OpenAI API
pythonimport openai

class AIAgent:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.context = []
        
    def generate_response(self, user_input: str) -> str:
        self.context.append({"role": "user", "content": user_input})
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=self.context
        )
        
        return response.choices[0].message.content
Supported Integration Options

Transformers: Hugging Face models (BERT, GPT, T5, etc.)
LLMs: OpenAI, Anthropic, Cohere APIs
Custom Models: TensorFlow, PyTorch, or scikit-learn models
RAG Systems: LangChain, LlamaIndex integration
Local Models: Llama.cpp, GGML, or ONNX models

📁 Project Structure
voice-ai-chatbot/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore            # Git ignore rules
├── LICENSE               # MIT License
└── docs/                 # Additional documentation
    ├── architecture.md   # Architecture details
    └── api.md           # API reference
🛠️ Development
Code Style
The project follows PEP 8 guidelines and uses:

Type hints for all functions
Docstrings for all classes and methods
Logging for debugging and monitoring

Testing
bash# Run tests (when implemented)
pytest tests/

# Run with coverage
pytest --cov=app tests/
Contributing

Fork the repository
Create a feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

🐛 Troubleshooting
Microphone Not Working
Issue: "Could not recognize speech"
Solutions:

Check microphone permissions in system settings
Ensure microphone is properly connected
Test microphone in other applications
Try adjusting the phrase_time_limit parameter

PyAudio Installation Issues
macOS: Install portaudio first
bashbrew install portaudio
Linux: Install development libraries
bashsudo apt-get install portaudio19-dev
Windows: Use pipwin for binary installation
bashpip install pipwin
pipwin install pyaudio
Audio Playback Issues
If audio doesn't play:

Check system volume settings
Verify browser audio permissions
Try disabling browser extensions
Check Streamlit audio player compatibility

📊 Performance Optimization

Context Management: Limits conversation history to last 10 exchanges
Async Processing: Consider implementing async I/O for API calls
Caching: Add caching for repeated queries
Model Optimization: Use quantized models for faster inference

🔐 Security Considerations

API keys should be stored in environment variables
Implement rate limiting for production deployments
Validate and sanitize user inputs
Use HTTPS for production deployments
Consider implementing user authentication

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
🤝 Acknowledgments

Streamlit for the amazing web framework
SpeechRecognition for speech-to-text capabilities
gTTS for text-to-speech synthesis
The open-source community for inspiration and tools

📧 Contact
Your Name -Jeyakaruppasamy
Project Link:https://github.com/Jeyakaruppasamy/Voice-AI-chatbot
🗺️ Roadmap

 Add support for multiple languages
 Implement conversation export (JSON, PDF)
 Add voice cloning capabilities
 Integrate advanced LLM models
 Implement user authentication
 Add conversation analytics dashboard
 Support for custom wake words
 Real-time transcription display
 Mobile app version
 Docker containerization

⭐ Star History
If you find this project useful, please consider giving it a star!

Built with ❤️ using Python and Streamlit
