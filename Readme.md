🚀 PolyLens — Real-Time Multilingual Translation System

A multimodal AI-powered translation system that works with images, speech, and text in real time.

📌 Project Overview

PolyLens is a real-time multilingual translation system that converts between:

📷 Image → Text → Translation
🎙 Speech → Text → Translation
✍ Text → Speech
🖼 Live Camera → Real-time Translation

Built using:

Flask Backend
Google Cloud APIs (OCR, Translation, TTS, STT)

👉 Supports 25+ languages with high accuracy and low latency.

✨ Key Features
🔥 Core Features
📷 Live Camera Translation
🖼 Image Upload Translation
✍ Text → Speech
🎙 Speech → Text + Translation
🧠 Advanced AI Features (Novelty)

PolyLens includes 13 research-level innovations:

Feature	Description
N1	OCR Repair (fixes noisy text)
N2	Speech-aware normalization
N3	Real-time latency tracking
N4	Semantic deduplication
N5	Script-aware TTS
N6	Discourse memory
N7	Domain-aware translation
N8	Multi-frame OCR voting
N9	Layout reconstruction
N10	Formality detection
N11	Word confidence highlighting
N12	Latency anomaly detection
N13	Auto-benchmark system

👉 These make PolyLens more intelligent than traditional translators.

🏗 System Architecture
Input (Camera / Image / Speech / Text)
            ↓
        OCR Engine
            ↓
   OCR Repair + Filtering
            ↓
     Translation Engine
            ↓
 Context + Domain Processing
            ↓
      Text Normalization
            ↓
       TTS / Output
📂 Project Structure
polylens/
│
├── app.py                  # Flask backend
├── service_account.json   # Google Cloud credentials
├── polylens_log.csv       # Performance logs
│
└── templates/
    └── index.html         # Frontend UI
⚙️ Tech Stack
Technology	Purpose
Flask	Backend server
Google Vision API	OCR
Google Translate API	Translation
Google TTS	Speech output
Google STT	Speech recognition
🛠 Installation Guide
1️⃣ Clone the Repository
git clone <your-repo-link>
cd polylens
2️⃣ Setup Environment
python -m venv venv

Activate:

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate
3️⃣ Install Dependencies
pip install flask google-cloud-vision google-cloud-translate google-cloud-texttospeech google-cloud-speech
4️⃣ Setup Google Credentials
set GOOGLE_APPLICATION_CREDENTIALS=service_account.json

(Mac/Linux use export)

5️⃣ Run the App
python app.py

Open browser:

http://127.0.0.1:5000
📸 Usage Guide
📷 Live Camera
Start camera
Point at text
Get instant translation + audio
🖼 Image Upload
Upload image
Extract text + translate
View OCR repair logs
✍ Text → Speech
Enter text
Translate + listen
🎙 Speech → Text
Speak input
Get translated output
📊 Performance Features
⏱ Latency Tracking (OCR / Translation / TTS)
📉 Anomaly Detection
📈 Benchmark Logging (CSV)
🎯 Confidence Visualization
⚠️ Troubleshooting
Issue	Solution
Camera not working	Allow browser permissions
Mic not working	Enable microphone access
API error	Check Google Cloud APIs
OCR low accuracy	Improve lighting
🎯 Key Achievements
✅ Real-time multimodal translation
✅ Context-aware processing
✅ High OCR accuracy using voting
✅ Reduced redundancy via semantic filtering
✅ Scalable architecture
🔮 Future Enhancements
🤖 Deep Learning-based OCR models
🌍 Offline translation support
📱 Mobile application
🧠 Better domain intelligence
👨‍💻 Author

Ayush Raj
B.Tech CSE — SRM Institute of Science and Technology

📜 License

This project is for academic and research purposes.

⭐ Final Note

PolyLens is not just a translator —
it is a context-aware intelligent multilingual system designed for real-world applications.