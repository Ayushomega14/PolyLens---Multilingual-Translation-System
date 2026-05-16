const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const output = document.getElementById("output");
const stats = document.getElementById("stats");
const targetLang = document.getElementById("targetLang");

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => video.srcObject = stream)
  .catch(() => alert("Camera access denied"));

function speak(text, lang) {
  const synth = window.speechSynthesis;

  function speakNow() {
    const voices = synth.getVoices();
    let voice =
      voices.find(v => v.lang === lang && v.name.includes("Google")) ||
      voices.find(v => v.lang.startsWith(lang.split("-")[0])) ||
      voices.find(v => v.lang.startsWith("en")) ||
      voices[0];

    const utter = new SpeechSynthesisUtterance(text);
    utter.voice = voice;
    utter.lang = voice.lang;
    utter.rate = 0.9;
    utter.pitch = 1.0;

    synth.cancel();
    synth.speak(utter);
  }

  if (!synth.getVoices().length) {
    synth.onvoiceschanged = speakNow;
  } else {
    speakNow();
  }
}

setInterval(() => {
  if (!video.videoWidth) return;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);

  fetch("/process", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image: canvas.toDataURL("image/jpeg"),
      target_lang: targetLang.value
    })
  })
  .then(res => res.json())
  .then(data => {
    if (data.skipped || !data.translated) return;

    output.innerText = data.translated;
    stats.innerText =
      `Latency: ${data.latency} ms | Confidence: ${data.confidence}`;

    speak(data.translated, targetLang.value);
  });

}, 3000);