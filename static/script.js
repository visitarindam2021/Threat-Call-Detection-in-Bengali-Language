let mediaRecorder;
let audioChunks = [];

async function uploadAudio() {
    flashButton('detectBtn');
    const input = document.getElementById('audioInput');
    const file = input.files[0];

    if (!file) {
        alert('Please select a voice file!');
        return;
    }

    const formData = new FormData();
    formData.append('audio', file);

    flashButton('audioInput');
    await sendToBackend(formData);
}

function startRecording() {
    flashButton('recordBtn');

    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();
            audioChunks = [];

            mediaRecorder.addEventListener("dataavailable", event => {
                audioChunks.push(event.data);
            });

            // UI updates
            document.getElementById('recordBtn').classList.add('active-btn');
            document.getElementById('stopBtn').classList.remove('active-btn');
            document.getElementById('recordBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('audioInput').disabled = true;

            // Show mic indicator
            document.getElementById('micStatus').style.display = 'block';
        })
        .catch(error => {
            console.error("Error accessing microphone:", error);
        });
    
    /*navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();
            audioChunks = [];

            mediaRecorder.addEventListener("dataavailable", event => {
                audioChunks.push(event.data);
            });

            document.getElementById('recordBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
        })
        .catch(error => {
            console.error("Error accessing microphone:", error);
        });*/
}

function stopRecording() {
    flashButton('stopBtn');
    mediaRecorder.stop();
    mediaRecorder.addEventListener("stop", () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recorded_audio.wav');

        sendToBackend(formData);
    });

    // UI updates
    document.getElementById('recordBtn').classList.remove('active-btn');
    document.getElementById('stopBtn').classList.remove('active-btn');
    document.getElementById('recordBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    document.getElementById('audioInput').disabled = false;

    // Hide mic indicator
    document.getElementById('micStatus').style.display = 'none';
    
    /*mediaRecorder.stop();
    mediaRecorder.addEventListener("stop", () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recorded_audio.wav');

        sendToBackend(formData);
    });

    document.getElementById('recordBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;*/
}

function flashButton(buttonId) {
    const btn = document.getElementById(buttonId);
    btn.classList.add('clicked');
    setTimeout(() => {
        btn.classList.remove('clicked');
    }, 300); // Adjust duration if you want it to last longer
}


async function sendToBackend(formData) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerText = "🔎 Analyzing...";
    resultDiv.style.color = "#00bcd4";

    const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData
    });

    const data = await response.json();

    if (data.result === 'Threat') {
        resultDiv.style.color = 'red';
        resultDiv.innerText = '⚠️ Threat Detected!';
    } else {
        resultDiv.style.color = 'lime';
        resultDiv.innerText = '✅ No Threat Detected!';
    }
}
