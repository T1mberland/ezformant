import init, { process_audio } from './pkg/ezformant.js';

const canvas = document.getElementById('spectrum');
const ctx = canvas.getContext('2d');

async function start() {
    await init();

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();
    source.connect(analyser);

    analyser.fftSize = 2048;
    const bufferLength = analyser.fftSize;
    const dataArray = new Float32Array(bufferLength);

    function drawSpectrum() {
        analyser.getFloatTimeDomainData(dataArray);

        const spectrum = process_audio(dataArray, 0); // `lpc_order` is unused in the Rust code

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();

        const barWidth = canvas.width / spectrum.length;

        for (let i = 0; i < spectrum.length; i++) {
            const barHeight = (spectrum[i] / Math.max(...spectrum)) * canvas.height;
            ctx.fillStyle = 'rgb(0, 0, 255)';
            ctx.fillRect(i * barWidth, canvas.height - barHeight, barWidth, barHeight);
        }

        ctx.stroke();

        requestAnimationFrame(drawSpectrum);
    }

    drawSpectrum();
}

start();
