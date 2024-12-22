import init, { process_audio, lpc_filter_freq_responce } from './pkg/ezformant.js';

const canvas = document.getElementById('spectrum');
const ctx = canvas.getContext('2d');

async function start() {
    await init();

    const stream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        autoGainControl: false,
        noiseSuppression: false,
        echoCancellation: false
      }
    });
    const audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();
    source.connect(analyser);

    analyser.fftSize = 2048;
    const bufferLength = analyser.fftSize;
    const dataArray = new Float32Array(bufferLength);
    const filterDataArray = new Float32Array(bufferLength);
    const sampleRate = audioContext.sampleRate;

    // Number of bars in the spectrum
    const numBars = 512; // Adjust as needed

    // Precompute logarithmic frequency boundaries
    const minFrequency = 20; // Minimum frequency to display
    const maxFrequency = sampleRate / 2; // Nyquist frequency

    const logMin = Math.log10(minFrequency);
    const logMax = Math.log10(maxFrequency);
    const logRange = logMax - logMin;

    // Function to get frequency for a given bin index
    function getFrequency(index) {
        return index * sampleRate / analyser.fftSize;
    }

    // Precompute frequency boundaries for each bar
    const frequencyBins = [];
    for (let i = 0; i < numBars; i++) {
        const logFreq = logMin + (i / numBars) * logRange;
        frequencyBins.push(Math.pow(10, logFreq));
    }

    function drawSpectrum() {
        analyser.getFloatTimeDomainData(dataArray);

        const spectrum = process_audio(Array.from(dataArray), 0); // Convert Float32Array to Array

        // Clear the canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const barWidth = canvas.width / numBars;

        // Optional: Use a logarithmic scale for the y-axis as well
        // const magnitudeScale = Math.log10(1 + Math.max(...spectrum));

        for (let i = 0; i < numBars; i++) {
            // Determine the frequency range for this bar
            const freqStart = frequencyBins[i];
            const freqEnd = frequencyBins[i + 1] || maxFrequency;

            // Find corresponding FFT bins
            const binStart = Math.floor(freqStart * analyser.fftSize / sampleRate);
            const binEnd = Math.floor(freqEnd * analyser.fftSize / sampleRate);

            // Aggregate the magnitude within these bins
            let magnitude = 0;
            for (let bin = binStart; bin <= binEnd; bin++) {
                if (bin < spectrum.length) {
                    magnitude += spectrum[bin];
                }
            }
            // Average the magnitude
            magnitude = magnitude / (binEnd - binStart + 1);

            // Normalize the magnitude (optional: apply logarithmic scaling)
            const normalizedMagnitude = magnitude / 100; // Adjust based on expected magnitude range

            const barHeight = normalizedMagnitude * canvas.height;

            ctx.fillStyle = 'rgb(0, 0, 255)';
            ctx.fillRect(i * barWidth, canvas.height - barHeight, barWidth, barHeight);
        }

        requestAnimationFrame(drawSpectrum);
    }

    function drawLPCFilter() {
      analyser.getFloatTimeDomainData(dataArray);

      for (let i = 0; i < dataArray.length; i++) {
        let window = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (dataArray.length - 1)));
        dataArray[i] *= window;
      }

      const freqResponce = lpc_filter_freq_responce(Array.from(dataArray), 8, sampleRate, numBars);

      if (freqResponce.every(value => value === 0)) {
          requestAnimationFrame(drawLPCFilter);
          return;
      }

      ctx.beginPath();
      ctx.moveTo(0,0);
      for (let i = 0; i < numBars; ++i) {
        const freqStart = frequencyBins[i];

        const binStart = Math.floor(freqStart * analyser.fftSize / sampleRate);

        const xPos = canvas.width * freqStart / maxFrequency;
        const yPos = freqResponce[freqStart];

        ctx.lineTo(xPos, yPos);
        ctx.moveTo(xPos, yPos);
      }
      ctx.stroke();
      //console.log(freqResponce);

      requestAnimationFrame(drawLPCFilter);
    }

    drawSpectrum();
    drawLPCFilter();
}

start();
