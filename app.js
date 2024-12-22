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

  analyser.fftSize = 1024;
  const bufferLength = analyser.fftSize;
  const dataArray = new Float32Array(bufferLength);
  const filterDataArray = new Float32Array(bufferLength);
  const sampleRate = audioContext.sampleRate;

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

  function frequencyToPosition(freq) {
    return (Math.log10(freq) - logMin) / logRange * canvas.width;
  }

  function drawSpectrum() {
    analyser.getFloatTimeDomainData(dataArray);

    const spectrum = process_audio(Array.from(dataArray), 0); // calculates FFT here.

    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Begin drawing the spectrum
    ctx.fillStyle = '#000'; // Background color
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = '#00ff00'; // Spectrum line color
    ctx.lineWidth = 2;
    ctx.beginPath();

    // Determine the maximum magnitude for normalization
    const maxMagnitude = Math.max(...spectrum.map(Math.abs));

    // Iterate through the spectrum data
    for (let i = 0; i < spectrum.length; i++) {
      const freq = getFrequency(i);
      if (freq < minFrequency || freq > maxFrequency) continue;

      const x = frequencyToPosition(freq);
      const magnitude = spectrum[i];
      // Normalize magnitude to fit the canvas height
      const y = canvas.height - (Math.abs(magnitude) / maxMagnitude) * canvas.height;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }

    ctx.stroke();

    // Optionally, draw frequency labels
    drawFrequencyLabels();

    requestAnimationFrame(drawSpectrum);
  }

  function drawFrequencyLabels() {
    const labelCount = 10; // Number of labels to display
    ctx.fillStyle = '#ffffff'; // Label color
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    for (let i = 0; i <= labelCount; i++) {
      const freq = minFrequency * Math.pow(10, (logRange * i) / labelCount);
      const x = frequencyToPosition(freq);
      ctx.beginPath();
      ctx.moveTo(x, canvas.height);
      ctx.lineTo(x, canvas.height - 10);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillText(Math.round(freq) + ' Hz', x, canvas.height - 25);
    }
  }

  function drawLPCFilter() {
    analyser.getFloatTimeDomainData(dataArray);

    const graphSize = 1024;

    const freqResponce = lpc_filter_freq_responce(Array.from(dataArray), 16, sampleRate, graphSize);
    const normalizeConst = 1;

    if (freqResponce.every(value => value === 0)) {
      requestAnimationFrame(drawLPCFilter);
      return;
    }
    ctx.strokeStyle = "red";

    ctx.beginPath();
    ctx.moveTo(0, canvas.height);
    for (let i = 0; i < graphSize; ++i) {
      const freq = i * maxFrequency / graphSize;

      if (freq < minFrequency) { continue; } 

      const xPos = frequencyToPosition(freq);
      const yPos = canvas.height - freqResponce[i] / normalizeConst;

      ctx.lineTo(xPos, yPos);
      ctx.moveTo(xPos, yPos);

      ctx.strokeStyle = "red";
    }
    ctx.stroke();
    //console.log(freqResponce);

    requestAnimationFrame(drawLPCFilter);
  }

  drawSpectrum();
  drawLPCFilter();
}

start();
