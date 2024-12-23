import init, { process_audio, lpc_filter_freq_response, lpc_filter_freq_response_with_peaks } from './pkg/ezformant.js';

const canvas = document.getElementById('spectrum');
const ctx = canvas.getContext('2d');

async function start() {
  try {
    await init();

    const stream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        autoGainControl: false,
        noiseSuppression: false,
        echoCancellation: false
      }
    });
    const audioContext = new AudioContext();
    await audioContext.resume(); // Ensure AudioContext is running
    const source = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();
    source.connect(analyser);

    const fftSize = 2048;
    analyser.fftSize = fftSize;
    const bufferLength = fftSize / 2;
    const dataArray = new Float32Array(fftSize);
    const spectrum = new Float32Array(bufferLength);
    const sampleRate = audioContext.sampleRate;

    // Precompute logarithmic frequency boundaries
    const minFrequency = 20; // Minimum frequency to display
    const maxFrequency = sampleRate / 2; // Nyquist frequency

    const logMin = Math.log10(minFrequency);
    const logMax = Math.log10(maxFrequency);
    const logRange = logMax - logMin;

    // Function to get frequency for a given bin index
    function getFrequency(index) {
      // Modified this line ONLY to align data with labels:
      return index * (maxFrequency / bufferLength);
    }

    function frequencyToPosition(freq) {
      return (Math.log10(freq) - logMin) / logRange * canvas.width;
    }

    function drawSpectrum() {
      analyser.getFloatTimeDomainData(dataArray);

      const spectrumData = process_audio(Array.from(dataArray), 0); // Ensure process_audio returns bufferLength data
      spectrum.set(spectrumData.slice(0, bufferLength)); // Use only unique bins

      // Clear the canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Begin drawing the spectrum
      ctx.fillStyle = '#000'; // Background color
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.strokeStyle = '#00ff00'; // Spectrum line color
      ctx.lineWidth = 2;
      ctx.beginPath();

      // Dynamically calculate maximum magnitude for normalization
      const maxMagnitude = Math.max(...spectrum.map(Math.abs), 10);

      // Iterate through the unique spectrum data
      for (let i = 0; i < bufferLength; i++) {
        const freq = getFrequency(i);
        if (freq < minFrequency || freq > maxFrequency) continue;

        const x = frequencyToPosition(freq);
        const magnitude = spectrum[i];

        const logMagnitude = Math.log10(Math.abs(magnitude) + 1); // Add 1 to avoid log10(0)
        const logMaxMagnitude = Math.log10(maxMagnitude + 1);
        const y = canvas.height - (logMagnitude / logMaxMagnitude) * canvas.height;

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
      const labelCount = 20; // Number of labels to display
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

      //console.log(lpc_filter_freq_response_with_peaks(Array.from(dataArray), sampleRate));

      const graphSize = 1024;
      //const formantsAndFreqResponce = lpc_filter_freq_response_with_peaks(Array.from(dataArray), 16, sampleRate, graphSize);
      //const freqResponce = formantsAndFreqResponce;
      const freqResponce = lpc_filter_freq_response(Array.from(dataArray), 16, sampleRate, graphSize);
      
      if (freqResponce.every(value => value === 0)) {
        requestAnimationFrame(drawLPCFilter);
        return;
      }

      // Normalize the frequency response
      const maxResponse = Math.max(...freqResponce);
      const normalizeConst = maxResponse > 0 ? maxResponse : 1;

      ctx.strokeStyle = "red";
      ctx.beginPath();
      let started = false;

      for (let i = 0; i < graphSize; ++i) {
        const freq = i * maxFrequency / graphSize;
        if (freq < minFrequency) continue;

        const xPos = frequencyToPosition(freq);

        const logResponse = Math.log10(freqResponce[i] + 1); // Add 1 to avoid log10(0)
        const logMaxResponse = Math.log10(normalizeConst + 1);
        const yPos = canvas.height - (logResponse / logMaxResponse) * canvas.height;

        if (!started) {
          ctx.moveTo(xPos, yPos);
          started = true;
        } else {
          ctx.lineTo(xPos, yPos);
        }
      }
      ctx.stroke();

      requestAnimationFrame(drawLPCFilter);
    }

    function calcFormants() {}

    drawSpectrum();
    drawLPCFilter();
  } catch (error) {
    console.error('Error accessing audio stream:', error);
    alert('Could not access the microphone. Please check your permissions.');
  }
}

start();
