import init from "./pkg/ezformant.js"; // Import only if needed in the main thread
import {
  process_audio,
  lpc_filter_freq_response,
  lpc_filter_freq_response_with_downsampling,
} from "./pkg/ezformant.js";

const canvas = document.getElementById("spectrum");
const ctx = canvas.getContext("2d");
const labelF1 = document.getElementById("label-f1");
const labelF2 = document.getElementById("label-f2");
const labelF3 = document.getElementById("label-f3");

const MAX_HISTORY = 1000;
let formantHistory = [];

let formant1 = 0.0;
let formant2 = 0.0;
let formant3 = 0.0;
let formant4 = 0.0;

let showFFTSpectrum = document.getElementById("showFFTSpectrum");
let showLPCSpectrum = document.getElementById("showLPCSpectrum");
let showFormants = document.getElementById("showFormants");

// Initialize the Web Worker as an ES Module
const formantWorker = new Worker("formantWorker.mjs", { type: "module" });

canvas.width = window.innerWidth;
canvas.height = window.innerHeight / 2;

document.getElementById("showSpectrumTabBtn").onclick = function () {
  document.getElementById("spectrum").style.display = "block";
  document.getElementById("historyCanvas").style.display = "none";
};

document.getElementById("showHistoryTabBtn").onclick = function () {
  document.getElementById("spectrum").style.display = "none";
  document.getElementById("historyCanvas").style.display = "block";
  console.log("test");
};

function addFormantsToHistory(f1, f2, f3, f4) {
  const timestamp = performance.now();
  // or you can store time in seconds relative to start
  formantHistory.push({ time: timestamp, f1, f2, f3, f4 });

  // Keep buffer from growing indefinitely
  if (formantHistory.length > MAX_HISTORY) {
    formantHistory.shift();
  }
}

// Handle messages from the worker
formantWorker.onmessage = function (e) {
  const { type, status, formants, error } = e.data;

  if (type === "init") {
    if (status === "success") {
      console.log("Worker initialized successfully.");
    } else {
      console.error("Worker initialization failed:", error);
      alert("Failed to initialize formant detection worker.");
    }
  } else if (type === "calcFormants") {
    if (status === "success") {
      [formant1, formant2, formant3, formant4] = formants;

      // Store these formants in our rolling history
      addFormantsToHistory(formant1, formant2, formant3, formant4);
    } else {
      console.error("Formant detection failed:", error);
    }
  }
};

function drawAxisTicks(
  hctx,
  {
    minTime, // earliest time in the window (ms)
    now, // current time (ms)
    timeWindow, // e.g. 5000 ms for a 5 second window
    minFreq, // e.g. 0 Hz
    maxFreq, // e.g. 5000 Hz
    xFromTime, // function that maps time to x-position
    yFromFreq, // function that maps frequency to y-position
    width, // canvas width in px
    height, // canvas height in px
  },
) {
  // Decide how many grid lines or ticks you’d like for time/frequency
  const numTimeTicks = 5;
  const numFreqTicks = 5;

  // Optional styling
  hctx.save();
  hctx.strokeStyle = "#444"; // Grid lines
  hctx.fillStyle = "#fff"; // Text color
  hctx.font = "12px sans-serif";

  // 1) TIME AXIS TICKS (vertical lines)
  //
  // We'll create 'numTimeTicks + 1' equally spaced lines across the time range.
  // 'minTime' is the left boundary of our time window, 'now' is the right boundary.

  for (let i = 0; i <= numTimeTicks; i++) {
    // The fraction along the time window for this tick
    const fraction = i / numTimeTicks;
    // The time in ms for this tick
    const tickTime = minTime + fraction * timeWindow;
    // Convert that to an x-position on the canvas
    const xPos = xFromTime(tickTime);

    // Draw the vertical line (from top to bottom of the canvas)
    hctx.beginPath();
    hctx.moveTo(xPos, 0);
    hctx.lineTo(xPos, height);
    hctx.stroke();

    // Create a label showing how many seconds from the start of the window
    // If you’d rather show absolute time, you can do (tickTime / 1000).toFixed(1).
    const elapsedSeconds = ((tickTime - minTime) / 1000).toFixed(1);
    const label = elapsedSeconds + "s";
    // Place the text near the bottom of the canvas
    hctx.fillText(label, xPos + 2, height - 5);
  }

  // 2) FREQUENCY AXIS TICKS (horizontal lines)
  //
  // We'll create 'numFreqTicks + 1' equally spaced lines from minFreq to maxFreq.

  for (let i = 0; i <= numFreqTicks; i++) {
    // The fraction along the frequency range
    const fraction = i / numFreqTicks;
    // The frequency in Hz for this tick
    const freq = minFreq + fraction * (maxFreq - minFreq);
    // Convert to y-position
    const yPos = yFromFreq(freq);

    // Draw the horizontal line (across the entire canvas width)
    hctx.beginPath();
    hctx.moveTo(0, yPos);
    hctx.lineTo(width, yPos);
    hctx.stroke();

    // Label the frequency
    const freqLabel = freq.toFixed(0) + " Hz";
    // Place it slightly to the left (or right) of the line
    hctx.fillText(freqLabel, 5, yPos - 5);
  }

  hctx.restore();
}

async function start() {
  try {
    // Initialize WASM in the main thread if necessary
    await init();

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        autoGainControl: false,
        noiseSuppression: false,
        echoCancellation: false,
      },
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
    const downsampleFactor = 4;

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
      return ((Math.log10(freq) - logMin) / logRange) * canvas.width;
    }

    function drawSpectrum() {
      analyser.getFloatTimeDomainData(dataArray);
      // Clear the canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#000"; // Background color
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      if (showFFTSpectrum.checked) {
        const spectrumData = process_audio(Array.from(dataArray)); // Ensure process_audio returns bufferLength data
        spectrum.set(spectrumData.slice(0, bufferLength)); // Use only unique bins

        // Begin drawing the spectrum
        ctx.fillStyle = "#000"; // Background color
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.strokeStyle = "#00ff00"; // Spectrum line color
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
          const y =
            canvas.height - (logMagnitude / logMaxMagnitude) * canvas.height;

          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }

        ctx.stroke();
      }

      updateFormantText();
      drawFrequencyLabels();

      requestAnimationFrame(drawSpectrum);
    }

    function updateFormantText() {
      labelF1.innerHTML = "F1: " + formant1.toFixed(0);
      labelF2.innerHTML = "F2: " + formant2.toFixed(0);
      labelF3.innerHTML = "F3: " + formant3.toFixed(0);
    }

    function drawFrequencyLabels() {
      const labelCount = 20; // Number of labels to display
      ctx.fillStyle = "#ffffff"; // Label color
      ctx.font = "12px Arial";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";

      for (let i = 0; i <= labelCount; i++) {
        const freq = minFrequency * Math.pow(10, (logRange * i) / labelCount);
        const x = frequencyToPosition(freq);
        ctx.beginPath();
        ctx.moveTo(x, canvas.height);
        ctx.lineTo(x, canvas.height - 10);
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.fillText(Math.round(freq) + " Hz", x, canvas.height - 25);
      }
    }

    function drawLPCFilter() {
      if (showLPCSpectrum.checked) {
        analyser.getFloatTimeDomainData(dataArray);

        const graphSize = 1024;

        // Instead of calling WASM functions directly, we need to offload this to the worker
        // Remove or comment out the following lines:
        // const freqResponce = lpc_filter_freq_response_with_downsampling(Array.from(dataArray), 16, sampleRate, downsampleFactor, graphSize);
        //
        // Instead, you can request the worker to compute the frequency response if needed.

        // For now, let's keep the existing code and focus on moving calcFormants to the worker
        // If you also want to move lpc_filter_freq_response_with_downsampling, you can follow similar steps

        // Proceeding with existing implementation

        const freqResponce = lpc_filter_freq_response_with_downsampling(
          Array.from(dataArray),
          16,
          sampleRate,
          downsampleFactor,
          graphSize,
        );

        if (freqResponce.every((value) => value === 0)) {
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
          const freq = (i * maxFrequency) / graphSize / downsampleFactor;
          if (freq < minFrequency) continue;

          const xPos = frequencyToPosition(freq);

          const logResponse = Math.log10(freqResponce[i] + 1); // Add 1 to avoid log10(0)
          const logMaxResponse = Math.log10(normalizeConst + 1);
          const yPos =
            canvas.height - (logResponse / logMaxResponse) * canvas.height;

          if (!started) {
            ctx.moveTo(xPos, yPos);
            started = true;
          } else {
            ctx.lineTo(xPos, yPos);
          }
        }

        ctx.stroke();
      }

      if (showFormants.checked) {
        // Draw the 1st formant
        ctx.strokeStyle = "white";
        ctx.beginPath();
        let xPos = frequencyToPosition(formant1);
        ctx.moveTo(xPos, 0);
        ctx.lineTo(xPos, canvas.height);
        ctx.stroke();
        ctx.fillText(formant1.toFixed(0), xPos, 0);

        // Draw the 2nd formant
        ctx.strokeStyle = "red";
        ctx.beginPath();
        xPos = frequencyToPosition(formant2);
        ctx.moveTo(xPos, 0);
        ctx.lineTo(xPos, canvas.height);
        ctx.stroke();
        ctx.fillText(formant2.toFixed(0), xPos, 0);

        // Draw the 3rd formant
        ctx.strokeStyle = "green";
        ctx.beginPath();
        xPos = frequencyToPosition(formant3);
        ctx.moveTo(xPos, 0);
        ctx.lineTo(xPos, canvas.height);
        ctx.stroke();
        ctx.fillText(formant3.toFixed(0), xPos, 0);
      }

      requestAnimationFrame(drawLPCFilter);
    }

    // Set up the interval to calculate formants using the worker
    setInterval(calcFormants, 100); // Run `calcFormants` every 500 milliseconds

    function calcFormants() {
      // Send data to the worker for processing
      formantWorker.postMessage({
        type: "calcFormants",
        data: {
          audioData: Array.from(dataArray), // Transfer as an array
          lpcOrder: 14, // Example LPC order, adjust as needed
          sampleRate: sampleRate,
          downsampleFactor: downsampleFactor,
        },
      });
    }

    // Initialize the worker by sending an 'init' message
    formantWorker.postMessage({ type: "init" });

    function drawFormantHistory() {
      const historyCanvas = document.getElementById("historyCanvas");
      const hctx = historyCanvas.getContext("2d");

      // Clear and fill background
      hctx.clearRect(0, 0, historyCanvas.width, historyCanvas.height);
      hctx.fillStyle = "#000";
      hctx.fillRect(0, 0, historyCanvas.width, historyCanvas.height);

      // Time range to display
      const now = performance.now();
      const timeWindow = 5000; // 5 seconds of data
      const minTime = now - timeWindow;

      // Frequency range
      const minFreq = 0;
      const maxFreq = 5000;

      // For convenience:
      const width = historyCanvas.width;
      const height = historyCanvas.height;

      // Map functions
      function xFromTime(t) {
        // t goes from [minTime, now]
        const fraction = (t - minTime) / timeWindow;
        return fraction * width;
      }
      function yFromFreq(f) {
        // [minFreq, maxFreq] mapped to [bottom, top]
        const fraction = (f - minFreq) / (maxFreq - minFreq);
        return height - fraction * height;
      }

      // Draw the axis ticks first (grid lines, etc.)
      drawAxisTicks(hctx, {
        minTime,
        now,
        timeWindow,
        minFreq,
        maxFreq,
        xFromTime,
        yFromFreq,
        width,
        height,
      });

      // Next, filter data to only keep the last 5 seconds
      const recentData = formantHistory.filter((d) => d.time >= minTime);
      if (recentData.length < 2) {
        requestAnimationFrame(drawFormantHistory);
        return;
      }

      // Draw each formant line
      drawSingleFormantLine(recentData, "f1", "#ff0000");
      drawSingleFormantLine(recentData, "f2", "#00ff00");
      drawSingleFormantLine(recentData, "f3", "#0000ff");
      // If you have a 4th formant, add it as well

      function drawSingleFormantLine(data, formantKey, color) {
        hctx.strokeStyle = color;
        hctx.beginPath();
        for (let i = 0; i < data.length; i++) {
          const x = xFromTime(data[i].time);
          const y = yFromFreq(data[i][formantKey]);
          if (i === 0) {
            hctx.moveTo(x, y);
          } else {
            hctx.lineTo(x, y);
          }
        }
        hctx.stroke();
      }

      requestAnimationFrame(drawFormantHistory);
    }

    // Kick it off once
    drawFormantHistory();

    drawSpectrum();
    drawLPCFilter();
  } catch (error) {
    console.error("Error accessing audio stream:", error);
    alert("Could not access the microphone. Please check your permissions.");
  }
}

start();
