import init, { 
  formant_detection_with_downsampling,
  lpc_filter_freq_response_with_downsampling
} from './pkg/ezformant.js';

let wasmInitialized = false;

// Initialize the WebAssembly module
async function initWasm() {
  if (!wasmInitialized) {
    await init(); // Initialize the WASM module
    wasmInitialized = true;
  }
}

// Listen for messages from the main thread
self.onmessage = async function(e) {
  const { type, data } = e.data;

  if (type === 'init') {
    // Initialize the WebAssembly module
    try {
      await initWasm();
      self.postMessage({ type: 'init', status: 'success' });
    } catch (error) {
      self.postMessage({ type: 'init', status: 'error', error: error.message });
    }
  }
  else if (type === 'calcFormants') {
    // Ensure WASM is initialized
    if (!wasmInitialized) {
      try {
        await initWasm();
      } catch (error) {
        self.postMessage({ type: 'calcFormants', status: 'error', error: error.message });
        return;
      }
    }

    // Destructure the incoming data
    const { audioData, lpcOrder, sampleRate, downsampleFactor } = data;

    try {
      // Perform formant detection using the provided function
      const formants = formant_detection_with_downsampling(
        audioData,
        lpcOrder,
        sampleRate,
        downsampleFactor
      );

      // Post the result back to the main thread
      self.postMessage({ type: 'calcFormants', status: 'success', formants });
    } catch (error) {
      self.postMessage({ type: 'calcFormants', status: 'error', error: error.message });
    }
  }
  else if (type === 'calcLPCFilter') {
    // Ensure WASM is initialized
    if (!wasmInitialized) {
      try {
        await initWasm();
      } catch (error) {
        self.postMessage({ type: 'calcLPCFilter', status: 'error', error: error.message });
        return;
      }
    }

    // Destructure the incoming data
    const { audioData, lpcOrder, sampleRate, downsampleFactor, graphSize } = data;

    try {
      // Perform LPC filter frequency response calculation
      const freqResponse = lpc_filter_freq_response_with_downsampling(
        audioData,
        lpcOrder,
        sampleRate,
        downsampleFactor,
        graphSize
      );

      // Post the result back to the main thread
      self.postMessage({ type: 'calcLPCFilter', status: 'success', freqResponse });
    } catch (error) {
      self.postMessage({ type: 'calcLPCFilter', status: 'error', error: error.message });
    }
  }
};
