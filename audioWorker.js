import init, { process_audio, lpc_filter_freq_response, formant_detection } from './pkg/ezformant.js';

let sampleRate = 44100; // Default sample rate, will be updated

self.onmessage = async function(event) {
  const { type, data } = event.data;

  if (type === 'INIT') {
    try {
      await init(); // Initialize the WASM module
      sampleRate = data.sampleRate;
      self.postMessage({ type: 'INIT_DONE' });
    } catch (error) {
      self.postMessage({ type: 'ERROR', message: error.message });
    }
  }

  if (type === 'PROCESS_AUDIO') {
    try {
      const audioData = data.audioData;
      const spectrumData = process_audio(audioData);
      const freqResponse = lpc_filter_freq_response(audioData, 16, sampleRate, 1024);
      const formants = formant_detection(audioData, 14, sampleRate / 4); // Assuming downsample factor 4

      self.postMessage({
        type: 'PROCESSED_DATA',
        spectrum: spectrumData.slice(0, 1024),
        freqResponse: freqResponse,
        formants: formants.slice(0, 4)
      }, [
        spectrumData.buffer,
        freqResponse.buffer,
        formants.buffer
      ]);
    } catch (error) {
      self.postMessage({ type: 'ERROR', message: error.message });
    }
  }
};
