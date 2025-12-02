import init, { formant_detection_with_downsampling, pitch_detection } from "../pkg/webapp.js";

type InitMessage = { type: "init" };
type CalcMessage = {
  type: "calcFormants";
  data: {
    audioData: number[];
    lpcOrder: number;
    sampleRate: number;
    downsampleFactor: number;
  };
};

type WorkerMessage = InitMessage | CalcMessage;

type WorkerResponse =
  | { type: "init"; status: "success" | "error"; error?: string }
  | {
      type: "calcFormants";
      status: "success" | "error";
      formants?: Float64Array;
      pitch?: number;
      error?: string;
    };

const workerScope: DedicatedWorkerGlobalScope = self as DedicatedWorkerGlobalScope;
let wasmInitialized = false;

async function ensureWasmLoaded() {
  if (!wasmInitialized) {
    await init();
    wasmInitialized = true;
  }
}

workerScope.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const message = event.data;

  if (message.type === "init") {
    try {
      await ensureWasmLoaded();
      workerScope.postMessage({ type: "init", status: "success" } satisfies WorkerResponse);
    } catch (error) {
      workerScope.postMessage({
        type: "init",
        status: "error",
        error: error instanceof Error ? error.message : String(error),
      } satisfies WorkerResponse);
    }
  }

  if (message.type === "calcFormants") {
    try {
      await ensureWasmLoaded();
      const input = Float64Array.from(message.data.audioData);

      const formants = formant_detection_with_downsampling(
        input,
        message.data.lpcOrder,
        message.data.sampleRate,
        message.data.downsampleFactor,
      );
      const pitch = pitch_detection(input, message.data.sampleRate);

      workerScope.postMessage({
        type: "calcFormants",
        status: "success",
        formants,
        pitch,
      } satisfies WorkerResponse);
    } catch (error) {
      workerScope.postMessage({
        type: "calcFormants",
        status: "error",
        error: error instanceof Error ? error.message : String(error),
      } satisfies WorkerResponse);
    }
  }
};
