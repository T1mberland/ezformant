import init, {
	lpc_filter_freq_response_with_downsampling,
	wasm_fourier,
} from "../pkg/webapp.js";

type WorkerRequest =
	| { type: "init" }
	| {
			type: "calcFormants";
			data: {
				audioData: number[];
				lpcOrder: number;
				sampleRate: number;
				downsampleFactor: number;
			};
	  };

type WorkerResponse =
	| { type: "init"; status: "success" | "error"; error?: string }
	| {
			type: "calcFormants";
			status: "success" | "error";
			formants?: number[];
			pitch?: number;
			error?: string;
	  };

type FormantSample = {
	time: number;
	f0: number;
	f1: number;
	f2: number;
	f3: number;
	f4: number;
};

const FFT_SIZE = 2048;
const MAX_HISTORY = 1000;
const FORMANT_ORDER = 14;
const LPC_SPECTRUM_ORDER = 16;
const DOWNSAMPLE_FACTOR = 4;

const spectrumCanvas = requireElement<HTMLCanvasElement>("spectrum");
const historyCanvas = requireElement<HTMLCanvasElement>("historyCanvas");
const ctx = get2DContext(spectrumCanvas);
const historyCtx = get2DContext(historyCanvas);

const labelF0 = requireElement<HTMLLabelElement>("label-f0");
const labelF1 = requireElement<HTMLLabelElement>("label-f1");
const labelF2 = requireElement<HTMLLabelElement>("label-f2");
const labelF3 = requireElement<HTMLLabelElement>("label-f3");

const showFFTSpectrum = requireElement<HTMLInputElement>("showFFTSpectrum");
const showLPCSpectrum = requireElement<HTMLInputElement>("showLPCSpectrum");
const showFormants = requireElement<HTMLInputElement>("showFormants");

const formantWorker = new Worker(
	new URL("./formantWorker.js", import.meta.url),
	{
		type: "module",
	},
);

const formantHistory: FormantSample[] = [];
let formant0 = 0;
let formant1 = 0;
let formant2 = 0;
let formant3 = 0;
let formant4 = 0;

function requireElement<T extends HTMLElement>(id: string): T {
	const el = document.getElementById(id);
	if (!el) {
		throw new Error(`Missing element with id ${id}`);
	}
	return el as T;
}

function get2DContext(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
	const context = canvas.getContext("2d");
	if (!context) {
		throw new Error("Failed to get 2D canvas context");
	}
	return context;
}

function addFormantsToHistory(
	f0: number,
	f1: number,
	f2: number,
	f3: number,
	f4: number,
) {
	formantHistory.push({
		time: performance.now(),
		f0,
		f1,
		f2,
		f3,
		f4,
	});

	if (formantHistory.length > MAX_HISTORY) {
		formantHistory.shift();
	}
}

function drawAxisTicks(
	context: CanvasRenderingContext2D,
	opts: {
		minTime: number;
		timeWindow: number;
		minFreq: number;
		maxFreq: number;
		xFromTime: (time: number) => number;
		yFromFreq: (freq: number) => number;
		width: number;
		height: number;
	},
) {
	const {
		minTime,
		timeWindow,
		minFreq,
		maxFreq,
		xFromTime,
		yFromFreq,
		width,
		height,
	} = opts;
	const numTimeTicks = 5;
	const numFreqTicks = 5;

	context.save();
	context.strokeStyle = "#444";
	context.fillStyle = "#fff";
	context.font = "12px sans-serif";

	for (let i = 0; i <= numTimeTicks; i += 1) {
		const fraction = i / numTimeTicks;
		const tickTime = minTime + fraction * timeWindow;
		const xPos = xFromTime(tickTime);

		context.beginPath();
		context.moveTo(xPos, 0);
		context.lineTo(xPos, height);
		context.stroke();

		const elapsedSeconds = (
			-(timeWindow - (tickTime - minTime)) / 1000
		).toFixed(1);
		context.fillText(`${elapsedSeconds}s`, xPos + 2, height - 5);
	}

	for (let i = 0; i <= numFreqTicks; i += 1) {
		const fraction = i / numFreqTicks;
		const freq = minFreq + fraction * (maxFreq - minFreq);
		const yPos = yFromFreq(freq);

		context.beginPath();
		context.moveTo(0, yPos);
		context.lineTo(width, yPos);
		context.stroke();

		context.fillText(`${freq.toFixed(0)} Hz`, 5, yPos - 5);
	}

	context.restore();
}

async function start() {
	try {
		spectrumCanvas.width = window.innerWidth;
		spectrumCanvas.height = window.innerHeight / 2;
		historyCanvas.width = spectrumCanvas.width;
		historyCanvas.height = spectrumCanvas.height;

		await init(); // load WASM before calling wasm_* helpers on the main thread

		const stream = await navigator.mediaDevices.getUserMedia({
			audio: {
				autoGainControl: false,
				noiseSuppression: false,
				echoCancellation: false,
			},
		});
		const audioContext = new AudioContext();
		await audioContext.resume();

		const source = audioContext.createMediaStreamSource(stream);
		const analyser = audioContext.createAnalyser();
		source.connect(analyser);

		analyser.fftSize = FFT_SIZE;
		const bufferLength = FFT_SIZE / 2;
		const dataArray = new Float32Array(FFT_SIZE);
		const spectrum = new Float32Array(bufferLength);
		const sampleRate = audioContext.sampleRate;

		const minFrequency = 20;
		const maxFrequency = sampleRate / 2;
		const logMin = Math.log10(minFrequency);
		const logMax = Math.log10(maxFrequency);
		const logRange = logMax - logMin;

		const frequencyToPosition = (freq: number) =>
			((Math.log10(freq) - logMin) / logRange) * spectrumCanvas.width;

		const getFrequency = (index: number) =>
			index * (maxFrequency / bufferLength);

		const calcFormants = () => {
			analyser.getFloatTimeDomainData(dataArray);
			const payload: WorkerRequest = {
				type: "calcFormants",
				data: {
					audioData: Array.from(dataArray),
					lpcOrder: FORMANT_ORDER,
					sampleRate,
					downsampleFactor: DOWNSAMPLE_FACTOR,
				},
			};
			formantWorker.postMessage(payload);
		};

		formantWorker.onmessage = (event: MessageEvent<WorkerResponse>) => {
			const message = event.data;
			if (message.type === "init") {
				if (message.status === "success") {
					setInterval(calcFormants, 100);
				} else {
					console.error("Worker initialization failed:", message.error);
					alert("Failed to initialize formant detection worker.");
				}
			} else if (message.type === "calcFormants") {
				if (
					message.status === "success" &&
					message.formants &&
					message.pitch !== undefined
				) {
					[formant1, formant2, formant3, formant4] = message.formants;
					formant0 = message.pitch;
					addFormantsToHistory(
						formant0,
						formant1,
						formant2,
						formant3,
						formant4,
					);
				} else if (message.status === "error") {
					console.error("Formant detection failed:", message.error);
				}
			}
		};

		const updateFormantText = () => {
			labelF0.textContent = `F0: ${formant0.toFixed(0)}`;
			labelF1.textContent = `F1: ${formant1.toFixed(0)}`;
			labelF2.textContent = `F2: ${formant2.toFixed(0)}`;
			labelF3.textContent = `F3: ${formant3.toFixed(0)}`;
		};

		const drawFrequencyLabels = () => {
			const labelCount = 20;
			ctx.fillStyle = "#ffffff";
			ctx.font = "12px Arial";
			ctx.textAlign = "center";
			ctx.textBaseline = "top";

			for (let i = 0; i <= labelCount; i += 1) {
				const freq = minFrequency * 10 ** ((logRange * i) / labelCount);
				const x = frequencyToPosition(freq);
				ctx.beginPath();
				ctx.moveTo(x, spectrumCanvas.height);
				ctx.lineTo(x, spectrumCanvas.height - 10);
				ctx.strokeStyle = "#ffffff";
				ctx.lineWidth = 1;
				ctx.stroke();
				ctx.fillText(`${Math.round(freq)} Hz`, x, spectrumCanvas.height - 25);
			}
		};

		const drawSpectrum = () => {
			analyser.getFloatTimeDomainData(dataArray);
			ctx.clearRect(0, 0, spectrumCanvas.width, spectrumCanvas.height);
			ctx.fillStyle = "#000";
			ctx.fillRect(0, 0, spectrumCanvas.width, spectrumCanvas.height);

			if (showFFTSpectrum.checked) {
				const spectrumData = wasm_fourier(dataArray);
				spectrum.set(spectrumData.slice(0, bufferLength));

				ctx.strokeStyle = "#00ff00";
				ctx.lineWidth = 2;
				ctx.beginPath();

				const maxMagnitude = Math.max(
					...spectrum.map((value) => Math.abs(value)),
					10,
				);

				for (let i = 0; i < bufferLength; i += 1) {
					const freq = getFrequency(i);
					if (freq < minFrequency || freq > maxFrequency) continue;

					const x = frequencyToPosition(freq);
					const magnitude = spectrum[i];

					const logMagnitude = Math.log10(Math.abs(magnitude) + 1);
					const logMaxMagnitude = Math.log10(maxMagnitude + 1);
					const y =
						spectrumCanvas.height -
						(logMagnitude / logMaxMagnitude) * spectrumCanvas.height;

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
		};

		const drawLPCFilter = () => {
			if (showLPCSpectrum.checked) {
				analyser.getFloatTimeDomainData(dataArray);

				const graphSize = 1024;
				const freqResponse = lpc_filter_freq_response_with_downsampling(
					Float64Array.from(dataArray),
					LPC_SPECTRUM_ORDER,
					sampleRate,
					DOWNSAMPLE_FACTOR,
					graphSize,
				);

				if (freqResponse.every((value) => value === 0)) {
					requestAnimationFrame(drawLPCFilter);
					return;
				}

				const maxResponse = Math.max(...freqResponse);
				const normalizeConst = maxResponse > 0 ? maxResponse : 1;

				ctx.strokeStyle = "red";
				ctx.beginPath();
				let started = false;

				for (let i = 0; i < graphSize; i += 1) {
					const freq = (i * maxFrequency) / graphSize / DOWNSAMPLE_FACTOR;
					if (freq < minFrequency) continue;

					const xPos = frequencyToPosition(freq);
					const logResponse = Math.log10(freqResponse[i] + 1);
					const logMaxResponse = Math.log10(normalizeConst + 1);
					const yPos =
						spectrumCanvas.height -
						(logResponse / logMaxResponse) * spectrumCanvas.height;

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
				const drawFormantLine = (value: number, color: string) => {
					const xPos = frequencyToPosition(value);
					ctx.strokeStyle = color;
					ctx.beginPath();
					ctx.moveTo(xPos, 0);
					ctx.lineTo(xPos, spectrumCanvas.height);
					ctx.stroke();
					ctx.fillStyle = color;
					ctx.fillText(value.toFixed(0), xPos, 0);
				};

				drawFormantLine(formant0, "#ff00ff");
				drawFormantLine(formant1, "white");
				drawFormantLine(formant2, "red");
				drawFormantLine(formant3, "green");
			}

			requestAnimationFrame(drawLPCFilter);
		};

		const drawFormantHistory = () => {
			historyCtx.clearRect(0, 0, historyCanvas.width, historyCanvas.height);
			historyCtx.fillStyle = "#000";
			historyCtx.fillRect(0, 0, historyCanvas.width, historyCanvas.height);

			const now = performance.now();
			const timeWindow = 5000;
			const minTime = now - timeWindow;
			const minFreq = 0;
			const maxFreq = 3000;
			const width = historyCanvas.width;
			const height = historyCanvas.height;

			const xFromTime = (time: number) =>
				((time - minTime) / timeWindow) * width;
			const yFromFreq = (freq: number) => {
				const fraction = (freq - minFreq) / (maxFreq - minFreq);
				return height - fraction * height;
			};

			drawAxisTicks(historyCtx, {
				minTime,
				timeWindow,
				minFreq,
				maxFreq,
				xFromTime,
				yFromFreq,
				width,
				height,
			});

			const recentData = formantHistory.filter(
				(entry) => entry.time >= minTime,
			);
			if (recentData.length >= 2) {
				const drawLine = (key: keyof FormantSample, color: string) => {
					historyCtx.strokeStyle = color;
					historyCtx.beginPath();
					recentData.forEach((entry, index) => {
						const x = xFromTime(entry.time);
						const y = yFromFreq(entry[key]);
						if (index === 0) {
							historyCtx.moveTo(x, y);
						} else {
							historyCtx.lineTo(x, y);
						}
					});
					historyCtx.stroke();
				};

				drawLine("f0", "#ff00ff");
				drawLine("f1", "#ff0000");
				drawLine("f2", "#00ff00");
				drawLine("f3", "#0000ff");
			}

			requestAnimationFrame(drawFormantHistory);
		};

		formantWorker.postMessage({ type: "init" } as WorkerRequest);
		drawFormantHistory();
		drawSpectrum();
		drawLPCFilter();
	} catch (error) {
		console.error("Error accessing audio stream:", error);
		alert("Could not access the microphone. Please check your permissions.");
	}
}

requireElement<HTMLButtonElement>("showSpectrumTabBtn").addEventListener(
	"click",
	() => {
		spectrumCanvas.style.display = "block";
		historyCanvas.style.display = "none";
	},
);

requireElement<HTMLButtonElement>("showHistoryTabBtn").addEventListener(
	"click",
	() => {
		spectrumCanvas.style.display = "none";
		historyCanvas.style.display = "block";
	},
);

void start();
