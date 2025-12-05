import { useEffect, useRef, useState } from "react";

type WasmBindings = typeof import("../../pkg/webapp.js");

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
const NOTE_NAMES = [
	"C",
	"C#",
	"D",
	"D#",
	"E",
	"F",
	"F#",
	"G",
	"G#",
	"A",
	"A#",
	"B",
];

function frequencyToNoteName(freq: number): string {
	if (!Number.isFinite(freq) || freq <= 0) return "—";
	const midi = Math.round(69 + 12 * Math.log2(freq / 440));
	const name = NOTE_NAMES[((midi % 12) + 12) % 12];
	const octave = Math.floor(midi / 12) - 1;
	return `${name}${octave}`;
}

export default function App() {
	const [activeView, setActiveView] = useState<"spectrum" | "history">(
		"spectrum",
	);
	const [showFFTSpectrum, setShowFFTSpectrum] = useState(true);
	const [showLPCSpectrum, setShowLPCSpectrum] = useState(true);
	const [showFormants, setShowFormants] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [formants, setFormants] = useState({
		f0: 0,
		f1: 0,
		f2: 0,
		f3: 0,
		f4: 0,
	});

	const spectrumCanvasRef = useRef<HTMLCanvasElement | null>(null);
	const historyCanvasRef = useRef<HTMLCanvasElement | null>(null);
	const historyRef = useRef<FormantSample[]>([]);
	const formantsRef = useRef(formants);
	const freqBoundsRef = useRef({
		minFrequency: 20,
		maxFrequency: 22050,
		logRange: Math.log10(22050) - Math.log10(20),
	});

	const showFFTSpectrumRef = useRef(showFFTSpectrum);
	const showLPCSpectrumRef = useRef(showLPCSpectrum);
	const showFormantsRef = useRef(showFormants);

	useEffect(() => {
		showFFTSpectrumRef.current = showFFTSpectrum;
	}, [showFFTSpectrum]);
	useEffect(() => {
		showLPCSpectrumRef.current = showLPCSpectrum;
	}, [showLPCSpectrum]);
	useEffect(() => {
		showFormantsRef.current = showFormants;
	}, [showFormants]);
	useEffect(() => {
		formantsRef.current = formants;
	}, [formants]);

	useEffect(() => {
		const spectrumCanvas = spectrumCanvasRef.current;
		const historyCanvas = historyCanvasRef.current;
		if (!spectrumCanvas || !historyCanvas) return;

		const history = historyRef.current;
		let rafSpectrum: number | null = null;
		let rafLpc: number | null = null;
		let rafHistory: number | null = null;
		let formantInterval: number | null = null;
		let worker: Worker | null = null;
		let audioContext: AudioContext | null = null;
		let analyser: AnalyserNode | null = null;
		let dataArray: Float32Array | null = null;
		let spectrum: Float32Array | null = null;
		let stream: MediaStream | null = null;
		let wasm: WasmBindings | null = null;

		const resize = () => {
			spectrumCanvas.width = window.innerWidth;
			spectrumCanvas.height = Math.max(320, window.innerHeight * 0.5);
			historyCanvas.width = spectrumCanvas.width;
			historyCanvas.height = spectrumCanvas.height;
		};

		const addFormantsToHistory = (sample: FormantSample) => {
			history.push(sample);
			if (history.length > MAX_HISTORY) history.shift();
		};

		const setup = async () => {
			try {
				resize();
				window.addEventListener("resize", resize);

				const wasmUrl = new URL("../../pkg/webapp.js", import.meta.url).href;
				wasm = (await import(/* @vite-ignore */ wasmUrl)) as WasmBindings;
				await wasm.default();

				stream = await navigator.mediaDevices.getUserMedia({
					audio: {
						autoGainControl: false,
						noiseSuppression: false,
						echoCancellation: false,
					},
				});

				audioContext = new AudioContext();
				await audioContext.resume();
				const source = audioContext.createMediaStreamSource(stream);
				analyser = audioContext.createAnalyser();
				analyser.fftSize = FFT_SIZE;
				source.connect(analyser);

				freqBoundsRef.current = {
					minFrequency: 20,
					maxFrequency: audioContext.sampleRate / 2,
					logRange: Math.log10(audioContext.sampleRate / 2) - Math.log10(20),
				};

				dataArray = new Float32Array(FFT_SIZE);
				spectrum = new Float32Array(FFT_SIZE / 2);

				worker = new Worker(new URL("../formantWorker.ts", import.meta.url), {
					type: "module",
				});

				const calcFormants = () => {
					if (!analyser || !dataArray || !audioContext || !worker) return;
					analyser.getFloatTimeDomainData(dataArray);
					const payload: WorkerRequest = {
						type: "calcFormants",
						data: {
							audioData: Array.from(dataArray),
							lpcOrder: FORMANT_ORDER,
							sampleRate: audioContext.sampleRate,
							downsampleFactor: DOWNSAMPLE_FACTOR,
						},
					};
					worker.postMessage(payload);
				};

				worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
					const message = event.data;
					if (message.type === "init") {
						if (message.status === "success") {
							formantInterval = window.setInterval(calcFormants, 100);
						} else {
							setError("Failed to initialize formant worker");
						}
						return;
					}

					if (
						message.type === "calcFormants" &&
						message.status === "success" &&
						message.formants &&
						message.pitch !== undefined
					) {
						const [f1, f2, f3, f4] = message.formants;
						setFormants({ f0: message.pitch, f1, f2, f3, f4 });
						addFormantsToHistory({
							time: performance.now(),
							f0: message.pitch,
							f1,
							f2,
							f3,
							f4,
						});
					} else if (
						message.type === "calcFormants" &&
						message.status === "error"
					) {
						setError(message.error ?? "Formant detection failed");
					}
				};

				worker.postMessage({ type: "init" } satisfies WorkerRequest);

				const drawSpectrum = () => {
					if (!analyser || !dataArray || !spectrum || !wasm) return;
					const ctx = spectrumCanvas.getContext("2d");
					if (!ctx) return;

					analyser.getFloatTimeDomainData(dataArray);
					ctx.clearRect(0, 0, spectrumCanvas.width, spectrumCanvas.height);
					ctx.fillStyle = "#f7f3ec";
					ctx.fillRect(0, 0, spectrumCanvas.width, spectrumCanvas.height);

					const { minFrequency, maxFrequency, logRange } =
						freqBoundsRef.current;
					const logMin = Math.log10(minFrequency);
					const bufferLength = spectrum.length;

					const frequencyToPosition = (freq: number) =>
						((Math.log10(freq) - logMin) / logRange) * spectrumCanvas.width;

					if (showFFTSpectrumRef.current) {
						const spectrumData = wasm.wasm_fourier(dataArray);
						spectrum.set(spectrumData.slice(0, bufferLength));

						ctx.strokeStyle = "#f26b38";
						ctx.lineWidth = 2;
						ctx.beginPath();

						const maxMagnitude = Math.max(
							...spectrum.map((value) => Math.abs(value)),
							10,
						);

						for (let i = 0; i < bufferLength; i += 1) {
							const freq = (i * maxFrequency) / bufferLength;
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

					const labelCount = 16;
					ctx.fillStyle = "#5e5247";
					ctx.font = "12px 'Soehne', 'Inter', sans-serif";
					ctx.textAlign = "center";
					ctx.textBaseline = "top";

					for (let i = 0; i <= labelCount; i += 1) {
						const freq =
							minFrequency *
							10 ** ((freqBoundsRef.current.logRange * i) / labelCount);
						const x = frequencyToPosition(freq);
						ctx.beginPath();
						ctx.moveTo(x, spectrumCanvas.height);
						ctx.lineTo(x, spectrumCanvas.height - 8);
						ctx.strokeStyle = "#c7bcad";
						ctx.lineWidth = 1;
						ctx.stroke();
						ctx.fillText(
							`${Math.round(freq)} Hz`,
							x,
							spectrumCanvas.height - 22,
						);
					}

					rafSpectrum = requestAnimationFrame(drawSpectrum);
				};

				const drawLPCFilter = () => {
					if (!analyser || !dataArray || !wasm) return;
					const ctx = spectrumCanvas.getContext("2d");
					if (!ctx) return;

					const { minFrequency, maxFrequency, logRange } =
						freqBoundsRef.current;
					const logMin = Math.log10(minFrequency);

					const frequencyToPosition = (freq: number) =>
						((Math.log10(freq) - logMin) / logRange) * spectrumCanvas.width;

					if (showLPCSpectrumRef.current) {
						analyser.getFloatTimeDomainData(dataArray);
						const graphSize = 1024;
						const freqResponse =
							wasm.lpc_filter_freq_response_with_downsampling(
								Float64Array.from(dataArray),
								LPC_SPECTRUM_ORDER,
								audioContext?.sampleRate ?? 44100,
								DOWNSAMPLE_FACTOR,
								graphSize,
							);

						const maxResponse = Math.max(...freqResponse);
						const normalizeConst = maxResponse > 0 ? maxResponse : 1;

						ctx.strokeStyle = "#2f6b4f";
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

					if (showFormantsRef.current) {
						const renderLine = (
							value: number,
							color: string,
							label: string,
						) => {
							const xPos = frequencyToPosition(value);
							ctx.strokeStyle = color;
							ctx.beginPath();
							ctx.moveTo(xPos, 0);
							ctx.lineTo(xPos, spectrumCanvas.height);
							ctx.stroke();
							ctx.fillStyle = color;
							ctx.fillText(label, xPos, 8);
						};

						const current = formantsRef.current;
						renderLine(
							current.f0,
							"#c24d2c",
							`${current.f0.toFixed(0)} (${frequencyToNoteName(current.f0)})`,
						);
						renderLine(current.f1, "#1f3f58", current.f1.toFixed(0));
						renderLine(current.f2, "#d06c3e", current.f2.toFixed(0));
						renderLine(current.f3, "#2f6b4f", current.f3.toFixed(0));
					}

					rafLpc = requestAnimationFrame(drawLPCFilter);
				};

				const drawFormantHistory = () => {
					const canvas = historyCanvasRef.current;
					if (!canvas) return;
					const ctx = canvas.getContext("2d");
					if (!ctx) return;

					ctx.clearRect(0, 0, canvas.width, canvas.height);
					ctx.fillStyle = "#f7f3ec";
					ctx.fillRect(0, 0, canvas.width, canvas.height);

					const now = performance.now();
					const timeWindow = 5000;
					const minTime = now - timeWindow;
					const minFreq = 0;
					const maxFreq = 3000;
					const width = canvas.width;
					const height = canvas.height;

					const xFromTime = (time: number) =>
						((time - minTime) / timeWindow) * width;
					const yFromFreq = (freq: number) => {
						const fraction = (freq - minFreq) / (maxFreq - minFreq);
						return height - fraction * height;
					};

					const numTimeTicks = 5;
					const numFreqTicks = 5;

					ctx.save();
					ctx.strokeStyle = "#d7ccbe";
					ctx.fillStyle = "#5e5247";
					ctx.font = "12px 'Soehne', 'Inter', sans-serif";

					for (let i = 0; i <= numTimeTicks; i += 1) {
						const fraction = i / numTimeTicks;
						const tickTime = minTime + fraction * timeWindow;
						const xPos = xFromTime(tickTime);
						ctx.beginPath();
						ctx.moveTo(xPos, 0);
						ctx.lineTo(xPos, height);
						ctx.stroke();
						const elapsedSeconds = (
							-(timeWindow - (tickTime - minTime)) / 1000
						).toFixed(1);
						ctx.fillText(`${elapsedSeconds}s`, xPos + 2, height - 5);
					}

					for (let i = 0; i <= numFreqTicks; i += 1) {
						const fraction = i / numFreqTicks;
						const freq = minFreq + fraction * (maxFreq - minFreq);
						const yPos = yFromFreq(freq);
						ctx.beginPath();
						ctx.moveTo(0, yPos);
						ctx.lineTo(width, yPos);
						ctx.stroke();
						ctx.fillText(`${freq.toFixed(0)} Hz`, 6, yPos - 5);
					}

					ctx.restore();

					const recent = history.filter((entry) => entry.time >= minTime);
					if (recent.length >= 2) {
						const drawLine = (key: keyof FormantSample, color: string) => {
							ctx.strokeStyle = color;
							ctx.beginPath();
							recent.forEach((entry, index) => {
								const x = xFromTime(entry.time);
								const y = yFromFreq(entry[key]);
								if (index === 0) {
									ctx.moveTo(x, y);
								} else {
									ctx.lineTo(x, y);
								}
							});
							ctx.stroke();
						};

						drawLine("f0", "#c24d2c");
						drawLine("f1", "#1f3f58");
						drawLine("f2", "#d06c3e");
						drawLine("f3", "#2f6b4f");
					}

					rafHistory = requestAnimationFrame(drawFormantHistory);
				};

				rafSpectrum = requestAnimationFrame(drawSpectrum);
				rafLpc = requestAnimationFrame(drawLPCFilter);
				rafHistory = requestAnimationFrame(drawFormantHistory);
			} catch (err) {
				const message = err instanceof Error ? err.message : String(err);
				setError(message);
			}
		};

		setup();

		return () => {
			if (rafSpectrum) cancelAnimationFrame(rafSpectrum);
			if (rafLpc) cancelAnimationFrame(rafLpc);
			if (rafHistory) cancelAnimationFrame(rafHistory);
			if (formantInterval !== null) clearInterval(formantInterval);
			if (worker) worker.terminate();
			if (stream) {
				stream.getTracks().forEach((track) => {
					track.stop();
				});
			}
			if (audioContext) audioContext.close();
			window.removeEventListener("resize", resize);
		};
	}, []);

	return (
		<div className="page">
			<header className="topbar">
				<div>
					<div className="eyebrow">EZ Formant</div>
					<h1>Formant & Pitch Explorer</h1>
					<p className="lede">
						Live spectrum, LPC envelope, and formant tracks powered by WebAudio
						+ WASM.
					</p>
				</div>
				<div className="badges">
					<span className="badge">WASM</span>
					<span className="badge">React</span>
					<span className="badge">Real-time</span>
				</div>
			</header>

			<section className="controls">
				<div className="segmented">
					<button
						type="button"
						className={activeView === "spectrum" ? "active" : ""}
						onClick={() => setActiveView("spectrum")}
					>
						Real-time Spectrum
					</button>
					<button
						type="button"
						className={activeView === "history" ? "active" : ""}
						onClick={() => setActiveView("history")}
					>
						Formant History
					</button>
				</div>
				<div className="toggles">
					<label className="toggle">
						<input
							type="checkbox"
							checked={showFFTSpectrum}
							onChange={(e) => setShowFFTSpectrum(e.target.checked)}
						/>
						<span>FFT spectrum</span>
					</label>
					<label className="toggle">
						<input
							type="checkbox"
							checked={showLPCSpectrum}
							onChange={(e) => setShowLPCSpectrum(e.target.checked)}
						/>
						<span>LPC envelope</span>
					</label>
					<label className="toggle">
						<input
							type="checkbox"
							checked={showFormants}
							onChange={(e) => setShowFormants(e.target.checked)}
						/>
						<span>Formant markers</span>
					</label>
				</div>
			</section>

			<section className="readout">
				<div className="metric primary">
					<div className="label">Pitch (F0)</div>
					<div className="value">
						{formants.f0.toFixed(0)} Hz{" "}
						<span className="note">{frequencyToNoteName(formants.f0)}</span>
					</div>
				</div>
				<div className="metric">
					<div className="label">F1</div>
					<div className="value">{formants.f1.toFixed(0)} Hz</div>
				</div>
				<div className="metric">
					<div className="label">F2</div>
					<div className="value">{formants.f2.toFixed(0)} Hz</div>
				</div>
				<div className="metric">
					<div className="label">F3</div>
					<div className="value">{formants.f3.toFixed(0)} Hz</div>
				</div>
				<div className="metric">
					<div className="label">F4</div>
					<div className="value">{formants.f4.toFixed(0)} Hz</div>
				</div>
			</section>

			<div className="canvas-shell">
				<canvas
					ref={spectrumCanvasRef}
					className={`canvas ${activeView === "spectrum" ? "visible" : "hidden"}`}
				/>
				<canvas
					ref={historyCanvasRef}
					className={`canvas ${activeView === "history" ? "visible" : "hidden"}`}
				/>
			</div>

			{error ? (
				<div className="error">Microphone or processing error: {error}</div>
			) : null}
		</div>
	);
}
