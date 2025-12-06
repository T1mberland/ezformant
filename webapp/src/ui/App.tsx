import {
	type ChangeEvent,
	useCallback,
	useEffect,
	useRef,
	useState,
} from "react";

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

type TrainingMode = "off" | "pitch" | "vowel";

type VowelTarget = {
	id: string;
	label: string;
	example: string;
	f1: number;
	f2: number;
};

type PitchTarget = {
	id: string;
	label: string;
	freq: number;
};

type InputMode = "mic" | "file";

type LpcPresetId = "speech" | "singer" | "noisy";

type LpcPreset = {
	id: LpcPresetId;
	label: string;
	description: string;
	formantOrder: number;
	spectrumOrder: number;
	downsampleFactor: number;
};

type DebugInfo = {
	windowSize: number | null;
	sampleRate: number | null;
	frameIntervalMs: number;
	windowDurationMs: number | null;
	lastFrameComputeMs: number | null;
	lpcCoefficients: number[] | null;
};

const FFT_SIZE = 2048;
const MAX_HISTORY = 1000;
const FORMANT_INTERVAL_MS = 100;
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

const VOWEL_TARGETS: VowelTarget[] = [
	{
		id: "i",
		label: "/i/",
		example: "heed",
		f1: 300,
		f2: 2300,
	},
	{
		id: "e",
		label: "/e/",
		example: "bed",
		f1: 400,
		f2: 2000,
	},
	{
		id: "a",
		label: "/a/",
		example: "father",
		f1: 700,
		f2: 1100,
	},
	{
		id: "o",
		label: "/o/",
		example: "thought",
		f1: 500,
		f2: 900,
	},
	{
		id: "u",
		label: "/u/",
		example: "food",
		f1: 350,
		f2: 800,
	},
];

const PITCH_TARGETS: PitchTarget[] = [
	{ id: "G3", label: "G3 (196 Hz)", freq: 196 },
	{ id: "A3", label: "A3 (220 Hz)", freq: 220 },
	{ id: "C4", label: "C4 (261 Hz)", freq: 261.63 },
	{ id: "E4", label: "E4 (329 Hz)", freq: 329.63 },
	{ id: "G4", label: "G4 (392 Hz)", freq: 392 },
	{ id: "A4", label: "A4 (440 Hz)", freq: 440 },
];

const LPC_PRESETS: LpcPreset[] = [
	{
		id: "speech",
		label: "Speech (default)",
		description: "Balanced for typical spoken vowels.",
		formantOrder: 14,
		spectrumOrder: 16,
		downsampleFactor: 4,
	},
	{
		id: "singer",
		label: "Singer (detailed)",
		description: "Higher order, less downsampling for rich harmonics.",
		formantOrder: 18,
		spectrumOrder: 20,
		downsampleFactor: 2,
	},
	{
		id: "noisy",
		label: "Noisy (robust)",
		description: "Smoother response for noisy environments.",
		formantOrder: 10,
		spectrumOrder: 12,
		downsampleFactor: 6,
	},
];

function frequencyToNoteName(freq: number): string {
	if (!Number.isFinite(freq) || freq <= 0) return "—";
	const midi = Math.round(69 + 12 * Math.log2(freq / 440));
	const name = NOTE_NAMES[((midi % 12) + 12) % 12];
	const octave = Math.floor(midi / 12) - 1;
	return `${name}${octave}`;
}

function formatDuration(seconds: number | null): string {
	if (!Number.isFinite(seconds) || seconds === null) return "—";
	const wholeSeconds = Math.round(seconds);
	const mins = Math.floor(wholeSeconds / 60);
	const secs = wholeSeconds % 60;
	return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function subtractMeanInPlace(data: Float64Array): void {
	const n = data.length;
	if (n === 0) return;
	let sum = 0;
	for (let i = 0; i < n; i += 1) {
		sum += data[i];
	}
	const mean = sum / n;
	for (let i = 0; i < n; i += 1) {
		data[i] -= mean;
	}
}

function applyHammingWindowInPlace(data: Float64Array): void {
	const n = data.length;
	if (n === 0) return;
	const last = n - 1;
	for (let i = 0; i < n; i += 1) {
		const ratio = i / last;
		const w = 0.54 - 0.46 * Math.cos(2 * Math.PI * ratio);
		data[i] *= w;
	}
}

function preEmphasizeInPlace(data: Float64Array, alpha: number): void {
	const n = data.length;
	if (n === 0) return;
	let prev = data[0];
	data[0] = (1 - alpha) * prev;
	for (let i = 1; i < n; i += 1) {
		const x = data[i];
		data[i] = x - alpha * prev;
		prev = x;
	}
}

function preprocessSignalForLpc(data: Float64Array, alpha: number): void {
	subtractMeanInPlace(data);
	applyHammingWindowInPlace(data);
	preEmphasizeInPlace(data, alpha);
}

function autocorrelateForLpc(
	signal: Float64Array,
	maxLag: number,
): Float64Array {
	const n = signal.length;
	const result = new Float64Array(maxLag + 1);
	for (let lag = 0; lag <= maxLag; lag += 1) {
		let acc = 0;
		for (let i = 0; i + lag < n; i += 1) {
			acc += signal[i] * signal[i + lag];
		}
		result[lag] = acc;
	}
	return result;
}

function levinsonForLpc(order: number, r: Float64Array): number[] {
	if (r.length < order + 1) {
		throw new Error("autocorrelation too short for requested LPC order");
	}
	const a = new Float64Array(order + 1);
	a[0] = 1;

	let e = Math.abs(r[0]) < 1e-12 ? 1e-12 : r[0];

	for (let i = 1; i <= order; i += 1) {
		let acc = r[i];
		for (let j = 1; j < i; j += 1) {
			acc += a[j] * r[i - j];
		}
		const k = -acc / e;
		const aNext = new Float64Array(a);
		for (let j = 1; j < i; j += 1) {
			aNext[j] = a[j] + k * a[i - j];
		}
		aNext[i] = k;
		a.set(aNext);

		e *= 1 - k * k;
		if (e < 1e-12) e = 1e-12;
	}

	return Array.from(a);
}

function computeLpcCoefficientsForDebug(
	input: Float32Array,
	order: number,
	downsampleFactor: number,
): number[] {
	if (!Number.isFinite(order) || order <= 0) return [];
	if (downsampleFactor <= 0) downsampleFactor = 1;
	if (input.length === 0) return [];

	// Downsample to roughly match the Rust pipeline.
	const downsampledLength = Math.max(
		Math.ceil(input.length / downsampleFactor),
		1,
	);
	const downsampled = new Float64Array(downsampledLength);
	for (let i = 0, j = 0; i < input.length && j < downsampledLength; i += downsampleFactor, j += 1) {
		downsampled[j] = input[i];
	}

	preprocessSignalForLpc(downsampled, 0.97);
	const r = autocorrelateForLpc(downsampled, order);

	try {
		return levinsonForLpc(order, r);
	} catch {
		return [];
	}
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
	const [trainingMode, setTrainingMode] = useState<TrainingMode>("off");
	const [selectedVowelId, setSelectedVowelId] = useState<string>("i");
	const [selectedPitchId, setSelectedPitchId] = useState<string>("A3");
	const [manualPitchHz, setManualPitchHz] = useState<number>(220);
	const [manualVowelF1, setManualVowelF1] = useState<number>(500);
	const [manualVowelF2, setManualVowelF2] = useState<number>(1500);
	const [isFrozen, setIsFrozen] = useState(false);
	const [inputMode, setInputMode] = useState<InputMode>("mic");
	const [theme, setTheme] = useState<"light" | "dark">("light");
	const [lpcPresetId, setLpcPresetId] = useState<LpcPresetId>("speech");
	const [fileStatus, setFileStatus] = useState<
		"idle" | "loading" | "playing" | "paused" | "ended" | "error"
	>("idle");
	const [fileName, setFileName] = useState("");
	const [fileDuration, setFileDuration] = useState<number | null>(null);
	const [fileError, setFileError] = useState<string | null>(null);
	const [filePosition, setFilePosition] = useState(0);
	const [isScrubbing, setIsScrubbing] = useState(false);
	const [debugPanelOpen, setDebugPanelOpen] = useState(false);
	const [debugInfo, setDebugInfo] = useState<DebugInfo>({
		windowSize: null,
		sampleRate: null,
		frameIntervalMs: FORMANT_INTERVAL_MS,
		windowDurationMs: null,
		lastFrameComputeMs: null,
		lpcCoefficients: null,
	});
	const [snapshotError, setSnapshotError] = useState<string | null>(null);
	const [showDeveloperUi] = useState(
		() =>
			typeof window !== "undefined" &&
			new URLSearchParams(window.location.search).has("dev"),
	);

	const spectrumCanvasRef = useRef<HTMLCanvasElement | null>(null);
	const historyCanvasRef = useRef<HTMLCanvasElement | null>(null);
	const historyRef = useRef<FormantSample[]>([]);
	const formantsRef = useRef(formants);
	const freqBoundsRef = useRef({
		minFrequency: 20,
		maxFrequency: 22050,
		logRange: Math.log10(22050) - Math.log10(20),
	});

	const audioContextRef = useRef<AudioContext | null>(null);
	const analyserRef = useRef<AnalyserNode | null>(null);
	const dataArrayRef = useRef<Float32Array | null>(null);
	const spectrumRef = useRef<Float32Array | null>(null);
	const wasmRef = useRef<WasmBindings | null>(null);
	const workerRef = useRef<Worker | null>(null);
	const formantIntervalRef = useRef<number | null>(null);
	const rafSpectrumRef = useRef<number | null>(null);
	const rafLpcRef = useRef<number | null>(null);
	const rafHistoryRef = useRef<number | null>(null);
	const analysisRunningRef = useRef(true);
	const micStreamRef = useRef<MediaStream | null>(null);
	const micSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
	const fileSourceRef = useRef<AudioBufferSourceNode | null>(null);
	const fileBufferRef = useRef<AudioBuffer | null>(null);
	const startMicInputRef = useRef<() => Promise<void> | null>(null);
	const startFilePlaybackRef =
		useRef<(file: File) => Promise<void> | null>(null);
	const filePlaybackStartRef = useRef<number | null>(null);
	const fileProgressRafRef = useRef<number | null>(null);
	const fileStatusRef = useRef(fileStatus);
	const isScrubbingRef = useRef(isScrubbing);
	const filePositionRef = useRef(0);
	const lpcPresetRef = useRef<LpcPreset>(LPC_PRESETS[0]);
	const startBufferPlaybackRef = useRef<
		((buffer: AudioBuffer, offset?: number) => Promise<void>) | null
	>(null);

	const showFFTSpectrumRef = useRef(showFFTSpectrum);
	const showLPCSpectrumRef = useRef(showLPCSpectrum);
	const showFormantsRef = useRef(showFormants);
	const isFrozenRef = useRef(isFrozen);
	const pitchTargetRef = useRef<number | null>(null);
	const vowelTargetRef = useRef<{ f1: number | null; f2: number | null }>({
		f1: null,
		f2: null,
	});
	const trainingModeRef = useRef<TrainingMode>("off");
	const selectedPitchIdRef = useRef(selectedPitchId);
	const selectedVowelIdRef = useRef(selectedVowelId);
	const debugEnabledRef = useRef(false);

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
		isFrozenRef.current = isFrozen;
	}, [isFrozen]);
	useEffect(() => {
		fileStatusRef.current = fileStatus;
	}, [fileStatus]);
	useEffect(() => {
		isScrubbingRef.current = isScrubbing;
	}, [isScrubbing]);
	useEffect(() => {
		filePositionRef.current = filePosition;
	}, [filePosition]);
	useEffect(() => {
		document.body.dataset.theme = theme;
	}, [theme]);
	useEffect(() => {
		trainingModeRef.current = trainingMode;
	}, [trainingMode]);
	useEffect(() => {
		selectedPitchIdRef.current = selectedPitchId;
	}, [selectedPitchId]);
	useEffect(() => {
		selectedVowelIdRef.current = selectedVowelId;
	}, [selectedVowelId]);
	useEffect(() => {
		debugEnabledRef.current = debugPanelOpen;
	}, [debugPanelOpen]);
	useEffect(() => {
		const preset =
			LPC_PRESETS.find((candidate) => candidate.id === lpcPresetId) ??
			LPC_PRESETS[0];
		lpcPresetRef.current = preset;
	}, [lpcPresetId]);
	useEffect(() => {
		// Default to mic mode on load so the "Use mic" control is selected.
		setInputMode("mic");
		setFileStatus("idle");
		setFileError(null);
		setFileName("");
		setFileDuration(null);
		setFilePosition(0);
	}, []);

	useEffect(() => {
		const startOnGesture = () => {
			if (inputMode !== "mic") return;
			if (micStreamRef.current) return;
			if (startMicInputRef.current) {
				void startMicInputRef.current();
			}
		};

		window.addEventListener("pointerdown", startOnGesture, { once: true });
		window.addEventListener("keydown", startOnGesture, { once: true });

		return () => {
			window.removeEventListener("pointerdown", startOnGesture);
			window.removeEventListener("keydown", startOnGesture);
		};
	}, [inputMode]);
	useEffect(() => {
		if (fileStatus === "ended" && fileProgressRafRef.current !== null) {
			cancelAnimationFrame(fileProgressRafRef.current);
			fileProgressRafRef.current = null;
		}
	}, [fileStatus]);
	useEffect(() => {
		let pitchTarget: number | null = null;
		if (trainingMode === "pitch") {
			if (selectedPitchId === "custom") {
				pitchTarget = manualPitchHz;
			} else {
				const preset = PITCH_TARGETS.find(
					(pitch) => pitch.id === selectedPitchId,
				);
				pitchTarget = preset?.freq ?? null;
			}
		}
		pitchTargetRef.current = pitchTarget;
	}, [trainingMode, selectedPitchId, manualPitchHz]);
	useEffect(() => {
		let f1: number | null = null;
		let f2: number | null = null;
		if (trainingMode === "vowel") {
			if (selectedVowelId === "custom") {
				f1 = manualVowelF1;
				f2 = manualVowelF2;
			} else {
				const preset = VOWEL_TARGETS.find(
					(vowel) => vowel.id === selectedVowelId,
				);
				f1 = preset?.f1 ?? null;
				f2 = preset?.f2 ?? null;
			}
		}
		vowelTargetRef.current = { f1, f2 };
	}, [trainingMode, selectedVowelId, manualVowelF1, manualVowelF2]);
	useEffect(() => {
		const spectrumCanvas = spectrumCanvasRef.current;
		const historyCanvas = historyCanvasRef.current;
		if (!spectrumCanvas || !historyCanvas) return;

		const history = historyRef.current;

		const resize = () => {
			const width = spectrumCanvas.clientWidth * window.devicePixelRatio;
			const height = 360 * window.devicePixelRatio;

			spectrumCanvas.width = width;
			spectrumCanvas.height = height;
			historyCanvas.width = width;
			historyCanvas.height = height;
		};

		const addFormantsToHistory = (sample: Omit<FormantSample, "time">) => {
			const lastTime =
				history.length > 0 ? history[history.length - 1].time : 0;
			const nextTime =
				history.length === 0 ? 0 : lastTime + FORMANT_INTERVAL_MS;
			const withTime: FormantSample = {
				time: nextTime,
				f0: sample.f0,
				f1: sample.f1,
				f2: sample.f2,
				f3: sample.f3,
				f4: sample.f4,
			};
			history.push(withTime);
			if (history.length > MAX_HISTORY) history.shift();
		};

		const handlePointerDown = (event: PointerEvent) => {
			event.preventDefault();
			toggleAnalysisFreeze();
		};

		const handleHistoryPointerDown = (event: PointerEvent) => {
			event.preventDefault();
			toggleAnalysisFreeze();
		};

		const handlePointerMove = (_event: PointerEvent) => {};

		const handlePointerUp = (_event: PointerEvent) => {};

		spectrumCanvas.addEventListener("pointerdown", handlePointerDown);
		historyCanvas.addEventListener("pointerdown", handleHistoryPointerDown);
		window.addEventListener("pointermove", handlePointerMove);
		window.addEventListener("pointerup", handlePointerUp);

		const stopCurrentInput = () => {
			if (fileProgressRafRef.current !== null) {
				cancelAnimationFrame(fileProgressRafRef.current);
				fileProgressRafRef.current = null;
			}
			if (fileSourceRef.current) {
				try {
					fileSourceRef.current.onended = null;
					fileSourceRef.current.stop();
				} catch {
					// ignore
				}
				try {
					fileSourceRef.current.disconnect();
				} catch {
					// ignore
				}
				fileSourceRef.current = null;
			}
			if (micSourceRef.current) {
				try {
					micSourceRef.current.disconnect();
				} catch {
					// ignore
				}
				micSourceRef.current = null;
			}
			if (micStreamRef.current) {
				micStreamRef.current.getTracks().forEach((track) => {
					track.stop();
				});
				micStreamRef.current = null;
			}
			filePlaybackStartRef.current = null;
		};

		const ensureAudioBackend = async () => {
			if (!audioContextRef.current) {
				const audioContext = new AudioContext();
				audioContextRef.current = audioContext;
				const analyser = audioContext.createAnalyser();
				analyser.fftSize = FFT_SIZE;
				analyserRef.current = analyser;
				dataArrayRef.current = new Float32Array(FFT_SIZE);
				spectrumRef.current = new Float32Array(FFT_SIZE / 2);
				freqBoundsRef.current = {
					minFrequency: 20,
					maxFrequency: audioContext.sampleRate / 2,
					logRange: Math.log10(audioContext.sampleRate / 2) - Math.log10(20),
				};
			}
			try {
				await audioContextRef.current?.resume();
			} catch {
				// ignore resume errors
			}
		};

		const startMicInput = async () => {
			await ensureAudioBackend();
			const audioContext = audioContextRef.current;
			const analyser = analyserRef.current;
			if (!audioContext || !analyser) return;
			stopCurrentInput();
			setFileStatus("idle");
			setFileError(null);
			try {
				const stream = await navigator.mediaDevices.getUserMedia({
					audio: {
						autoGainControl: false,
						noiseSuppression: false,
						echoCancellation: false,
					},
				});
				micStreamRef.current = stream;
				const source = audioContext.createMediaStreamSource(stream);
				micSourceRef.current = source;
				source.connect(analyser);
				setInputMode("mic");
			} catch (err) {
				const message = err instanceof Error ? err.message : String(err);
				setError(message);
				setFileStatus("error");
			}
		};
		startMicInputRef.current = startMicInput;

		const startBufferPlayback = async (buffer: AudioBuffer, offset = 0) => {
			await ensureAudioBackend();
			const audioContext = audioContextRef.current;
			const analyser = analyserRef.current;
			if (!audioContext || !analyser) return;
			stopCurrentInput();
			const source = audioContext.createBufferSource();
			source.buffer = buffer;
			source.connect(analyser);
			source.connect(audioContext.destination);
			fileSourceRef.current = source;
			filePlaybackStartRef.current = audioContext.currentTime - offset;
			setFileDuration(buffer.duration);
			setFilePosition(Math.min(offset, buffer.duration));
			setInputMode("file");
			setFileStatus("playing");
			source.onended = () => {
				setFileStatus("ended");
				filePlaybackStartRef.current = null;
				setFilePosition(buffer.duration);
			};
			source.start(0, offset);
			const tickProgress = () => {
				if (!audioContext || filePlaybackStartRef.current === null) return;
				const elapsed = audioContext.currentTime - filePlaybackStartRef.current;
				if (!isScrubbingRef.current) {
					setFilePosition(Math.min(elapsed, buffer.duration));
				}
				if (elapsed < buffer.duration && fileStatusRef.current !== "ended") {
					fileProgressRafRef.current = requestAnimationFrame(tickProgress);
				}
			};
			fileProgressRafRef.current = requestAnimationFrame(tickProgress);
		};

		startBufferPlaybackRef.current = startBufferPlayback;

		const startFilePlayback = async (file: File) => {
			await ensureAudioBackend();
			const audioContext = audioContextRef.current;
			if (!audioContext) return;
			setFileError(null);
			setError(null);
			setFileStatus("loading");
			try {
				const arrayBuffer = await file.arrayBuffer();
				const decoded = await audioContext.decodeAudioData(
					arrayBuffer.slice(0),
				);
				fileBufferRef.current = decoded;
				setFileName(file.name);
				await startBufferPlayback(decoded, 0);
			} catch (err) {
				const message = err instanceof Error ? err.message : String(err);
				setFileError(`Could not load file: ${message}`);
				setFileStatus("error");
			}
		};
		startFilePlaybackRef.current = startFilePlayback;

		const calcFormants = () => {
			if (isFrozenRef.current || !analysisRunningRef.current) return;
			const analyser = analyserRef.current;
			const dataArray = dataArrayRef.current;
			const worker = workerRef.current;
			const audioContext = audioContextRef.current;
			if (!analyser || !dataArray || !audioContext || !worker) return;
			analyser.getFloatTimeDomainData(dataArray);
			const { formantOrder, downsampleFactor } = lpcPresetRef.current;

			if (debugEnabledRef.current) {
				const start = performance.now();
				const coeffs = computeLpcCoefficientsForDebug(
					dataArray,
					formantOrder,
					downsampleFactor,
				);
				const durationMs = performance.now() - start;
				const windowSize = dataArray.length;
				const sampleRate = audioContext.sampleRate;
				const windowDurationMs = (windowSize / sampleRate) * 1000;

				setDebugInfo((previous) => ({
					...previous,
					windowSize,
					sampleRate,
					frameIntervalMs: FORMANT_INTERVAL_MS,
					windowDurationMs,
					lastFrameComputeMs: durationMs,
					lpcCoefficients: coeffs,
				}));
			}

			const payload: WorkerRequest = {
				type: "calcFormants",
				data: {
					audioData: Array.from(dataArray),
					lpcOrder: formantOrder,
					sampleRate: audioContext.sampleRate,
					downsampleFactor,
				},
			};
			worker.postMessage(payload);
		};

		const drawSpectrum = () => {
			const analyser = analyserRef.current;
			const dataArray = dataArrayRef.current;
			const spectrum = spectrumRef.current;
			const wasm = wasmRef.current;
			if (!analyser || !dataArray || !spectrum || !wasm) return;
			const ctx = spectrumCanvas.getContext("2d");
			if (!ctx) return;

			if (isFrozenRef.current || !analysisRunningRef.current) {
				rafSpectrumRef.current = null;
				return;
			}

			const styles = getComputedStyle(document.body);
			const canvasBg =
				styles.getPropertyValue("--canvas-bg").trim() || "#f7f3ec";
			const canvasGrid =
				styles.getPropertyValue("--canvas-grid").trim() || "#c7bcad";
			const canvasText =
				styles.getPropertyValue("--canvas-text").trim() || "#5e5247";

			analyser.getFloatTimeDomainData(dataArray);
			ctx.clearRect(0, 0, spectrumCanvas.width, spectrumCanvas.height);
			ctx.fillStyle = canvasBg;
			ctx.fillRect(0, 0, spectrumCanvas.width, spectrumCanvas.height);

			const { minFrequency, maxFrequency, logRange } = freqBoundsRef.current;
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
			ctx.fillStyle = canvasText;
			ctx.font = "16px 'Soehne', 'Inter', sans-serif";
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
				ctx.strokeStyle = canvasGrid;
				ctx.lineWidth = 1;
				ctx.stroke();
				ctx.fillText(`${Math.round(freq)} Hz`, x, spectrumCanvas.height - 22);
			}

			rafSpectrumRef.current = requestAnimationFrame(drawSpectrum);
		};

		const drawLPCFilter = () => {
			const analyser = analyserRef.current;
			const dataArray = dataArrayRef.current;
			const wasm = wasmRef.current;
			if (!analyser || !dataArray || !wasm) return;
			const ctx = spectrumCanvas.getContext("2d");
			if (!ctx) return;

			const { minFrequency, maxFrequency, logRange } = freqBoundsRef.current;
			const logMin = Math.log10(minFrequency);

			const frequencyToPosition = (freq: number) =>
				((Math.log10(freq) - logMin) / logRange) * spectrumCanvas.width;

			if (isFrozenRef.current || !analysisRunningRef.current) {
				rafLpcRef.current = null;
				return;
			}

			if (showLPCSpectrumRef.current) {
				analyser.getFloatTimeDomainData(dataArray);
				const graphSize = 1024;
				const { spectrumOrder, downsampleFactor } = lpcPresetRef.current;
				const freqResponse = wasm.lpc_filter_freq_response_with_downsampling(
					Float64Array.from(dataArray),
					spectrumOrder,
					audioContextRef.current?.sampleRate ?? 44100,
					downsampleFactor,
					graphSize,
				);

				const maxResponse = Math.max(...freqResponse);
				const normalizeConst = maxResponse > 0 ? maxResponse : 1;

				ctx.strokeStyle = "#2f6b4f";
				ctx.beginPath();
				let started = false;

				for (let i = 0; i < graphSize; i += 1) {
					const freq = (i * maxFrequency) / graphSize / downsampleFactor;
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
				const drawTargetArrow = (freq: number, color: string) => {
					if (!Number.isFinite(freq) || freq <= 0) return;
					if (freq < minFrequency || freq > maxFrequency) return;
					const xPos = frequencyToPosition(freq);
					const baseY = spectrumCanvas.height - 14;
					const size = 7;

					ctx.save();
					ctx.fillStyle = color;
					ctx.strokeStyle = "rgba(0, 0, 0, 0.2)";
					ctx.lineWidth = 1;
					ctx.beginPath();
					ctx.moveTo(xPos, baseY);
					ctx.lineTo(xPos - size, baseY + size);
					ctx.lineTo(xPos + size, baseY + size);
					ctx.closePath();
					ctx.fill();
					ctx.stroke();
					ctx.restore();
				};

				const renderLine = (value: number, color: string, label: string) => {
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

				const pitchTarget = pitchTargetRef.current;
				const vowelTargets = vowelTargetRef.current;
				if (pitchTarget !== null) {
					drawTargetArrow(pitchTarget, "#f26b38");
				}
				if (vowelTargets.f1 !== null) {
					drawTargetArrow(vowelTargets.f1, "#1f3f58");
				}
				if (vowelTargets.f2 !== null) {
					drawTargetArrow(vowelTargets.f2, "#d06c3e");
				}
			}

			rafLpcRef.current = requestAnimationFrame(drawLPCFilter);
		};

		const drawFormantHistory = () => {
			const canvas = historyCanvasRef.current;
			if (!canvas) return;
			const ctx = canvas.getContext("2d");
			if (!ctx) return;

			if (isFrozenRef.current || !analysisRunningRef.current) {
				rafHistoryRef.current = null;
				return;
			}

			ctx.clearRect(0, 0, canvas.width, canvas.height);
			const styles = getComputedStyle(document.body);
			const canvasBg =
				styles.getPropertyValue("--canvas-bg").trim() || "#f7f3ec";
			const canvasGrid =
				styles.getPropertyValue("--canvas-grid").trim() || "#d7ccbe";
			const canvasText =
				styles.getPropertyValue("--canvas-text").trim() || "#5e5247";

			ctx.fillStyle = canvasBg;
			ctx.fillRect(0, 0, canvas.width, canvas.height);

			const timeWindow = 5000;
			const latestTime =
				history.length > 0 ? history[history.length - 1].time : 0;
			const minTime = Math.max(0, latestTime - timeWindow);
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
			ctx.strokeStyle = canvasGrid;
			ctx.fillStyle = canvasText;
			ctx.font = "16px 'Soehne', 'Inter', sans-serif";

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

			rafHistoryRef.current = requestAnimationFrame(drawFormantHistory);
		};

		function stopAnalysisLoops() {
			analysisRunningRef.current = false;
			if (formantIntervalRef.current !== null) {
				clearInterval(formantIntervalRef.current);
				formantIntervalRef.current = null;
			}
			if (rafSpectrumRef.current !== null) {
				cancelAnimationFrame(rafSpectrumRef.current);
				rafSpectrumRef.current = null;
			}
			if (rafLpcRef.current !== null) {
				cancelAnimationFrame(rafLpcRef.current);
				rafLpcRef.current = null;
			}
			if (rafHistoryRef.current !== null) {
				cancelAnimationFrame(rafHistoryRef.current);
				rafHistoryRef.current = null;
			}
		}

		function startAnalysisLoops() {
			if (isFrozenRef.current) {
				analysisRunningRef.current = false;
				return;
			}
			analysisRunningRef.current = true;
			if (formantIntervalRef.current === null) {
				formantIntervalRef.current = window.setInterval(calcFormants, 100);
			}
			if (rafSpectrumRef.current === null) {
				rafSpectrumRef.current = requestAnimationFrame(drawSpectrum);
			}
			if (rafLpcRef.current === null) {
				rafLpcRef.current = requestAnimationFrame(drawLPCFilter);
			}
			if (rafHistoryRef.current === null) {
				rafHistoryRef.current = requestAnimationFrame(drawFormantHistory);
			}
		}

		function toggleAnalysisFreeze() {
			const nextFrozen = !isFrozenRef.current;
			isFrozenRef.current = nextFrozen;
			setIsFrozen(nextFrozen);
			if (nextFrozen) {
				stopAnalysisLoops();
				void audioContextRef.current?.suspend().catch(() => {});
			} else {
				void audioContextRef.current?.resume().catch(() => {});
				startAnalysisLoops();
			}
		}

		const setup = async () => {
			try {
				resize();
				window.addEventListener("resize", resize);

				const wasmUrl = new URL("../../pkg/webapp.js", import.meta.url).href;
				const wasm = (await import(/* @vite-ignore */ wasmUrl)) as WasmBindings;
				wasmRef.current = wasm;
				await wasm.default();

				const audioContext = new AudioContext();
				audioContextRef.current = audioContext;
				await audioContext.resume();

				const analyser = audioContext.createAnalyser();
				analyser.fftSize = FFT_SIZE;
				analyserRef.current = analyser;

				freqBoundsRef.current = {
					minFrequency: 20,
					maxFrequency: audioContext.sampleRate / 2,
					logRange: Math.log10(audioContext.sampleRate / 2) - Math.log10(20),
				};

				dataArrayRef.current = new Float32Array(FFT_SIZE);
				spectrumRef.current = new Float32Array(FFT_SIZE / 2);

				setDebugInfo((previous) => {
					const windowSize = FFT_SIZE;
					const sampleRate = audioContext.sampleRate;
					return {
						...previous,
						windowSize,
						sampleRate,
						frameIntervalMs: FORMANT_INTERVAL_MS,
						windowDurationMs: (windowSize / sampleRate) * 1000,
					};
				});

				const worker = new Worker(
					new URL("../formantWorker.ts", import.meta.url),
					{
						type: "module",
					},
				);
				workerRef.current = worker;

				worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
					const message = event.data;
					if (message.type === "init") {
						if (message.status === "success") {
							startAnalysisLoops();
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
						if (isFrozenRef.current) {
							return;
						}
						const [rawF1, rawF2, rawF3, rawF4] = message.formants;
						const sanitize = (value: number | undefined) =>
							Number.isFinite(value) && value > 0 ? value : 0;
						const f0 = sanitize(message.pitch);
						const f1 = sanitize(rawF1);
						const f2 = sanitize(rawF2);
						const f3 = sanitize(rawF3);
						const f4 = sanitize(rawF4);
						setFormants({ f0, f1, f2, f3, f4 });
						addFormantsToHistory({
							f0,
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
			} catch (err) {
				const message = err instanceof Error ? err.message : String(err);
				setError(message);
			}
		};

		setup();

		return () => {
			spectrumCanvas.removeEventListener("pointerdown", handlePointerDown);
			historyCanvas.removeEventListener(
				"pointerdown",
				handleHistoryPointerDown,
			);
			window.removeEventListener("pointermove", handlePointerMove);
			window.removeEventListener("pointerup", handlePointerUp);
			stopAnalysisLoops();
			if (workerRef.current) workerRef.current.terminate();
			stopCurrentInput();
			if (audioContextRef.current) audioContextRef.current.close();
			window.removeEventListener("resize", resize);
		};
	}, []);

	const selectedPitch = PITCH_TARGETS.find(
		(pitch) => pitch.id === selectedPitchId,
	);
	const selectedVowel = VOWEL_TARGETS.find(
		(vowel) => vowel.id === selectedVowelId,
	);

	let pitchTargetFreq: number | null = null;
	let pitchTargetLabel = "—";

	if (trainingMode === "pitch") {
		if (selectedPitchId === "custom") {
			pitchTargetFreq = manualPitchHz;
			pitchTargetLabel = `${Math.round(manualPitchHz)} Hz (custom)`;
		} else if (selectedPitch) {
			pitchTargetFreq = selectedPitch.freq;
			pitchTargetLabel = selectedPitch.label;
		}
	}

	let pitchDiffHz: number | null = null;
	let pitchDeltaLabel = "—";
	let pitchDirectionLabel = "";

	if (trainingMode === "pitch" && pitchTargetFreq !== null && formants.f0 > 0) {
		pitchDiffHz = formants.f0 - pitchTargetFreq;
		const sign = pitchDiffHz > 0 ? "+" : "";
		pitchDeltaLabel = `${sign}${Math.round(pitchDiffHz)} Hz`;
		if (Math.abs(pitchDiffHz) < 5) {
			pitchDirectionLabel = "on target";
		} else if (pitchDiffHz > 0) {
			pitchDirectionLabel = "above target";
		} else {
			pitchDirectionLabel = "below target";
		}
	}

	let vowelSummary = "Choose a target vowel to practice.";
	let vowelTargetF1: number | null = null;
	let vowelTargetF2: number | null = null;
	let vowelTargetDisplay = "—";

	if (trainingMode === "vowel") {
		if (selectedVowelId === "custom") {
			vowelTargetF1 = manualVowelF1;
			vowelTargetF2 = manualVowelF2;
			vowelTargetDisplay = `${Math.round(
				manualVowelF1,
			)} Hz / ${Math.round(manualVowelF2)} Hz (custom)`;
		} else if (selectedVowel) {
			vowelTargetF1 = selectedVowel.f1;
			vowelTargetF2 = selectedVowel.f2;
			vowelTargetDisplay = `${selectedVowel.f1.toFixed(
				0,
			)} Hz / ${selectedVowel.f2.toFixed(0)} Hz`;
		}
	}

	if (
		trainingMode === "vowel" &&
		vowelTargetF1 !== null &&
		vowelTargetF2 !== null &&
		formants.f1 > 0 &&
		formants.f2 > 0
	) {
		const deltaF1 = formants.f1 - vowelTargetF1;
		const deltaF2 = formants.f2 - vowelTargetF2;
		const distance = Math.hypot(deltaF1, deltaF2);

		let quality = "far from target";
		if (distance < 150) {
			quality = "very close";
		} else if (distance < 350) {
			quality = "in the ballpark";
		}

		const signF1 = deltaF1 > 0 ? "+" : "";

		vowelSummary = `${quality} (ΔF1 ${signF1}${Math.round(
			deltaF1,
		)} Hz, ΔF2 ${deltaF2 > 0 ? "+" : ""}${Math.round(deltaF2)} Hz)`;
	}

	const handleFileSelect = async (event: ChangeEvent<HTMLInputElement>) => {
		const file = event.target.files?.[0];
		if (!file) return;
		if (startFilePlaybackRef.current) {
			await startFilePlaybackRef.current(file);
		}
		event.target.value = "";
	};

	const handleUseMic = () => {
		if (startMicInputRef.current) {
			void startMicInputRef.current();
		}
	};

	const handleScrubStart = () => {
		setIsScrubbing(true);
		isScrubbingRef.current = true;
	};

	const handleScrubChange = (event: ChangeEvent<HTMLInputElement>) => {
		const next = Number.parseFloat(event.target.value);
		if (Number.isFinite(next)) {
			setFilePosition(next);
			filePositionRef.current = next;
		}
	};

	const handleScrubEnd = () => {
		setIsScrubbing(false);
		isScrubbingRef.current = false;
		const buffer = fileBufferRef.current;
		const startBufferPlayback = startBufferPlaybackRef.current;
		if (!buffer) return;
		if (!startBufferPlayback) return;
		const nextPosition = Math.min(Math.max(filePosition, 0), buffer.duration);
		filePositionRef.current = nextPosition;
		setFilePosition(nextPosition);
		void startBufferPlayback(buffer, nextPosition);
	};

	const handleTogglePlay = useCallback(() => {
		const buffer = fileBufferRef.current;
		if (!buffer) return;
		const startBufferPlayback = startBufferPlaybackRef.current;
		if (!startBufferPlayback) return;
		if (fileStatusRef.current === "playing") {
			if (fileSourceRef.current) {
				try {
					fileSourceRef.current.onended = null;
					fileSourceRef.current.stop();
				} catch {
					// ignore
				}
				fileSourceRef.current = null;
			}
			if (fileProgressRafRef.current !== null) {
				cancelAnimationFrame(fileProgressRafRef.current);
				fileProgressRafRef.current = null;
			}
			const audioContext = audioContextRef.current;
			if (audioContext && filePlaybackStartRef.current !== null) {
				const elapsed = audioContext.currentTime - filePlaybackStartRef.current;
				const next = Math.min(elapsed, buffer.duration);
				setFilePosition(next);
				filePositionRef.current = next;
			}
			filePlaybackStartRef.current = null;
			setFileStatus("paused");
			return;
		}

		const offset =
			fileStatusRef.current === "ended"
				? 0
				: Math.min(filePositionRef.current, buffer.duration);
		void startBufferPlayback(buffer, offset);
	}, []);

	useEffect(() => {
		const onKeyDown = (event: KeyboardEvent) => {
			if (event.code === "Space") {
				event.preventDefault();
				handleTogglePlay();
			}
		};
		window.addEventListener("keydown", onKeyDown);
		return () => {
			window.removeEventListener("keydown", onKeyDown);
		};
	}, [handleTogglePlay]);

	const buildSnapshot = () => ({
		version: 1,
		createdAt: new Date().toISOString(),
		settings: {
			theme,
			inputMode,
			trainingMode,
			lpcPresetId,
			showFFTSpectrum,
			showLPCSpectrum,
			showFormants,
			selectedPitchId,
			manualPitchHz,
			selectedVowelId,
			manualVowelF1,
			manualVowelF2,
		},
		stats: {
			sampleRate: debugInfo.sampleRate,
			windowSize: debugInfo.windowSize,
			frameIntervalMs: debugInfo.frameIntervalMs,
			windowDurationMs: debugInfo.windowDurationMs,
			lastFrameComputeMs: debugInfo.lastFrameComputeMs,
			formants,
		},
	});

	const handleExportSnapshot = () => {
		const snapshot = buildSnapshot();
		const blob = new Blob([JSON.stringify(snapshot, null, 2)], {
			type: "application/json",
		});
		const url = URL.createObjectURL(blob);
		const link = document.createElement("a");
		link.href = url;
		link.download = "ezformant-snapshot.json";
		document.body.append(link);
		link.click();
		link.remove();
		URL.revokeObjectURL(url);
	};

	const handleImportSnapshot = (event: ChangeEvent<HTMLInputElement>) => {
		const file = event.target.files?.[0];
		if (!file) return;
		setSnapshotError(null);

		const reader = new FileReader();
		reader.onload = () => {
			try {
				const text = String(reader.result);
				const parsed = JSON.parse(text) as {
					version?: number;
					settings?: {
						theme?: string;
						trainingMode?: TrainingMode;
						lpcPresetId?: LpcPresetId;
						showFFTSpectrum?: boolean;
						showLPCSpectrum?: boolean;
						showFormants?: boolean;
						selectedPitchId?: string;
						manualPitchHz?: number;
						selectedVowelId?: string;
						manualVowelF1?: number;
						manualVowelF2?: number;
					};
				};

				const snapshotSettings = parsed.settings ?? {};

				if (snapshotSettings.theme === "light" || snapshotSettings.theme === "dark") {
					setTheme(snapshotSettings.theme);
				}
				if (
					snapshotSettings.trainingMode === "off" ||
					snapshotSettings.trainingMode === "pitch" ||
					snapshotSettings.trainingMode === "vowel"
				) {
					setTrainingMode(snapshotSettings.trainingMode);
				}
				if (
					snapshotSettings.lpcPresetId === "speech" ||
					snapshotSettings.lpcPresetId === "singer" ||
					snapshotSettings.lpcPresetId === "noisy"
				) {
					setLpcPresetId(snapshotSettings.lpcPresetId);
				}
				if (typeof snapshotSettings.showFFTSpectrum === "boolean") {
					setShowFFTSpectrum(snapshotSettings.showFFTSpectrum);
				}
				if (typeof snapshotSettings.showLPCSpectrum === "boolean") {
					setShowLPCSpectrum(snapshotSettings.showLPCSpectrum);
				}
				if (typeof snapshotSettings.showFormants === "boolean") {
					setShowFormants(snapshotSettings.showFormants);
				}
				if (typeof snapshotSettings.selectedPitchId === "string") {
					setSelectedPitchId(snapshotSettings.selectedPitchId);
				}
				if (
					typeof snapshotSettings.manualPitchHz === "number" &&
					Number.isFinite(snapshotSettings.manualPitchHz)
				) {
					setManualPitchHz(snapshotSettings.manualPitchHz);
				}
				if (typeof snapshotSettings.selectedVowelId === "string") {
					setSelectedVowelId(snapshotSettings.selectedVowelId);
				}
				if (
					typeof snapshotSettings.manualVowelF1 === "number" &&
					Number.isFinite(snapshotSettings.manualVowelF1)
				) {
					setManualVowelF1(snapshotSettings.manualVowelF1);
				}
				if (
					typeof snapshotSettings.manualVowelF2 === "number" &&
					Number.isFinite(snapshotSettings.manualVowelF2)
				) {
					setManualVowelF2(snapshotSettings.manualVowelF2);
				}
			} catch {
				setSnapshotError("Could not parse snapshot JSON file.");
			}
		};
		reader.onerror = () => {
			setSnapshotError("Could not read snapshot file.");
		};
		reader.readAsText(file);
		event.target.value = "";
	};

	const hasLoadedFile = fileBufferRef.current !== null;
	const micReady = micStreamRef.current !== null;
	const fileStatusLabel = (() => {
		switch (fileStatus) {
			case "loading":
				return "Loading file…";
			case "playing":
				return "Playing file";
			case "paused":
				return "Paused";
			case "ended":
				return "Playback finished";
			case "error":
				return "File error";
			default:
				if (inputMode === "mic") {
					return micReady ? "Live microphone" : "Mic not started";
				}
				return hasLoadedFile ? "File ready" : "Choose mic or file";
		}
	})();
	const _currentLpcPreset =
		LPC_PRESETS.find((preset) => preset.id === lpcPresetId) ?? LPC_PRESETS[0];

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
				<button
					type="button"
					className="action-button"
					onClick={() =>
						setTheme((current) => (current === "light" ? "dark" : "light"))
					}
				>
					{theme === "light" ? "Switch to dark" : "Switch to light"}
				</button>
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
					{isFrozen ? <span className="badge frozen-badge">Frozen</span> : null}
					<details className="advanced-toggle">
						<summary>Advanced</summary>
						<div className="advanced-body">
							<label className="toggle small">
								<input
									type="checkbox"
									checked={showFFTSpectrum}
									onChange={(e) => setShowFFTSpectrum(e.target.checked)}
								/>
								<span>FFT spectrum</span>
							</label>
							<label className="toggle small">
								<input
									type="checkbox"
									checked={showLPCSpectrum}
									onChange={(e) => setShowLPCSpectrum(e.target.checked)}
								/>
								<span>LPC envelope</span>
							</label>
							<label className="toggle small">
								<input
									type="checkbox"
									checked={showFormants}
									onChange={(e) => setShowFormants(e.target.checked)}
								/>
								<span>Formant markers</span>
							</label>
							<label className="toggle small">
								<span>LPC preset</span>
								<select
									value={lpcPresetId}
									onChange={(event) =>
										setLpcPresetId(event.target.value as LpcPresetId)
									}
								>
									{LPC_PRESETS.map((preset) => (
										<option key={preset.id} value={preset.id}>
											{preset.label}
										</option>
									))}
								</select>
							</label>
						</div>
					</details>
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
			</section>

			<section className="panels-row">
				<section className="metric input-card">
					<div className="input-header">
						<div>
							<div className="label">Input source</div>
							<p className="input-hint">Use mic or a local audio file.</p>
						</div>
						<div className="input-actions">
							<button
								type="button"
								className={`action-button ${inputMode === "mic" ? "primary" : ""}`}
								onClick={handleUseMic}
								disabled={fileStatus === "loading"}
							>
								Use mic
							</button>
							<label
								className={`upload-label action-button ${
									fileStatus === "loading" ? "disabled" : ""
								}`}
							>
								<input
									type="file"
									accept="audio/*"
									onChange={handleFileSelect}
									disabled={fileStatus === "loading"}
								/>
								<span>
									{fileStatus === "loading" ? "Loading…" : "Pick audio"}
								</span>
							</label>
							<button
								type="button"
								className="action-button"
								onClick={handleTogglePlay}
								disabled={
									!hasLoadedFile ||
									fileStatus === "loading" ||
									fileStatus === "error"
								}
							>
								{fileStatus === "playing" ? "Pause" : "Play"}
							</button>
						</div>
					</div>
					<div className="input-status">
						<div className={`status-pill ${fileStatus}`}>{fileStatusLabel}</div>
						<div className="status-details compact">
							<div className="status-line">
								{inputMode === "mic"
									? micReady
										? "Live microphone."
										: "Click “Use mic” to start."
									: fileName || "No file selected."}
							</div>
							{inputMode === "file" && fileDuration ? (
								<div className="status-sub">
									{formatDuration(fileDuration)} total
								</div>
							) : null}
							{fileError ? (
								<div className="error-inline">{fileError}</div>
							) : null}
							{inputMode === "file" && hasLoadedFile && fileDuration ? (
								<div className="scrub-row">
									<input
										type="range"
										min={0}
										max={fileDuration}
										step={0.01}
										value={
											isScrubbing
												? filePosition
												: Math.min(filePosition, fileDuration)
										}
										onMouseDown={handleScrubStart}
										onTouchStart={handleScrubStart}
										onChange={handleScrubChange}
										onMouseUp={handleScrubEnd}
										onTouchEnd={handleScrubEnd}
									/>
									<div className="scrub-times">
										<span>{formatDuration(filePosition)}</span>
										<span>{formatDuration(fileDuration)}</span>
									</div>
								</div>
							) : null}
						</div>
					</div>
				</section>

				<section className="trainer">
					<div className="metric trainer-card">
						<div className="trainer-header">
							<div className="label">Target trainer</div>
							<button
								type="button"
								className="action-button"
								onClick={() =>
									setTrainingMode((current) =>
										current === "off" ? "pitch" : "off",
									)
								}
							>
								{trainingMode === "off" ? "Open trainer" : "Close"}
							</button>
						</div>

						{trainingMode === "off" ? (
							<p className="trainer-hint">
								Practice matching your pitch or vowel targets against the live
								signal.
							</p>
						) : (
							<>
								<div className="trainer-modes">
									<button
										type="button"
										className={trainingMode === "pitch" ? "active" : ""}
										onClick={() => setTrainingMode("pitch")}
									>
										Pitch
									</button>
									<button
										type="button"
										className={trainingMode === "vowel" ? "active" : ""}
										onClick={() => setTrainingMode("vowel")}
									>
										Vowel
									</button>
								</div>

								{trainingMode === "pitch" ? (
									<div className="trainer-body">
										<label className="trainer-field">
											<span>Target note</span>
											<select
												value={selectedPitchId}
												onChange={(event) =>
													setSelectedPitchId(event.target.value)
												}
											>
												{PITCH_TARGETS.map((target) => (
													<option key={target.id} value={target.id}>
														{target.label}
													</option>
												))}
												<option value="custom">Custom (Hz)</option>
											</select>
										</label>
										{selectedPitchId === "custom" ? (
											<label className="trainer-field">
												<span>Custom F0</span>
												<input
													type="number"
													min={40}
													max={2000}
													value={manualPitchHz}
													onChange={(event) => {
														const next = Number.parseFloat(
															event.target.value,
														);
														if (Number.isFinite(next)) {
															setManualPitchHz(next);
														}
													}}
												/>
												<span>Hz</span>
											</label>
										) : null}
										<div className="trainer-readout">
											<span>Target {pitchTargetLabel}</span>
											<span>Current {formants.f0.toFixed(0)} Hz</span>
											<span>
												Δ{" "}
												{pitchDiffHz !== null
													? `${pitchDeltaLabel} (${pitchDirectionLabel})`
													: "—"}
											</span>
										</div>
									</div>
								) : null}

								{trainingMode === "vowel" ? (
									<div className="trainer-body">
										<label className="trainer-field">
											<span>Target vowel</span>
											<select
												value={selectedVowelId}
												onChange={(event) =>
													setSelectedVowelId(event.target.value)
												}
											>
												{VOWEL_TARGETS.map((target) => (
													<option key={target.id} value={target.id}>
														{target.label} – {target.example}
													</option>
												))}
												<option value="custom">Custom F1/F2</option>
											</select>
										</label>
										{selectedVowelId === "custom" ? (
											<div className="trainer-body">
												<label className="trainer-field">
													<span>Custom F1</span>
													<input
														type="number"
														min={100}
														max={2000}
														value={manualVowelF1}
														onChange={(event) => {
															const next = Number.parseFloat(
																event.target.value,
															);
															if (Number.isFinite(next)) {
																setManualVowelF1(next);
															}
														}}
													/>
													<span>Hz</span>
												</label>
												<label className="trainer-field">
													<span>Custom F2</span>
													<input
														type="number"
														min={300}
														max={4000}
														value={manualVowelF2}
														onChange={(event) => {
															const next = Number.parseFloat(
																event.target.value,
															);
															if (Number.isFinite(next)) {
																setManualVowelF2(next);
															}
														}}
													/>
													<span>Hz</span>
												</label>
											</div>
										) : null}
										<div className="trainer-readout">
											<span>Target F1/F2 {vowelTargetDisplay}</span>
											<span>
												Current F1/F2{" "}
												{formants.f1 > 0 && formants.f2 > 0
													? `${formants.f1.toFixed(
															0,
														)} Hz / ${formants.f2.toFixed(0)} Hz`
													: "—"}
											</span>
											<span>{vowelSummary}</span>
										</div>
									</div>
								) : null}
							</>
						)}
					</div>
				</section>

			</section>

			{showDeveloperUi ? (
				<section className="metric developer-panel">
					<div className="developer-header">
						<div>
							<div className="label">Developer / diagnostics</div>
							<div className="status-sub">
								Inspect LPC internals and export/import JSON snapshots.
							</div>
						</div>
						<button
							type="button"
							className="action-button"
							onClick={() => setDebugPanelOpen((open) => !open)}
						>
							{debugPanelOpen ? "Hide debug panel" : "Show debug panel"}
						</button>
					</div>
					{debugPanelOpen ? (
						<div className="developer-body">
							<div className="developer-grid">
								<div className="developer-stat">
									<div className="label">Sample rate</div>
									<div className="value small">
										{debugInfo.sampleRate
											? `${debugInfo.sampleRate.toFixed(0)} Hz`
											: "—"}
									</div>
								</div>
								<div className="developer-stat">
									<div className="label">Window size</div>
									<div className="value small">
										{debugInfo.windowSize
											? `${debugInfo.windowSize} samples`
											: "—"}
									</div>
								</div>
								<div className="developer-stat">
									<div className="label">Window duration</div>
									<div className="value small">
										{debugInfo.windowDurationMs
											? `${debugInfo.windowDurationMs.toFixed(1)} ms`
											: "—"}
									</div>
								</div>
								<div className="developer-stat">
									<div className="label">Analysis interval</div>
									<div className="value small">
										{`${debugInfo.frameIntervalMs.toFixed(0)} ms`}
									</div>
								</div>
								<div className="developer-stat">
									<div className="label">Last LPC compute</div>
									<div className="value small">
										{debugInfo.lastFrameComputeMs
											? `${debugInfo.lastFrameComputeMs.toFixed(2)} ms`
											: "—"}
									</div>
								</div>
							</div>
							<div className="developer-section">
								<div className="label">LPC coefficients</div>
								<div className="developer-coeffs">
									{debugInfo.lpcCoefficients &&
									debugInfo.lpcCoefficients.length > 0 ? (
										debugInfo.lpcCoefficients.map((coef, index) => {
											const label = `a${index}`;
											return (
												<code key={label}>{`${label}=${coef.toFixed(
													4,
												)}`}</code>
											);
										})
									) : (
										<span className="status-sub">
											No coefficients yet – keep this panel open while audio is
											running.
										</span>
									)}
								</div>
							</div>
							<div className="developer-section">
								<div className="label">Snapshots</div>
								<div className="developer-actions">
									<button
										type="button"
										className="action-button"
										onClick={handleExportSnapshot}
									>
										Export JSON snapshot
									</button>
									<label className="upload-label action-button">
										<input
											type="file"
											accept="application/json,.json"
											onChange={handleImportSnapshot}
										/>
										<span>Import JSON snapshot</span>
									</label>
								</div>
								{snapshotError ? (
									<div className="error-inline">{snapshotError}</div>
								) : null}
							</div>
						</div>
					) : null}
				</section>
			) : null}

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
