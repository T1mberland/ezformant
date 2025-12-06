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
	const [fileStatus, setFileStatus] = useState<
		"idle" | "loading" | "playing" | "paused" | "ended" | "error"
	>("idle");
	const [fileName, setFileName] = useState("");
	const [fileDuration, setFileDuration] = useState<number | null>(null);
	const [fileError, setFileError] = useState<string | null>(null);
	const [filePosition, setFilePosition] = useState(0);
	const [isScrubbing, setIsScrubbing] = useState(false);

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
	const micStreamRef = useRef<MediaStream | null>(null);
	const micSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
	const fileSourceRef = useRef<AudioBufferSourceNode | null>(null);
	const fileBufferRef = useRef<AudioBuffer | null>(null);
	const startMicInputRef = useRef<() => Promise<void> | null>(null);
	const startFilePlaybackRef =
		useRef<(file: File) => Promise<void> | null>(null);
	const replayFileRef = useRef<(() => void) | null>(null);
	const filePlaybackStartRef = useRef<number | null>(null);
	const fileProgressRafRef = useRef<number | null>(null);
	const fileStatusRef = useRef(fileStatus);
	const isScrubbingRef = useRef(isScrubbing);
	const filePositionRef = useRef(0);
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
		trainingModeRef.current = trainingMode;
	}, [trainingMode]);
	useEffect(() => {
		selectedPitchIdRef.current = selectedPitchId;
	}, [selectedPitchId]);
	useEffect(() => {
		selectedVowelIdRef.current = selectedVowelId;
	}, [selectedVowelId]);
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

		type DragTarget = "pitch" | "vowelF1" | "vowelF2";
		let draggingTarget: DragTarget | null = null;

		const resize = () => {
			const width = spectrumCanvas.clientWidth * window.devicePixelRatio;
			const height = 360 * window.devicePixelRatio;

			spectrumCanvas.width = width;
			spectrumCanvas.height = height;
			historyCanvas.width = width;
			historyCanvas.height = height;
		};

		const addFormantsToHistory = (sample: FormantSample) => {
			history.push(sample);
			if (history.length > MAX_HISTORY) history.shift();
		};

		const positionToFrequency = (x: number) => {
			const { minFrequency, maxFrequency, logRange } = freqBoundsRef.current;
			const logMin = Math.log10(minFrequency);
			const fraction = Math.min(Math.max(x / spectrumCanvas.width, 0), 1);
			const logFreq = logMin + fraction * logRange;
			const freq = 10 ** logFreq;
			return Math.min(Math.max(freq, minFrequency), maxFrequency);
		};

		const handlePointerDown = (event: PointerEvent) => {
			const rect = spectrumCanvas.getBoundingClientRect();
			const scaleX = spectrumCanvas.width / rect.width;
			const scaleY = spectrumCanvas.height / rect.height;
			const x = (event.clientX - rect.left) * scaleX;
			const y = (event.clientY - rect.top) * scaleY;

			const { minFrequency, logRange } = freqBoundsRef.current;
			const logMin = Math.log10(minFrequency);
			const frequencyToPosition = (freq: number) =>
				((Math.log10(freq) - logMin) / logRange) * spectrumCanvas.width;

			type Candidate = { kind: DragTarget; x: number };
			const candidates: Candidate[] = [];

			const pitchTarget =
				trainingModeRef.current === "pitch" ? pitchTargetRef.current : null;
			const vowelTargets =
				trainingModeRef.current === "vowel" ? vowelTargetRef.current : null;

			if (pitchTarget !== null) {
				candidates.push({ kind: "pitch", x: frequencyToPosition(pitchTarget) });
			}
			if (vowelTargets?.f1 !== null) {
				candidates.push({
					kind: "vowelF1",
					x: frequencyToPosition(vowelTargets.f1),
				});
			}
			if (vowelTargets?.f2 !== null) {
				candidates.push({
					kind: "vowelF2",
					x: frequencyToPosition(vowelTargets.f2),
				});
			}

			const hitThreshold = 22;
			let best: Candidate | null = null;
			let bestDist = Number.POSITIVE_INFINITY;

			candidates.forEach((candidate) => {
				const dist = Math.hypot(candidate.x - x, y - spectrumCanvas.height);
				if (dist < hitThreshold && dist < bestDist) {
					best = candidate;
					bestDist = dist;
				}
			});

			if (!best) {
				const nextFrozen = !isFrozenRef.current;
				setIsFrozen(nextFrozen);
				isFrozenRef.current = nextFrozen;
				return;
			}
			draggingTarget = best.kind;
			event.preventDefault();
		};

		const handleHistoryPointerDown = (event: PointerEvent) => {
			event.preventDefault();
			const nextFrozen = !isFrozenRef.current;
			setIsFrozen(nextFrozen);
			isFrozenRef.current = nextFrozen;
		};

		const handlePointerMove = (event: PointerEvent) => {
			if (!draggingTarget) return;
			const rect = spectrumCanvas.getBoundingClientRect();
			const scaleX = spectrumCanvas.width / rect.width;
			const x = (event.clientX - rect.left) * scaleX;
			const freq = positionToFrequency(x);

			if (draggingTarget === "pitch") {
				if (trainingModeRef.current !== "pitch") return;
				const clamped = Math.min(Math.max(freq, 40), 2000);
				if (selectedPitchIdRef.current !== "custom") {
					setSelectedPitchId("custom");
				}
				setManualPitchHz(clamped);
			} else if (draggingTarget === "vowelF1") {
				if (trainingModeRef.current !== "vowel") return;
				const clamped = Math.min(Math.max(freq, 100), 2000);
				if (selectedVowelIdRef.current !== "custom") {
					setSelectedVowelId("custom");
				}
				setManualVowelF1(clamped);
			} else if (draggingTarget === "vowelF2") {
				if (trainingModeRef.current !== "vowel") return;
				const clamped = Math.min(Math.max(freq, 300), 4000);
				if (selectedVowelIdRef.current !== "custom") {
					setSelectedVowelId("custom");
				}
				setManualVowelF2(clamped);
			}
		};

		const handlePointerUp = () => {
			draggingTarget = null;
		};

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
		replayFileRef.current = () => {
			const buffer = fileBufferRef.current;
			if (buffer) {
				void startBufferPlayback(buffer, 0);
			}
		};

		const calcFormants = () => {
			if (isFrozenRef.current) return;
			const analyser = analyserRef.current;
			const dataArray = dataArrayRef.current;
			const worker = workerRef.current;
			const audioContext = audioContextRef.current;
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

		const drawSpectrum = () => {
			const analyser = analyserRef.current;
			const dataArray = dataArrayRef.current;
			const spectrum = spectrumRef.current;
			const wasm = wasmRef.current;
			if (!analyser || !dataArray || !spectrum || !wasm) return;
			const ctx = spectrumCanvas.getContext("2d");
			if (!ctx) return;

			if (isFrozenRef.current) {
				rafSpectrumRef.current = requestAnimationFrame(drawSpectrum);
				return;
			}

			analyser.getFloatTimeDomainData(dataArray);
			ctx.clearRect(0, 0, spectrumCanvas.width, spectrumCanvas.height);
			ctx.fillStyle = "#f7f3ec";
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
			ctx.fillStyle = "#5e5247";
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
				ctx.strokeStyle = "#c7bcad";
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

			if (isFrozenRef.current) {
				rafLpcRef.current = requestAnimationFrame(drawLPCFilter);
				return;
			}

			if (showLPCSpectrumRef.current) {
				analyser.getFloatTimeDomainData(dataArray);
				const graphSize = 1024;
				const freqResponse = wasm.lpc_filter_freq_response_with_downsampling(
					Float64Array.from(dataArray),
					LPC_SPECTRUM_ORDER,
					audioContextRef.current?.sampleRate ?? 44100,
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

			if (isFrozenRef.current) {
				rafHistoryRef.current = requestAnimationFrame(drawFormantHistory);
				return;
			}

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
							formantIntervalRef.current = window.setInterval(
								calcFormants,
								100,
							);
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
				rafSpectrumRef.current = requestAnimationFrame(drawSpectrum);
				rafLpcRef.current = requestAnimationFrame(drawLPCFilter);
				rafHistoryRef.current = requestAnimationFrame(drawFormantHistory);
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
			if (rafSpectrumRef.current) cancelAnimationFrame(rafSpectrumRef.current);
			if (rafLpcRef.current) cancelAnimationFrame(rafLpcRef.current);
			if (rafHistoryRef.current) cancelAnimationFrame(rafHistoryRef.current);
			if (formantIntervalRef.current !== null)
				clearInterval(formantIntervalRef.current);
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

	const handleReplayFile = () => {
		if (replayFileRef.current) {
			replayFileRef.current();
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
					{isFrozen ? <span className="badge frozen-badge">Frozen</span> : null}
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

			<section className="metric input-card">
				<div className="input-header">
					<div>
						<div className="label">Input source</div>
						<p className="input-hint">
							Analyze live mic or load a WAV/MP3/OGG file (local only).
						</p>
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
							onClick={handleReplayFile}
							disabled={!hasLoadedFile || fileStatus === "loading"}
						>
							Replay file
						</button>
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
					<div className="status-details">
						<div className="status-line">
							{inputMode === "mic"
								? micReady
									? "Listening to your microphone (on-device only)."
									: "Click “Use mic” to start live analysis."
								: fileName
									? `File: ${fileName}`
									: "No file selected yet."}
						</div>
						{inputMode === "file" && fileDuration ? (
							<div className="status-sub">
								Duration {formatDuration(fileDuration)}
							</div>
						) : null}
						{fileError ? <div className="error-inline">{fileError}</div> : null}
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

			<section className="trainer">
				<div className="metric trainer-card">
					<div className="trainer-header">
						<div className="label">Target trainer</div>
						<div className="trainer-modes">
							<button
								type="button"
								className={trainingMode === "off" ? "active" : ""}
								onClick={() => setTrainingMode("off")}
							>
								Off
							</button>
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
					</div>

					{trainingMode === "pitch" ? (
						<div className="trainer-body">
							<label className="trainer-field">
								<span>Target note</span>
								<select
									value={selectedPitchId}
									onChange={(event) => setSelectedPitchId(event.target.value)}
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
											const next = Number.parseFloat(event.target.value);
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
									onChange={(event) => setSelectedVowelId(event.target.value)}
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
												const next = Number.parseFloat(event.target.value);
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
												const next = Number.parseFloat(event.target.value);
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

					{trainingMode === "off" ? (
						<p className="trainer-hint">
							Pick a pitch or vowel target to see how far your live signal is
							from the goal.
						</p>
					) : null}
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
