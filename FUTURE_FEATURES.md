Ideas for future EZ Formant development
======================================

Visual & UX
-----------

- [ ] Snap-to-vowel overlays on the F1/F2 history plot (ellipses for canonical /i, e, a, o, u/ regions).
- [ ]  “Hold to analyze” mode: only record/compute formants while a mouse button or key is held.
- [ ]  Screenshot/export buttons for spectrum and history views (PNG/SVG).

Audio & recording
-----------------

- [ ]  Record/playback mode for short clips so users can scrub spectrum/history without live mic.
- [ ]  Support loading a local audio file (e.g., WAV/OGG) for offline analysis.
- [ ]  Simple input level meter and noise gate to avoid analyzing silence/background noise.

Analysis features
-----------------

- [ ]  Stability/confidence indicator for F0/F1/F2 over the last N frames.
- [ ]  Display formant bandwidths / Q factors (e.g., as shaded regions around peaks).
- [ ]  Advanced LPC settings: expose LPC order and downsampling factor via presets (“speech”, “singer”, “noisy”).
- [ ]  Mark detected peaks on the FFT that correspond to formants.

Pedagogical / practice tools
----------------------------

- [x] Target trainer: pick a target vowel or pitch+formant combo and show live deviation.
- [ ]  History bookmarks to label segments (e.g., first vs. second syllable) and toggle between them.
- [ ]  Support for scripted sessions: predefined sequences of sounds with per-token summaries of F0 and formants.

Developer/diagnostic
--------------------

- [ ]  Debug panel showing LPC coefficients, window size, sample rate, and timing (ms per frame).
- [ ]  Export/import JSON snapshots of settings and basic statistics for reproducible bug reports or experiments.

