/**
 * Procedural audio generator — produces CC0-equivalent WAV files from pure math.
 *
 * No downloads, no licensing ambiguity: every sample is computed here.
 * Run: npx tsx scripts/generate-audio.ts
 */

import * as fs from "fs";
import * as path from "path";

const OUTPUT_DIR = path.resolve(__dirname, "..", "public", "audio");
const SFX_DIR = path.join(OUTPUT_DIR, "sfx");
const MUSIC_DIR = path.join(OUTPUT_DIR, "music");

const SAMPLE_RATE = 44100;

// --- WAV writer ---------------------------------------------------------

function writeWavMono(samples: Float32Array, outputPath: string): void {
  const numSamples = samples.length;
  const bytesPerSample = 2;
  const buf = Buffer.alloc(44 + numSamples * bytesPerSample);
  buf.write("RIFF", 0);
  buf.writeUInt32LE(36 + numSamples * bytesPerSample, 4);
  buf.write("WAVE", 8);
  buf.write("fmt ", 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20); // PCM
  buf.writeUInt16LE(1, 22); // mono
  buf.writeUInt32LE(SAMPLE_RATE, 24);
  buf.writeUInt32LE(SAMPLE_RATE * bytesPerSample, 28);
  buf.writeUInt16LE(bytesPerSample, 32);
  buf.writeUInt16LE(16, 34);
  buf.write("data", 36);
  buf.writeUInt32LE(numSamples * bytesPerSample, 40);
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    buf.writeInt16LE(Math.floor(s * 32767), 44 + i * 2);
  }
  fs.writeFileSync(outputPath, buf);
}

function writeWavStereo(
  left: Float32Array,
  right: Float32Array,
  outputPath: string
): void {
  const numSamples = left.length;
  const bytesPerSample = 2;
  const totalSize = numSamples * 2 * bytesPerSample;
  const buf = Buffer.alloc(44 + totalSize);
  buf.write("RIFF", 0);
  buf.writeUInt32LE(36 + totalSize, 4);
  buf.write("WAVE", 8);
  buf.write("fmt ", 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(2, 22); // stereo
  buf.writeUInt32LE(SAMPLE_RATE, 24);
  buf.writeUInt32LE(SAMPLE_RATE * 2 * bytesPerSample, 28);
  buf.writeUInt16LE(2 * bytesPerSample, 32);
  buf.writeUInt16LE(16, 34);
  buf.write("data", 36);
  buf.writeUInt32LE(totalSize, 40);
  for (let i = 0; i < numSamples; i++) {
    const l = Math.max(-1, Math.min(1, left[i]));
    const r = Math.max(-1, Math.min(1, right[i]));
    buf.writeInt16LE(Math.floor(l * 32767), 44 + i * 4);
    buf.writeInt16LE(Math.floor(r * 32767), 44 + i * 4 + 2);
  }
  fs.writeFileSync(outputPath, buf);
}

// --- SFX generators -----------------------------------------------------

function adsr(
  len: number,
  attack: number,
  decay: number,
  sustain: number,
  release: number
): Float32Array {
  const env = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    const t = i / len;
    let a: number;
    if (t < attack) a = t / attack;
    else if (t < attack + decay) a = 1 - ((t - attack) / decay) * (1 - sustain);
    else if (t < 1 - release) a = sustain;
    else a = sustain * (1 - (t - (1 - release)) / release);
    env[i] = a;
  }
  return env;
}

function genWhoosh(durationSec = 0.6): Float32Array {
  const n = Math.floor(durationSec * SAMPLE_RATE);
  const out = new Float32Array(n);
  const env = adsr(n, 0.1, 0.2, 0.5, 0.2);
  // Pink-ish noise via low-pass filter
  let lpState = 0;
  for (let i = 0; i < n; i++) {
    const noise = Math.random() * 2 - 1;
    lpState = lpState * 0.6 + noise * 0.4;
    // Rising sweep in cutoff: add resonance via second lp
    const sweep = 0.3 + (i / n) * 0.5;
    out[i] = lpState * env[i] * sweep * 0.5;
  }
  return out;
}

function genTick(durationSec = 0.04): Float32Array {
  const n = Math.floor(durationSec * SAMPLE_RATE);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / SAMPLE_RATE;
    const env = Math.exp(-t * 200);
    out[i] = (Math.sin(2 * Math.PI * 2200 * t) * 0.4 + (Math.random() * 2 - 1) * 0.3) * env;
  }
  return out;
}

function genNumberTick(durationSec = 0.05): Float32Array {
  const n = Math.floor(durationSec * SAMPLE_RATE);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / SAMPLE_RATE;
    const env = Math.exp(-t * 120);
    out[i] = Math.sin(2 * Math.PI * 1600 * t) * env * 0.4;
  }
  return out;
}

function genPop(durationSec = 0.12): Float32Array {
  const n = Math.floor(durationSec * SAMPLE_RATE);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / SAMPLE_RATE;
    const env = Math.exp(-t * 35);
    const freq = 800 + 1500 * Math.exp(-t * 20);
    out[i] = Math.sin(2 * Math.PI * freq * t) * env * 0.5;
  }
  return out;
}

function genImpact(durationSec = 0.8): Float32Array {
  const n = Math.floor(durationSec * SAMPLE_RATE);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / SAMPLE_RATE;
    const bassEnv = Math.exp(-t * 6);
    const noiseEnv = Math.exp(-t * 12);
    const bass = Math.sin(2 * Math.PI * 60 * t + Math.sin(2 * Math.PI * 15 * t)) * bassEnv;
    const noise = (Math.random() * 2 - 1) * noiseEnv * 0.3;
    const mid = Math.sin(2 * Math.PI * 140 * t) * bassEnv * 0.4;
    out[i] = (bass + noise + mid) * 0.75;
  }
  return out;
}

function genNotification(durationSec = 0.25): Float32Array {
  const n = Math.floor(durationSec * SAMPLE_RATE);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / SAMPLE_RATE;
    const env = Math.exp(-t * 9);
    const f1 = Math.sin(2 * Math.PI * 1200 * t);
    const f2 = Math.sin(2 * Math.PI * 1800 * t);
    out[i] = (f1 * 0.5 + f2 * 0.3) * env * 0.5;
  }
  return out;
}

function genRiser(durationSec = 1.2): Float32Array {
  const n = Math.floor(durationSec * SAMPLE_RATE);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / SAMPLE_RATE;
    const progress = t / durationSec;
    const freq = 200 * Math.pow(8, progress);
    const env = progress < 0.8 ? progress / 0.8 : 1 - (progress - 0.8) / 0.2;
    const noise = (Math.random() * 2 - 1) * 0.2;
    out[i] = (Math.sin(2 * Math.PI * freq * t) * 0.5 + noise) * env * 0.4;
  }
  return out;
}

// --- Ambient music generator -------------------------------------------

/** Simple slow pad progression in A minor over 75 seconds, stereo */
function genAmbientPad(
  durationSec = 75
): { left: Float32Array; right: Float32Array } {
  const n = Math.floor(durationSec * SAMPLE_RATE);
  const left = new Float32Array(n);
  const right = new Float32Array(n);

  // Chord progression (16s per chord, A minor / F / C / G)
  const chords = [
    [220, 261.63, 329.63],         // A minor (A C E)
    [174.61, 220, 261.63],         // F (F A C)
    [261.63, 329.63, 392],         // C (C E G)
    [196, 246.94, 293.66],         // G (G B D)
  ];

  const chordLen = durationSec / chords.length;

  for (let i = 0; i < n; i++) {
    const t = i / SAMPLE_RATE;
    const chordIdx = Math.min(chords.length - 1, Math.floor(t / chordLen));
    const chord = chords[chordIdx];
    const chordProg = (t % chordLen) / chordLen;
    // Crossfade between chords for 2 seconds
    const crossfade = 2;
    const prevChord = chords[(chordIdx - 1 + chords.length) % chords.length];
    const transitionProgress = Math.min(1, chordProg * chordLen / crossfade);

    let sampleL = 0;
    let sampleR = 0;

    // Blend of current + previous chord during transition
    for (let k = 0; k < 3; k++) {
      const prev = prevChord[k];
      const curr = chord[k];

      // Detune slightly for stereo width
      const detuneL = 1.0;
      const detuneR = 1.0025;

      // Sine with subtle vibrato
      const vibrato = Math.sin(2 * Math.PI * 4.5 * t) * 0.002;

      const prevFreqL = prev * (1 + vibrato) * detuneL;
      const prevFreqR = prev * (1 + vibrato) * detuneR;
      const currFreqL = curr * (1 + vibrato) * detuneL;
      const currFreqR = curr * (1 + vibrato) * detuneR;

      // Harmonic series (fundamental + 2nd harmonic softer)
      const voiceL =
        Math.sin(2 * Math.PI * prevFreqL * t) * (1 - transitionProgress) +
        Math.sin(2 * Math.PI * currFreqL * t) * transitionProgress;
      const voiceR =
        Math.sin(2 * Math.PI * prevFreqR * t) * (1 - transitionProgress) +
        Math.sin(2 * Math.PI * currFreqR * t) * transitionProgress;

      sampleL += voiceL / 3;
      sampleR += voiceR / 3;
    }

    // Soft filter (low-pass via running average) to mellow the sound
    // Add a subtle filtered noise for texture
    const noise = (Math.random() * 2 - 1) * 0.02;
    sampleL += noise;
    sampleR += noise * 0.8;

    // Fade in/out on the whole track
    const fadeIn = Math.min(1, t / 3);
    const fadeOut = Math.min(1, (durationSec - t) / 3);
    const env = fadeIn * fadeOut;

    // Final gain
    left[i] = sampleL * 0.28 * env;
    right[i] = sampleR * 0.28 * env;
  }

  // Simple low-pass smoothing to mellow the sines
  const smooth = (arr: Float32Array) => {
    const a = 0.22;
    let last = 0;
    for (let i = 0; i < arr.length; i++) {
      last = last * (1 - a) + arr[i] * a;
      arr[i] = last;
    }
  };
  smooth(left);
  smooth(right);

  return { left, right };
}

// --- Main ---------------------------------------------------------------

function ensureDir(p: string) {
  if (!fs.existsSync(p)) fs.mkdirSync(p, { recursive: true });
}

function main() {
  ensureDir(SFX_DIR);
  ensureDir(MUSIC_DIR);

  const sfxPlan: Array<{ name: string; fn: () => Float32Array }> = [
    { name: "whoosh.wav", fn: () => genWhoosh(0.6) },
    { name: "typewriter-tick.wav", fn: () => genTick(0.04) },
    { name: "number-tick.wav", fn: () => genNumberTick(0.05) },
    { name: "impact-boom.wav", fn: () => genImpact(0.8) },
    { name: "notification-pop.wav", fn: () => genPop(0.12) },
    { name: "subtle-riser.wav", fn: () => genRiser(1.2) },
  ];

  for (const { name, fn } of sfxPlan) {
    const samples = fn();
    const out = path.join(SFX_DIR, name);
    writeWavMono(samples, out);
    const kb = (fs.statSync(out).size / 1024).toFixed(1);
    console.log(`✓ ${name}  (${kb}KB, ${(samples.length / SAMPLE_RATE).toFixed(2)}s)`);
  }

  console.log("\n🎵 generating ambient pad (120s stereo)...");
  const { left, right } = genAmbientPad(120);
  const musicOut = path.join(MUSIC_DIR, "pitch-ambient.wav");
  writeWavStereo(left, right, musicOut);
  const mb = (fs.statSync(musicOut).size / 1024 / 1024).toFixed(2);
  console.log(`✓ pitch-ambient.wav  (${mb}MB, 120.00s stereo)`);

  console.log("\n✅ audio generation complete");
}

main();
