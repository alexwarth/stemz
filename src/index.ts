const { abs, max, min, random, round, sin, tan, PI } = Math;
const TAU = 2 * PI;

// #region interface

interface ASRArgs {
  attack?: number;
  sustainLevel?: number;
  sustain?: number;
  release?: number;
  curve?(value: number): number;
}

export abstract class Stem {
  abstract valueAt(t: number): number;

  ref() {
    return new Ref(this);
  }

  add(...args: (number | Stem)[]) {
    return add(this, ...args.map(asStem));
  }

  mix(...args: (number | Stem)[]) {
    return mix(this, ...args.map(asStem));
  }

  mul(...args: (number | Stem)[]) {
    return mul(this, ...args.map(asStem));
  }

  div(divisor: number | Stem) {
    return div(this, divisor);
  }

  lpf(cutoffFreq: number | Stem, q: number | Stem = 0.707) {
    return new BiquadFilter('lp', this, asStem(cutoffFreq), asStem(q));
  }

  hpf(cutoffFreq: number | Stem, q: number | Stem = 0.707) {
    return new BiquadFilter('hp', this, asStem(cutoffFreq), asStem(q));
  }

  asr(args: ASRArgs) {
    return this.mul(asr(args));
  }

  duration(sustain: number) {
    return this.asr({ attack: 0.01, sustain, release: 0.01 });
  }

  pluck(release = 1) {
    return this.asr({ attack: 0.001, release });
  }

  at(timeInSeconds: number) {
    return this.tshift(constant(timeInSeconds));
  }

  tshift(amountInSeconds: number | Stem) {
    return new TimeShift(this, asStem(amountInSeconds));
  }

  tscale(factor: number | Stem) {
    return new TimeScale(this, asStem(factor));
  }

  repeat(periodInSeconds: number) {
    return new Repeat(this, periodInSeconds);
  }

  map(valueFn: (value: number) => number) {
    return new Map(this, valueFn);
  }
}

// #endregion interface

// #region sources

class State extends Stem {
  constructor(public value: number) {
    super();
  }

  valueAt(_t: number) {
    return this.value;
  }
}

class Ref extends Stem {
  constructor(public source: Stem) {
    super();
  }

  valueAt(t: number) {
    return this.source ? this.source.valueAt(t) : 0;
  }
}

class Constant extends Stem {
  constructor(public readonly value: number) {
    super();
  }

  valueAt(_t: number) {
    return this.value;
  }
}

class WhiteNoise extends Stem {
  valueAt(_t: number) {
    return 2 * random() - 1;
  }
}

// Important: these guys are stateful, so aliasing is not allowed!
// (E.g., if you `mix` the same `sine` stem w/ itself, you'll get
// a sine that's 2x the frequency.)
abstract class Osc extends Stem {
  public phase = 0;

  constructor(
    public readonly freq: Stem,
    private readonly phaseToValue: (phase: number) => number,
  ) {
    super();
  }

  valueAt(t: number) {
    const freq = this.freq.valueAt(t);
    this.phase += (TAU * freq) / sampleRate;
    return this.phaseToValue(this.phase);
  }
}

class Sine extends Osc {
  constructor(freq: Stem) {
    super(freq, sin);
  }
}

class Square extends Osc {
  constructor(freq: Stem) {
    super(freq, phase => (sin(phase) > 0 ? 1 : -1));
  }
}

class Sawtooth extends Osc {
  constructor(freq: Stem) {
    super(freq, phase => ((phase / TAU) % 1) * 2 - 1);
  }
}

class SineN extends Osc {
  constructor(freq: Stem, overtoneCount: number) {
    super(
      freq,
      phase => {
        let ans = 0;
        for (let i = 1; i <= overtoneCount; i++) {
          ans += sin(i * phase) / overtoneCount;
        }
        return ans;
      }
    );
  }
}

class PinkNoise extends Stem {
  private static readonly bufferSize = 16;
  private readonly buffer: number[] = new Array(PinkNoise.bufferSize).fill(0);
  private currentIndex = 0;

  valueAt(_t: number) {
    // Generate white noise
    const whiteNoise = 2 * random() - 1;

    // Update the buffer with the new value
    this.buffer[this.currentIndex] = whiteNoise;
    this.currentIndex = (this.currentIndex + 1) % PinkNoise.bufferSize;

    // Calculate the pink noise by averaging the buffer values
    let pinkNoise = 0;
    for (const value of this.buffer) {
      pinkNoise += value;
    }
    pinkNoise /= PinkNoise.bufferSize;

    return pinkNoise;
  }
}

// TODO: add an instance variable for the recording's sample rate
// (to support recordings whose sample rates are different from
// the one that we're using in WebAudio)
export class RecordedSound extends Stem {
  constructor(
    public readonly samples: Float32Array,
    public numSamples: number
  ) {
    super();
  }

  valueAt(t: number) {
    if (t < 0) {
      return 0;
    }

    const sampleIdx = round(t * sampleRate);
    if (sampleIdx >= this.samples.length) {
      return 0;
    }

    // TODO: use linear interpolation to compute better sample value
    //   x[n + f] = (1 - f) * x[n] + f * x[n + 1] (needs to wrap around)
    // (what we're doing right now is called "nearest neighbor")
    return this.samples[sampleIdx];
  }

  expand() {
    let maxAmplitude = 0;
    for (let idx = 0; idx < this.samples.length; idx++) {
      maxAmplitude = max(maxAmplitude, abs(this.samples[idx]));
    }

    const m = 1 / maxAmplitude;
    for (let idx = 0; idx < this.samples.length; idx++) {
      this.samples[idx] *= m;
    }

    return this;
  }
}

export class ASREnvelope extends Stem {
  private noteOffPending = false;
  private onReleaseEnded?: () => void;

  constructor(
    public attack: number,
    public readonly sustainLevel: number,
    public sustain: number,
    public readonly release: number,
    public readonly curve?: (value: number) => number
  ) {
    super();
  }

  noteOff(onReleaseEnded: () => void) {
    this.noteOffPending = true;
    this.onReleaseEnded = onReleaseEnded;
  }

  valueAt(t: number) {
    if (this.noteOffPending) {
      if (t < this.attack) {
        this.attack = this.sustain = 0;
      } else if (t < this.attack + this.sustain) {
        this.sustain = t - this.attack;
      }
      this.noteOffPending = false;
    }

    const curve = this.curve ?? (v => v);

    if (t < 0) {
      return 0;
    }

    if (t < this.attack) {
      const frac = t / this.attack;
      return curve(frac) * this.sustainLevel;
    }

    t -= this.attack;
    if (t < this.sustain) {
      return this.sustainLevel;
    }

    t -= this.sustain;
    if (t < this.release) {
      const frac = t / this.release;
      return curve(1 - frac) * this.sustainLevel;
    }

    if (this.onReleaseEnded) {
      this.onReleaseEnded();
      this.onReleaseEnded = undefined;
    }

    return 0;
  }
}

class Ramp extends Stem {
  constructor(
    public readonly time: number,
    public readonly initialValue: number,
    public readonly finalValue: number,
    public readonly curveFn?: (value: number) => number
  ) {
    super();
  }

  valueAt(t: number) {
    let m: number;
    if (t < 0) {
      m = this.initialValue;
    } else if (t < this.time) {
      const frac = t / this.time;
      m = this.initialValue + frac * (this.finalValue - this.initialValue);
    } else {
      m = this.finalValue;
    }

    if (this.curveFn) {
      m = this.curveFn(m);
    }

    return m;
  }
}

// #endregion sources

// #region operators

class Add extends Stem {
  public readonly sources = new Set<Stem>();

  constructor(sources: Stem[]) {
    super();
    for (const source of sources) {
      this.plugIn(source);
    }
  }

  plugIn(source: Stem) {
    this.sources.add(source);
  }

  unplug(source: Stem) {
    this.sources.delete(source);
  }

  valueAt(t: number) {
    let ans = 0;
    for (const source of this.sources) {
      ans += source.valueAt(t);
    }
    return ans;
  }
}

class Mix extends Stem {
  constructor(public readonly sources: Stem[]) {
    super();
  }

  valueAt(t: number) {
    let ans = 0;
    for (const source of this.sources) {
      ans += source.valueAt(t);
    }
    return ans / this.sources.length;
  }
}

class Mul extends Stem {
  constructor(public readonly sources: Stem[]) {
    super();
  }

  valueAt(t: number) {
    let ans = 1;
    for (const source of this.sources) {
      ans *= source.valueAt(t);
    }
    return ans;
  }
}

class Div extends Stem {
  constructor(
    public readonly dividend: Stem,
    public readonly divisor: Stem
  ) {
    super();
  }

  valueAt(t: number) {
    return this.dividend.valueAt(t) / this.divisor.valueAt(t);
  }
}

// based on https://www.earlevel.com/main/2012/11/26/biquad-c-source-code/
class BiquadFilter extends Stem {
  private z1 = 0;
  private z2 = 0;

  constructor(
    public readonly type: 'hp' | 'lp',
    public readonly source: Stem,
    public readonly freqCutoff: Stem,
    public readonly q: Stem
  ) {
    super();
  }

  valueAt(t: number) {
    // fc comes in the range [0, 1] but Redmon's stuff wants [0, 0.5]
    const fc = max(0, min(this.freqCutoff.valueAt(t), 1)) / 2;
    const q = this.q.valueAt(t);
    const k = tan(PI * fc);

    let norm: number;
    let a0: number;
    let a1: number;
    let a2: number;
    let b1: number;
    let b2: number;

    if (this.type === 'lp') {
      norm = 1 / (1 + k / q + k * k);
      a0 = k * k * norm;
      a1 = 2 * a0;
      a2 = a0;
      b1 = 2 * (k * k - 1) * norm;
      b2 = (1 - k / q + k * k) * norm;
    } else {
      norm = 1 / (1 + k / q + k * k);
      a0 = 1 * norm;
      a1 = -2 * a0;
      a2 = a0;
      b1 = 2 * (k * k - 1) * norm;
      b2 = (1 - k / q + k * k) * norm;
    }

    const input = this.source.valueAt(t);
    const output = input * a0 + this.z1;
    this.z1 = input * a1 + this.z2 - b1 * output;
    this.z2 = input * a2 - b2 * output;

    return output;
  }
}

class TimeShift extends Stem {
  constructor(
    public readonly source: Stem,
    public readonly amountInSeconds: Stem
  ) {
    super();
  }

  valueAt(t: number) {
    return this.source.valueAt(t - this.amountInSeconds.valueAt(t));
  }
}

class TimeScale extends Stem {
  constructor(
    public readonly source: Stem,
    public readonly factor: Stem
  ) {
    super();
  }

  valueAt(t: number) {
    return this.source.valueAt(t * this.factor.valueAt(t));
  }
}

class Repeat extends Stem {
  constructor(
    public readonly source: Stem,
    public readonly period: number
  ) {
    super();
  }

  valueAt(t: number) {
    return this.source.valueAt(t % this.period);
  }
}

class Map extends Stem {
  constructor(
    public readonly source: Stem,
    public readonly valueFn: (value: number) => number
  ) {
    super();
  }

  valueAt(t: number) {
    return this.valueFn(this.source.valueAt(t));
  }
}

function asStem(x: number | Stem): Stem {
  return typeof x === 'number' ? constant(x) : x;
}

// #endregion operators

// #region exports

export function state(initialValue = 0) {
  return new State(initialValue);
}

export function constant(value: number) {
  return new Constant(value);
}

export const silence = new Constant(0);

export const whiteNoise = new WhiteNoise();

export const pinkNoise = new PinkNoise();

export function sine(freq: number | Stem) {
  return new Sine(asStem(freq));
}

export function square(freq: number | Stem) {
  return new Square(asStem(freq));
}

export function sawtooth(freq: number | Stem) {
  return new Sawtooth(asStem(freq));
}

export function sinen(freq: number | Stem, overtoneCount: number) {
  return new SineN(asStem(freq), overtoneCount);
}

class WaveTableOsc extends Osc {
  constructor(freq: Stem, waveTable: number[], freq0: Stem) {
    super(
      freq.div(freq0).mul(sampleRate / waveTable.length),
      phase => {
        const fracIdx = ((phase % TAU) / TAU) * waveTable.length;
        const idx1 = Math.floor(fracIdx);
        const idx2 = (idx1 + 1) % waveTable.length;
        const amt2 = fracIdx - idx1;
        const amt1 = 1 - amt2;
        return amt1 * waveTable[idx1] + amt2 * waveTable[idx2];
      }
    );
  }
}

function makeWaveTable(fn: (phase: number) => number, size: number) {
  const waveTable = new Array(size);
  for (let idx = 0; idx < size; idx++) {
    const phase = (idx / size) * TAU;
    waveTable[idx] = fn(phase);
  }
  return waveTable;
}

const waveTables = {
  sine: makeWaveTable(Math.sin, 1024),
  square: makeWaveTable(phase => (Math.sin(phase) >= 0 ? 1 : -1), 1024),
  sawtooth: makeWaveTable(phase => (2 * phase) / TAU - 1, 1024),
};

export function wavetable(
  freq: number | Stem,
  wt: number[],
  freqMultiplier: number | Stem = 1
) {
  return new WaveTableOsc(asStem(freq), wt, asStem(freqMultiplier));
}

export function wtSine(freq: number | Stem) {
  return wavetable(freq, waveTables.sine);
}

export function wtSquare(freq: number | Stem) {
  return wavetable(freq, waveTables.square);
}

export function wtSawtooth(freq: number | Stem) {
  return wavetable(freq, waveTables.sawtooth);
}

export function ramp(args: {
  time: number;
  initialValue?: number;
  finalValue?: number;
  curve?(value: number): number;
}) {
  const initialValue = args.initialValue ?? 0;
  const finalValue = args.finalValue ?? 1;
  return new Ramp(args.time, initialValue, finalValue, args.curve);
}

export function asr(args: ASRArgs) {
  const attack = args.attack ?? 0;
  const sustainLevel = args.sustainLevel ?? 1;
  const sustain = args.sustain ?? 0;
  const release = args.release ?? 0;
  return new ASREnvelope(attack, sustainLevel, sustain, release, args.curve);
}

export function recordedSound(
  samples: Float32Array,
  numSamples = samples.length
) {
  return new RecordedSound(samples, numSamples);
}

export function add(...args: (number | Stem)[]) {
  return new Add(args.map(asStem));
}

export function mix(...args: (number | Stem)[]) {
  return new Mix(args.map(asStem));
}

export function mul(...args: (number | Stem)[]) {
  return new Mul(args.map(asStem));
}

export function div(dividend: number | Stem, divisor: number | Stem) {
  return new Div(asStem(dividend), asStem(divisor));
}

// #endregion exports
