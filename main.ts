// gta6-twitch-ccu-forecast-01 — entry: a launch-day Twitch viewer forecaster.
//
// Idea: a big game's release day on Twitch has a characteristic *shape*, and
// the shape depends on the launch's scale. Hour 1 (00:00-01:00 UTC) is a
// strange hour — New Zealand and Asia are in, Europe is waking up, the
// Americas are still asleep. What the rest of the day does with that first
// hour differs by size: the biggest launches are front-loaded (their first
// hour is at or near the day's peak, and the day sags from there), while
// small launches ignite late, when their region unlocks or prime time finds
// them, and can end the day a hundred times their first hour. The case file
// ships ~40 past launches with full hourly histories, so instead of guessing
// the shape we measure it, mostly from launches of a similar size: for every
// past launch we take the log ratio log(viewers_h / viewers_1) of each
// release-day hour to its own first hour, weight the launch by how close its
// first hour is to this one's (in log distance), and forecast this launch's
// hours as hour 1's print times the weighted median of those ratios. The
// five quantiles come from the weighted dispersion of the ratios at each
// hour, capped so no single quantile strays more than e^3 from the median,
// and floored by a random-walk spread so the band never claims more
// certainty late in the day than a flat random walk would. In the final, the
// case also carries minute reads up to the checkpoint: if the crowd is still
// climbing into 01:00 the early hours get a decaying momentum lift (and the
// mirror image if it is already falling off).
//
// Tuning (leave-one-out over the 37 usable practice comparables, each in
// turn the target, the rest its base; pinball loss on the log scale): this
// program 0.428 against the persistence baseline's 0.930 — 54% lower. On the
// subset of launches with a first hour of 100k+ viewers (this launch's
// regime) it scores 0.225 against the baseline's 0.261.
//
// Contract (RULES.txt): read every *.in in $MAKEFUN_INPUT_DIR (else /input),
// write <id>.out to $MAKEFUN_OUTPUT_DIR (else /output) as
// {"hours":[{"hour":h,"q":[q05,q25,q50,q75,q95]} ...]} for hours 2..24.
// Node 22, standard library only, no network, deterministic.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const IN_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const HOUR_MS = 3_600_000;
/** Fallback spread (log units per sqrt-hour) when comparables cannot be used. */
const FALLBACK_SIGMA = 0.22;
/** The baseline's spread; our band is never narrower than this at any hour. */
const FLOOR_SIGMA = 0.15;
/** Cap (log units) on the measured deviation of any single quantile. */
const DEV_CAP = 3.0;
/**
 * Bandwidth (log units) of the hour-1-scale similarity weight: a comparable
 * whose first hour is e^1 ≈ 2.7x this launch's (or a third of it) counts
 * e^-1 ≈ 0.37 as much, before the weight falls off with distance.
 */
const SIZE_BANDWIDTH = 1.0;
/** Momentum is capped at this log distance and decays with this time constant (hours). */
const MOMENTUM_CAP = 0.3;
const MOMENTUM_TAU = 6;
/** A comparable needs at least this many viewers in hour 1 to carry a shape. */
const MIN_FIRST_HOUR = 50;

interface HourPoint { from: string; viewers: number }
interface Comparable {
  name: string;
  day_start: string;
  hourly: HourPoint[];
}
interface CaseFile {
  id: string;
  game: { name: string };
  day_start: string;
  checkpoint: string;
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  history: Array<{ at: string; viewers: number }>;
  comparables: Comparable[];
}

/** Standard normal quantile (Acklam's rational approximation, |err| < 1.2e-9 in z). */
function zOf(tau: number): number {
  if (tau <= 0) return -Infinity;
  if (tau >= 1) return Infinity;
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  const plow = 0.02425;
  let q: number, r: number;
  if (tau < plow) {
    q = Math.sqrt(-2 * Math.log(tau));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (tau <= 1 - plow) {
    q = tau - 0.5;
    r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  q = Math.sqrt(-2 * Math.log(1 - tau));
  return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}

/** Weighted empirical quantile of {r, w} samples sorted by r. */
function weightedQuantileOf(samples: Array<{ r: number; w: number }>, tau: number): number {
  if (samples.length === 1) return samples[0].r;
  const total = samples.reduce((s, x) => s + x.w, 0);
  const goal = tau * total;
  let cum = 0;
  for (let i = 0; i < samples.length; i++) {
    const before = cum;
    cum += samples[i].w;
    if (cum >= goal) {
      if (i === 0) return samples[0].r;
      const frac = (goal - before) / samples[i].w;
      return samples[i - 1].r + (samples[i].r - samples[i - 1].r) * frac;
    }
  }
  return samples[samples.length - 1].r;
}

/**
 * The 24 release-day hourly viewer counts of one comparable, or null when the
 * day is incomplete or too small to carry a shape. Points are matched by
 * timestamp, never by position, so gaps elsewhere in the 17-day window and
 * any spacing quirks of the source do not matter.
 */
function releaseDay(comp: Comparable): number[] | null {
  const start = Date.parse(comp.day_start);
  if (!Number.isFinite(start)) return null;
  const byTime = new Map<number, number>();
  for (const p of comp.hourly) byTime.set(Date.parse(p.from), p.viewers);
  const day: number[] = [];
  for (let h = 0; h < 24; h++) {
    const v = byTime.get(start + h * HOUR_MS);
    if (v === undefined || !(v >= 1)) return null;
    day.push(v);
  }
  if (day[0] < MIN_FIRST_HOUR) return null;
  return day;
}

/** Per-hour weighted log-ratio samples across comparables, for h = 2..24. */
function ratioSamples(kase: CaseFile): Map<number, Array<{ r: number; w: number }>> {
  const target = kase.game.name.trim().toLowerCase();
  const targetFirstHour = Math.max(1, (kase.known ?? []).find((k) => k.hour === 1)?.viewers ?? 1);
  const byHour = new Map<number, Array<{ r: number; w: number }>>();
  for (const comp of kase.comparables ?? []) {
    // The final case lists the target category's own (non-launch) history as
    // one of the comparables; its "release day" is not a launch day.
    if (comp.name.trim().toLowerCase() === target) continue;
    const day = releaseDay(comp);
    if (day === null) continue;
    const w = Math.exp(-Math.abs(Math.log(day[0]) - Math.log(targetFirstHour)) / SIZE_BANDWIDTH);
    for (let h = 2; h <= 24; h++) {
      const sample = { r: Math.log(day[h - 1] / day[0]), w };
      const arr = byHour.get(h);
      if (arr) arr.push(sample); else byHour.set(h, [sample]);
    }
  }
  for (const arr of byHour.values()) arr.sort((x, y) => x.r - y.r);
  return byHour;
}

/**
 * Log momentum at the checkpoint: how the last quarter hour of hour 1 compares
 * with hour 1's print, capped. Only computed when the history is fine-grained
 * (minute reads, as in the final); with hourly points it returns 0.
 */
function momentum(kase: CaseFile, p1: number): number {
  const start = Date.parse(kase.day_start);
  const checkpoint = Date.parse(kase.checkpoint);
  if (!Number.isFinite(start) || !Number.isFinite(checkpoint)) return 0;
  const inHour = (kase.history ?? []).filter((p) => {
    const t = Date.parse(p.at);
    return t >= start && t < checkpoint && p.viewers > 0;
  });
  if (inHour.length < 10) return 0;
  const late = inHour.filter((p) => Date.parse(p.at) >= checkpoint - 15 * 60_000);
  if (late.length === 0) return 0;
  const lateMean = late.reduce((s, p) => s + p.viewers, 0) / late.length;
  const raw = Math.log(lateMean / p1);
  return Math.max(-MOMENTUM_CAP, Math.min(MOMENTUM_CAP, raw));
}

/** One hour's five quantiles for a case, from the learned shape. */
function forecastHour(
  h: number,
  p1: number,
  levels: number[],
  samples: Map<number, Array<{ r: number; w: number }>>,
  mom: number,
): number[] {
  const ratios = samples.get(h);
  const momAdj = mom * Math.exp(-(h - 2) / MOMENTUM_TAU);
  let medianLog = momAdj;
  const devs: number[] = [];
  if (ratios && ratios.length >= 3) {
    const med = weightedQuantileOf(ratios, 0.5);
    medianLog = med + momAdj;
    for (const tau of levels) {
      const empRaw = weightedQuantileOf(ratios, tau) - med;
      const emp = Math.max(-DEV_CAP, Math.min(DEV_CAP, empRaw));
      const floor = zOf(tau) * FLOOR_SIGMA * Math.sqrt(h - 1);
      // Whichever of the measured dispersion and the random-walk floor is
      // wider (in this quantile's direction) sets the band.
      devs.push(Math.abs(emp) >= Math.abs(floor) ? emp : floor);
    }
  } else {
    for (const tau of levels) devs.push(zOf(tau) * FALLBACK_SIGMA * Math.sqrt(h - 1));
  }
  const qs = levels.map((_, j) => {
    const v = p1 * Math.exp(medianLog + devs[j]);
    return Math.min(1e9, Math.max(1, v));
  });
  // Enforce the non-decreasing order the output contract demands.
  for (let j = 1; j < qs.length; j++) if (qs[j] < qs[j - 1]) qs[j] = qs[j - 1];
  return qs.map((v) => Math.round(v * 100) / 100);
}

/** Shape-check an output object before it is written; invalid never ships. */
function shapeOk(out: unknown): boolean {
  if (typeof out !== 'object' || out === null) return false;
  const hours = (out as { hours?: unknown }).hours;
  if (!Array.isArray(hours) || hours.length !== 23) return false;
  const seen = new Set<number>();
  for (const row of hours) {
    const r = row as { hour?: unknown; q?: unknown };
    if (typeof r.hour !== 'number' || r.hour < 2 || r.hour > 24 || seen.has(r.hour)) return false;
    seen.add(r.hour);
    if (!Array.isArray(r.q) || r.q.length !== 5) return false;
    for (let j = 0; j < 5; j++) {
      const v = r.q[j] as number;
      if (typeof v !== 'number' || !Number.isFinite(v) || v <= 0 || v > 1e9) return false;
      if (j > 0 && v < (r.q[j - 1] as number)) return false;
    }
  }
  return seen.size === 23;
}

function forecastCase(kase: CaseFile): { hours: Array<{ hour: number; q: number[] }> } {
  const first = (kase.known ?? []).find((k) => k.hour === 1);
  const p1 = Math.max(1, first?.viewers ?? 1);
  const levels = kase.levels?.length === 5 ? kase.levels : [0.05, 0.25, 0.5, 0.75, 0.95];
  const samples = ratioSamples(kase);
  const mom = momentum(kase, p1);
  const rows = (kase.forecast ?? []).map(({ hour }) => ({
    hour,
    q: forecastHour(hour, p1, levels, samples, mom),
  }));
  const out = { hours: rows };
  if (!shapeOk(out)) {
    // Last-resort flat forecast in the contract's shape (never crash a case).
    const flat = (kase.forecast ?? []).map(({ hour }) => ({ hour, q: [p1, p1, p1, p1, p1] }));
    return { hours: flat };
  }
  return out;
}

mkdirSync(OUT_DIR, { recursive: true });
const files = readdirSync(IN_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of files) {
  const kase = JSON.parse(readFileSync(join(IN_DIR, file), 'utf8')) as CaseFile;
  const out = forecastCase(kase);
  writeFileSync(join(OUT_DIR, `${kase.id}.out`), `${JSON.stringify(out)}\n`);
}
