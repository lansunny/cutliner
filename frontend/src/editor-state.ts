import type { Candidate } from './studio-types';

export type ClipRange = [number, number];
export function formatTimecode(seconds: number, precise = false) {
  if (!Number.isFinite(seconds)) return '00:00';
  const ticks = precise ? Math.round(Math.max(0, seconds) * 100) : Math.floor(Math.max(0, seconds)) * 100;
  const whole = Math.floor(ticks / 100);
  return `${String(Math.floor(whole / 60)).padStart(2, '0')}:${String(whole % 60).padStart(2, '0')}${precise ? '.' + String(ticks % 100).padStart(2, '0') : ''}`;
}
export const sameRange = (first: ClipRange, second: ClipRange) => first.length === 2 && second.length === 2 && first.every(Number.isFinite) && second.every(Number.isFinite) && first.every((value, i) => Math.abs(value - second[i]) < .00001);
export const modelRange = (candidate: Candidate): ClipRange => [candidate.model_start_seconds ?? candidate.start_seconds, candidate.model_end_seconds ?? candidate.end_seconds];
export const reviewedRange = (candidate: Candidate | undefined, range: ClipRange) => Boolean(candidate?.review_verdict === 'pass' && candidate.review_scope && sameRange(range, [candidate.review_scope.start, candidate.review_scope.end]));

const draftKey = (media: string, run: string, candidate: string) => `cutliner:trim:v1:${media}:${run}:${candidate}`;

export function restoreRange(media: string, run: string, candidate: Candidate): ClipRange {
  const current: ClipRange = [candidate.start_seconds, candidate.end_seconds];
  try {
    const draft = JSON.parse(localStorage.getItem(draftKey(media, run, candidate.candidate_id)) || 'null');
    if (draft && Array.isArray(draft.base) && draft.base.length === 2 && sameRange(current, draft.base)
      && Array.isArray(draft.range) && draft.range.length === 2 && draft.range.every(Number.isFinite)
      && Number.isFinite(draft.duration) && draft.range[0] >= 0 && draft.range[1] > draft.range[0]
      && draft.range[1] <= draft.duration) return draft.range;
  } catch { /* Storage may be disabled; the server copy remains available. */ }
  return current;
}

export function saveDraft(media: string, run: string, candidate: Candidate, range: ClipRange, duration: number) {
  if (!range.every(Number.isFinite) || range[0] < 0 || range[0] >= range[1] || range[1] > duration) return;
  try {
    const key = draftKey(media, run, candidate.candidate_id);
    const base: ClipRange = [candidate.start_seconds, candidate.end_seconds];
    if (sameRange(base, range)) localStorage.removeItem(key);
    else localStorage.setItem(key, JSON.stringify({ base, range, duration }));
  } catch { /* Export and review still persist their exact scope on the server. */ }
}
