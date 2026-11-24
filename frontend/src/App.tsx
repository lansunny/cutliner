import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { flowPalettes } from './flowPalettes';
import { ArrowDownToLine, ArrowRight, ArrowUpRight, AudioLines, Check, ChevronDown, CircleHelp, Clapperboard, Expand, Film, History, Layers3, LoaderCircle, MousePointer2, PanelRight, Pause, Play, Plus, RotateCcw, ScanLine, Scissors, Search, Settings2, ShieldCheck, SkipBack, SkipForward, SlidersHorizontal, Square, Star, Volume2, VolumeX, WifiOff, Workflow, X } from 'lucide-react';
import AgentConsole from './AgentConsole';
import ArtifactVersions, { artifactStatus } from './ArtifactVersions';
import type { Artifact, Bootstrap, Candidate, Delivery, Job, Media, Run } from './studio-types';
import { formatTimecode as timecode, modelRange, restoreRange, reviewedRange, sameRange, saveDraft } from './editor-state';
declare global { interface Window { LiquidGlass?: { refresh: () => void; rebuild: () => void } } }

const routeNames: Record<string, string> = { visual: '视觉', audio_visual: '声画', hybrid: '联合' };
const artifactMedia = (artifact?: Artifact) => artifact?.media ? { ...artifact.media, filename: `cutliner_v${String(artifact.version).padStart(2, '0')}.mp4` } : null;
function historyLabel(run: Run) {
  if (run.review_status === 'manual_exported') return '手动导出';
  if (run.review_status === 'review_blocked') return '复核待确认';
  if (run.review_status === 'operation_failed') return '本次操作未完成';
  if (run.review_status === 'cancelled') return '已停止';
  if (run.review_status === 'interrupted') return '服务重启 · 已中断';
  if (run.review_status === 'no_match') return '当前范围未通过';
  return run.delivery?.reviewed ? 'Agent 已交付' : `${routeNames[run.route]} · ${run.candidates.length} 个候选`;
}
const FlowBackground = lazy(() => import('./FlowBackground'));
const IconButton = ({ label, children, className = '', ...props }: { label: string; children: ReactNode; className?: string } & React.ButtonHTMLAttributes<HTMLButtonElement>) => <button type="button" className={`icon-button lg ${className}`} aria-label={label} title={label} {...props}>{children}</button>;

export default function App() {
  const [data, setData] = useState<Bootstrap | null>(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [activeId, setActiveId] = useState('');
  const [runId, setRunId] = useState('');
  const [candidateId, setCandidateId] = useState('');
  const [query, setQuery] = useState('');
  const [visualEn, setVisualEn] = useState('');
  const [route, setRoute] = useState('visual');
  const [events, setEvents] = useState<string[]>(['Truck']);
  const [cloudPlanning, setCloudPlanning] = useState(false);
  const [cloudReview, setCloudReview] = useState(false);
  const [mode, setMode] = useState<'agent' | 'manual'>('agent');
  const [autoConsent, setAutoConsent] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [position, setPosition] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [trim, setTrim] = useState<[number, number]>([0, 10]);
  const [wave, setWave] = useState<number[]>([]);
  const [waveStatus, setWaveStatus] = useState('loading');
  const [waveDuration, setWaveDuration] = useState(0);
  const [job, setJob] = useState<Job | null>(null);
  const [connection, setConnection] = useState<'connecting' | 'connected' | 'reconnecting'>('connecting');
  const [stopping, setStopping] = useState(false);
  const [retryConsent, setRetryConsent] = useState(false);
  const [section, setSection] = useState<'media' | 'history'>('media');
  const [libraryOpen, setLibraryOpen] = useState(() => window.matchMedia('(min-width: 761px)').matches);
  const [searchOpen, setSearchOpen] = useState(true);
  const [recordsOpen, setRecordsOpen] = useState(false);
  const [libraryQuery, setLibraryQuery] = useState('');
  const [savingFavorite, setSavingFavorite] = useState<string | null>(null);
  const [importPath, setImportPath] = useState('');
  const [importing, setImporting] = useState(false);
  const [output, setOutput] = useState<Media | null>(null);
  const [delivery, setDelivery] = useState<Delivery | null>(null);
  const [artifactId, setArtifactId] = useState('');
  const [loadingArtifact, setLoadingArtifact] = useState(false);
  const [unsupportedMedia, setUnsupportedMedia] = useState('');
  const [previewRange, setPreviewRange] = useState<{ url: string; start: number; end: number } | null>(null);
  const [onlyReviewed, setOnlyReviewed] = useState(false);
  const [fit, setFit] = useState(true);
  const [glass, setGlass] = useState(true);
  const [backgroundPaused, setBackgroundPaused] = useState(false);
  const [palette, setPalette] = useState(0);
  const [duration, setDuration] = useState(0);
  const [showSource, setShowSource] = useState(false);
  const [outputPosition, setOutputPosition] = useState(0);
  const [outputPlaying, setOutputPlaying] = useState(false);
  const video = useRef<HTMLVideoElement>(null);
  const outputVideo = useRef<HTMLVideoElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const importDialog = useRef<HTMLDialogElement>(null);
  const exportDialog = useRef<HTMLDialogElement>(null);
  const settingsDialog = useRef<HTMLDialogElement>(null);
  const helpDialog = useRef<HTMLDialogElement>(null);
  const autoDialog = useRef<HTMLDialogElement>(null);
  const retryDialog = useRef<HTMLDialogElement>(null);
  const previewDialog = useRef<HTMLDialogElement>(null);
  const recordsDialog = useRef<HTMLDialogElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const timeline = useRef<HTMLDivElement>(null);
  const trimHistory = useRef<[number, number][]>([]);
  const handledJob = useRef('');
  const liveRunSeen = useRef('');
  const pendingDraft = useRef<() => void>(() => {});
  const sessionRef = useRef<Bootstrap | null>(null);
  const jobRef = useRef<Job | null>(null);
  const reconnecting = useRef(false);
  sessionRef.current = data;
  jobRef.current = job;
  const media = data?.media.find(m => m.id === activeId);
  const filter = libraryQuery.trim().toLocaleLowerCase();
  const libraryMedia = (data?.media ?? []).filter(item => `${item.name} ${item.filename}`.toLocaleLowerCase().includes(filter));
  const libraryRuns = (data?.runs ?? []).filter(run => `${run.request} ${run.task.query ?? ''}`.toLocaleLowerCase().includes(filter));
  const currentRun = data?.runs.find(r => r.id === runId && r.media_id === activeId);
  const selectedArtifact = currentRun?.deliveries?.find(item => item.id === artifactId);
  const hasOutputView = Boolean(output || selectedArtifact);
  const viewingOutput = mode === 'agent' && hasOutputView && !showSource;
  const candidates = currentRun?.candidates ?? [];
  const candidate = candidates.find(c => c.candidate_id === candidateId);
  const visibleCandidates = onlyReviewed ? candidates.filter(c => c.review_verdict === 'pass') : candidates;
  const jobActive = job?.status === 'running' || job?.status === 'cancelling';
  const busy = jobActive || importing || submitting;
  const sourceDuration = media?.duration || 10;
  const total = Math.min(duration || sourceDuration, sourceDuration);
  const sourceRange: [number, number] = candidate || mode === 'manual' ? trim : [0, total];
  const trimChanged = Boolean(candidate && !sameRange(trim, modelRange(candidate)));
  const trimReviewed = reviewedRange(candidate, trim);
  const contextJob = job && (!job.media_id || job.media_id === activeId) && (!job.run || job.run.media_id === activeId) && (!job.result?.run || job.result.run.id === runId) ? job : null;
  const viewContext = useRef({ runId, activeId, busy });
  viewContext.current = { runId, activeId, busy };
  pendingDraft.current = () => {
    if (mode === 'manual' && !busy && candidate && runId && media) saveDraft(activeId, runId, candidate, trim, total);
  };

  const selectRun = useCallback((run: Run) => {
    pendingDraft.current();
    setShowSource(false);
    setActiveId(run.media_id); setRunId(run.id); setQuery(run.request || run.task.query || '');
    setVisualEn(run.task.visual_query_en || ''); setRoute(run.route);
    setEvents(run.task.audio_event_classes?.length ? run.task.audio_event_classes : ['Truck']);
    const saved = run.delivery;
    setArtifactId(run.latest_artifact_id || saved?.id || '');
    const c = run.candidates.find(c => c.candidate_id === run.active_candidate_id) || run.candidates.find(c => c.candidate_id === saved?.candidate_id) || run.candidates[0];
    const range = c ? restoreRange(run.media_id, run.id, c) : [0, 10] as [number, number];
    setCandidateId(c?.candidate_id || '');
    setTrim(range);
    setOutput(saved?.media || null); setDelivery(saved || null); setPosition(range[0]); trimHistory.current = [];
  }, []);

  const reconnect = useCallback(async () => {
    if (reconnecting.current) return;
    reconnecting.current = true;
    try {
      const response = await fetch('/api/bootstrap', { signal: AbortSignal.timeout(5000) });
      if (!response.ok) throw new Error('本地服务暂未连接');
      const next: Bootstrap = await response.json();
      const previous = sessionRef.current;
      const restarted = previous && previous.instance_id !== next.instance_id;
      sessionRef.current = next;
      setData(next); setConnection('connected');
      if (!previous) {
        if (next.runs[0]) selectRun(next.runs[0]); else if (next.media[0]) { setActiveId(next.media[0].id); setTrim([0, next.media[0].duration]); }
      }
      const priorJob = jobRef.current;
      let recovered = next.active_job || (!previous ? next.latest_job : null);
      if (restarted && priorJob && !recovered) {
        if (next.latest_job?.id === priorJob.id) recovered = next.latest_job;
        else {
          const response = await fetch(`/api/jobs/${priorJob.id}`, { signal: AbortSignal.timeout(5000) });
          if (response.ok) recovered = await response.json() as Job;
          else if (response.status !== 404) throw new Error('原任务状态暂不可用');
        }
      }
      if (recovered) {
        setJob(recovered);
        const run = recovered.result?.run || recovered.run || next.runs.find(r => r.id === recovered.run_id);
        const restoreView = !previous || ['running', 'cancelling'].includes(recovered.status) || Boolean(priorJob && ['running', 'cancelling'].includes(priorJob.status));
        if (restoreView && run) selectRun(run);
        else if (restoreView && recovered.media_id) { setActiveId(recovered.media_id); setRunId(''); setOutput(null); setDelivery(null); setQuery(recovered.request || recovered.scope?.query || ''); }
      } else if (restarted && jobRef.current && ['running', 'cancelling'].includes(jobRef.current.status)) {
        setJob({ ...jobRef.current, status: 'interrupted', stage: '服务重启，未找到原任务', can_retry: false });
      }
      if (restarted) setNotice('服务已重新连接；未自动重跑任务');
    } catch { setConnection('reconnecting'); }
    finally { reconnecting.current = false; }
  }, [selectRun]);

  const post = useCallback(async (path: string, body: unknown) => {
    let response: Response;
    try {
      response = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Studio-Token': sessionRef.current?.token || '' }, body: JSON.stringify(body), signal: AbortSignal.timeout(15000) });
    } catch {
      setConnection('reconnecting'); void reconnect();
      throw new Error('连接中断，正在确认任务状态。请勿重复提交。');
    }
    const value = await response.json();
    if (response.status === 403) {
      await reconnect();
      throw new Error('本地连接凭证已刷新，请确认任务状态后重新操作');
    }
    if (!response.ok) throw new Error(value.error || '请求失败');
    return value;
  }, [reconnect]);

  async function toggleFavorite(item: Media) {
    if (savingFavorite !== null) return;
    setSavingFavorite(item.id);
    try {
      const { media: updated } = await post(`/api/media/${item.id}/preferences`, { favorite: !item.favorite }) as { media: Media };
      setData(old => old ? { ...old, media: old.media.map(row => row.id === updated.id ? updated : row).sort((a,b) => Number(Boolean(b.favorite)) - Number(Boolean(a.favorite)) || (b.updated_at ?? 0) - (a.updated_at ?? 0)) } : old);
    } catch (e) { setError(e instanceof Error ? e.message : '收藏失败'); }
    finally { setSavingFavorite(null); }
  }

  function openRangePreview() {
    if (!media || media.duration <= 0) return;
    const useTrim = Boolean(candidate) || mode === 'manual';
    const start = Math.min(Math.max(0, useTrim ? trim[0] : position), Math.max(0, media.duration - .1));
    const end = Math.min(media.duration, start + 30, useTrim && trim[1] > start ? trim[1] : media.duration);
    if (end <= start) return;
    video.current?.pause();
    setPreviewRange({ url: `/api/preview/${media.id}?start=${start}&end=${end}`, start, end });
    previewDialog.current?.showModal();
  }

  useEffect(() => {
    void reconnect();
    const timer = setInterval(() => { if (!jobRef.current || !['running', 'cancelling'].includes(jobRef.current.status)) void reconnect(); }, 8000);
    return () => clearInterval(timer);
  }, [reconnect]);

  useEffect(() => {
    const refresh = () => window.LiquidGlass?.refresh();
    if (!document.querySelector('script[data-glass]')) {
      const script = document.createElement('script'); script.src = '/liquid-glass.js'; script.dataset.glass = 'true'; script.onload = refresh; document.body.appendChild(script);
    }
    const timer = window.setTimeout(refresh, 80); return () => clearTimeout(timer);
  }, [data, activeId, glass, playing, output, mode, job?.status, autoConsent]);

  useEffect(() => {
    if (!job?.run || job.status !== 'running' || liveRunSeen.current === job.run.id) return;
    liveRunSeen.current = job.run.id;
    const next = job.run;
    setData(old => old ? { ...old, runs: [next, ...old.runs.filter(r => r.id !== next.id)] } : old);
    selectRun(next);
  }, [job, selectRun]);

  useEffect(() => {
    const c = job?.run?.candidates.find(c => c.candidate_id === job.candidate_progress?.id);
    if (!c || job?.status !== 'running' || mode !== 'agent') return;
    setCandidateId(c.candidate_id); setTrim([c.start_seconds, c.end_seconds]); setPosition(c.start_seconds);
    if (video.current) video.current.currentTime = c.start_seconds;
  }, [job?.candidate_progress?.id, job?.status, job?.run?.id, mode]);

  useEffect(() => { setDuration(0); }, [media?.id]);

  useEffect(() => {
    if (!media || mode !== 'manual') return;
    setWave([]); setWaveStatus('loading');
    const controller = new AbortController();
    fetch(`/api/waveform/${media.id}`, { signal: controller.signal }).then(r => r.json()).then(w => {
      if (w.error) throw new Error(w.error);
      setWave(w.values); setWaveDuration(w.duration); setWaveStatus(w.has_audio ? 'ready' : 'silent');
    }).catch(e => { if (e.name !== 'AbortError') setWaveStatus('error'); });
    return () => controller.abort();
  }, [media?.id, mode]);

  useEffect(() => {
    if (video.current) video.current.playbackRate = speed;
    if (outputVideo.current) outputVideo.current.playbackRate = speed;
  }, [speed]);

  useEffect(() => { setPlaying(false); setOutputPlaying(false); setOutputPosition(0); }, [viewingOutput, activeId, output?.id]);

  useEffect(() => {
    if (!notice) return; const timer = setTimeout(() => setNotice(''), 6000); return () => clearTimeout(timer);
  }, [notice]);

  useEffect(() => {
    const artifact = currentRun?.deliveries?.find(item => item.id === artifactId);
    if (!artifact) return;
    setOutput(artifactMedia(artifact));
    setDelivery(artifact.media ? { ...artifact, media: artifact.media } : null);
  }, [currentRun?.deliveries, artifactId]);

  useEffect(() => {
    if (mode !== 'manual' || busy || !candidate || !runId || !media) return;
    const timer = window.setTimeout(() => saveDraft(activeId, runId, candidate, trim, total), 250);
    return () => window.clearTimeout(timer);
  }, [activeId, runId, candidate, trim, total, mode, busy, media]);

  useEffect(() => {
    const flush = () => pendingDraft.current();
    window.addEventListener('pagehide', flush);
    return () => window.removeEventListener('pagehide', flush);
  }, []);

  useEffect(() => {
    if (!job || !['running', 'cancelling'].includes(job.status)) return;
    let active = true;
    let pending = false;
    const poll = async () => {
      if (pending) return;
      pending = true;
      try {
        const response = await fetch(`/api/jobs/${job.id}`, { signal: AbortSignal.timeout(5000) });
        if (response.status === 404) {
          const refresh = await fetch('/api/bootstrap', { signal: AbortSignal.timeout(5000) });
          if (!refresh.ok) throw new Error('服务不可用');
          const latest: Bootstrap = await refresh.json();
          if (active) {
            sessionRef.current = latest; setData(latest); setConnection('connected');
            setJob(latest.active_job || { ...job, status: 'interrupted', stage: '当前服务未找到原任务，未自动重跑', can_retry: false });
          }
          return;
        }
        if (!response.ok) throw new Error('服务暂不可用');
        const result: Job = await response.json();
        if (active) {
          if (result.instance_id && result.instance_id !== sessionRef.current?.instance_id) await reconnect();
          setConnection('connected'); setJob(result);
        }
      } catch { if (active) setConnection('reconnecting'); }
      finally { pending = false; }
    };
    void poll();
    const timer = setInterval(poll, 1500); return () => { active = false; clearInterval(timer); };
  }, [job?.id, job?.status, reconnect]);

  useEffect(() => {
    if (!job || ['running', 'cancelling'].includes(job.status) || handledJob.current === job.id) return;
    handledJob.current = job.id;
    if (job.status !== 'complete') {
      if (job.run) {
        const next = job.run;
        setData(old => old ? { ...old, runs: [next, ...old.runs.filter(r => r.id !== next.id)] } : old);
        selectRun(next);
      }
      if (job.status === 'failed') setError(job.error || '任务未完成');
      else setNotice(job.status === 'cancelled' ? '已停止，已有记录和文件保留' : '任务已中断，未自动重跑');
      return;
    }
    if (job.result?.run) {
      const next = job.result.run;
      setData(old => old ? { ...old, runs: [next, ...old.runs.filter(r => r.id !== next.id)] } : old);
      selectRun(next);
    }
    if (job.result?.media) {
      const out = job.result.media;
      setOutput(out);
      const artifact = job.result.run?.deliveries?.find(item => item.media?.id === out.id);
      setDelivery(artifact ? { ...artifact, media: out } : { media: out, reviewed: Boolean(job.result.reviewed), candidate_id: job.result.candidate_id || candidateId, start: job.result.start, end: job.result.end });
      if (artifact) setArtifactId(artifact.id);
      if (job.result.candidate_id) setCandidateId(job.result.candidate_id);
      if (job.result.start !== undefined && job.result.end !== undefined) {
        setTrim([job.result.start, job.result.end]); setPosition(job.result.start);
      }
    }
    setNotice(job.result?.message || '任务已完成');
  }, [job, selectRun, runId, candidateId]);

  const seek = useCallback((at: number) => {
    const value = Math.max(0, Math.min(total, at));
    if (video.current) video.current.currentTime = value; setPosition(value);
  }, [total]);

  function seekSource(at: number) {
    if (!media) return;
    setShowSource(true);
    seek(at);
  }

  const seekOutput = useCallback((at: number) => {
    if (!outputVideo.current || !output) return;
    const value = Math.max(0, Math.min(output.duration, at));
    outputVideo.current.currentTime = value;
    setOutputPosition(value);
  }, [output]);

  async function chooseArtifact(id: string, preserveError = false) {
    if (!currentRun || busy || loadingArtifact) return;
    pendingDraft.current(); setLoadingArtifact(true);
    const context = { runId, activeId };
    try {
      const response = await fetch(`/api/runs/${runId}`, { signal: AbortSignal.timeout(5000) });
      const refreshed = await response.json() as Run & { error?: string };
      if (!response.ok) throw new Error(refreshed.error || '成片记录暂不可用');
      if (viewContext.current.runId !== context.runId || viewContext.current.activeId !== context.activeId || viewContext.current.busy) return;
      const artifact = refreshed.deliveries?.find(item => item.id === id);
      if (!artifact) throw new Error('该版本记录不存在，请重新选择');
      setData(old => old ? { ...old, runs: old.runs.map(run => run.id === refreshed.id ? refreshed : run) } : old);
      setArtifactId(id); setOutput(artifactMedia(artifact)); setDelivery(artifact.media ? { ...artifact, media: artifact.media } : null);
      setMode('agent'); setShowSource(false); if (!preserveError) setError('');
    } catch (e) { setError((e as Error).message); }
    finally { setLoadingArtifact(false); }
  }

  function applyArtifactRange(artifact: Artifact) {
    if (busy || artifact.source_current === false || artifact.start === undefined || artifact.end === undefined) return;
    const target = candidates.find(item => item.candidate_id === artifact.candidate_id);
    if (!target || artifact.start < 0 || artifact.end > total || artifact.start >= artifact.end) return;
    pendingDraft.current(); setMode('manual'); setCandidateId(target.candidate_id);
    trimHistory.current = []; setTrim([artifact.start, artifact.end]); seek(artifact.start);
  }

  const togglePlay = useCallback(async () => {
    const player = viewingOutput ? outputVideo.current : video.current;
    if (!player) return;
    try {
      if (!player.paused) { player.pause(); return; }
      if (!viewingOutput && player.currentTime >= sourceRange[1] - .05) seek(sourceRange[0]);
      await player.play();
    }
    catch { setError('浏览器无法播放该视频编码。可尝试 MP4（H.264）素材。'); }
  }, [viewingOutput, seek, sourceRange[0], sourceRange[1]]);

  const changeTrim = useCallback((next: [number, number], save = true) => {
    if (busy) return;
    if (save) trimHistory.current.push(trim);
    setTrim([Math.max(0, next[0]), Math.min(total, next[1])]);
  }, [trim, total, busy]);

  const undo = useCallback(() => {
    if (busy) return;
    const previous = trimHistory.current.pop(); if (previous) setTrim(previous); else setNotice('没有可撤销的裁剪操作');
  }, [busy]);

  useEffect(() => {
    const keydown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.closest('input, textarea, select, dialog, [contenteditable], [role="slider"]') || document.querySelector('dialog[open]')) return;
      if (e.code === 'Space' && !target.closest('button, a, summary') && (viewingOutput ? output : media)) { e.preventDefault(); void togglePlay(); }
      if (e.code === 'ArrowLeft' || e.code === 'ArrowRight') {
        if (!(viewingOutput ? output : media)) return;
        e.preventDefault();
        const step = e.code === 'ArrowLeft' ? -1 : 1;
        if (viewingOutput) seekOutput(outputPosition + step); else seek(position + step);
      }
      if (mode === 'manual' && !busy) {
        if (e.code === 'KeyI' && position < trim[1] - .05) changeTrim([position, trim[1]]);
        if (e.code === 'KeyO' && position > trim[0] + .05) changeTrim([trim[0], position]);
        if ((e.ctrlKey || e.metaKey) && e.code === 'KeyZ') { e.preventDefault(); undo(); }
      }
    };
    window.addEventListener('keydown', keydown); return () => window.removeEventListener('keydown', keydown);
  }, [togglePlay, position, outputPosition, seek, seekOutput, trim, changeTrim, undo, mode, viewingOutput, output, media, busy]);

  function chooseMedia(item: Media) {
    if (busy) return;
    pendingDraft.current();
    setJob(null); setActiveId(item.id); setOutput(null); setDelivery(null); setArtifactId(''); setPosition(0); setCandidateId(''); setTrim([0, item.duration]); trimHistory.current = [];
    const existing = data?.runs.find(r => r.media_id === item.id);
    if (existing) selectRun(existing); else { setRunId(''); setQuery(''); setVisualEn(''); if (item.category) { setRoute('audio_visual'); setEvents([item.category]); } }
  }

  function chooseCandidate(c: Candidate) {
    pendingDraft.current();
    const range = restoreRange(activeId, runId, c);
    setMode('manual'); setCandidateId(c.candidate_id); trimHistory.current = []; setTrim(range); seek(range[0]);
  }

  function enterFineTune() {
    if (busy || mode === 'manual') return;
    setMode('manual');
    seek(trim[0]);
  }

  async function startAutoEdit() {
    if (!autoConsent || busy) return;
    setSubmitting(true); setError('');
    try {
      const { job: submitted } = await post('/api/auto-edit', { media_id: activeId, query: query.trim(), allow_cloud: autoConsent });
      setOutput(null); setDelivery(null); setArtifactId(''); setRunId(''); setCandidateId('');
      setJob(submitted);
      autoDialog.current?.close(); setAutoConsent(false);
    } catch (e) { setError((e as Error).message); }
    finally { setSubmitting(false); }
  }

  async function startRetrieval(e: React.FormEvent) {
    e.preventDefault(); setError('');
    if (busy) return;
    pendingDraft.current();
    if (mode === 'agent') { setAutoConsent(false); autoDialog.current?.showModal(); return; }
    if (!cloudPlanning && route !== 'audio_visual' && !visualEn.trim()) { setError('本地视觉模型需要英文查询，请在「检索设置」中填写，或启用智能解析。'); return; }
    setSubmitting(true);
    try {
      const { job: submitted } = await post('/api/retrieve', { media_id: activeId, query, route, visual_query_en: visualEn.trim(), events, cloud_planning: cloudPlanning, allow_cloud: cloudPlanning });
      setOutput(null); setDelivery(null); setArtifactId(''); setRunId(''); setCandidateId('');
      setJob(submitted);
    } catch (e) { setError((e as Error).message); }
    finally { setSubmitting(false); }
  }

  async function startExport(review = false) {
    if (busy || !candidate || (review && !cloudReview)) return;
    pendingDraft.current();
    setSubmitting(true);
    setError('');
    try {
      const { job: submitted } = await post(review ? '/api/review' : '/api/export', { run_id: runId, candidate_id: candidateId, start: trim[0], end: trim[1], allow_cloud: cloudReview });
      setJob(submitted);
      exportDialog.current?.close();
    } catch (e) { setError((e as Error).message); }
    finally { setSubmitting(false); }
  }

  async function stopJob() {
    if (!job || !jobActive || job.status === 'cancelling' || stopping) return;
    setStopping(true); setError('');
    try { setJob(await post(`/api/jobs/${job.id}/cancel`, {})); }
    catch (e) { setError((e as Error).message); }
    finally { setStopping(false); }
  }

  async function retryJob() {
    if (!job?.can_retry || busy || (job.requires_cloud && !retryConsent)) return;
    pendingDraft.current(); setSubmitting(true); setError('');
    try {
      const { job: submitted } = await post(`/api/jobs/${job.id}/retry`, { allow_cloud: retryConsent });
      setJob(submitted); retryDialog.current?.close(); setRetryConsent(false);
    } catch (e) { setError((e as Error).message); }
    finally { setSubmitting(false); }
  }

  async function importMedia(file?: File) {
    pendingDraft.current();
    setImporting(true); setError('');
    try {
      let result;
      if (file) {
        const response = await fetch('/api/import/upload', { method: 'POST', headers: { 'X-Studio-Token': data?.token || '', 'X-Filename-Encoded': encodeURIComponent(file.name) }, body: file });
        result = await response.json(); if (!response.ok) throw new Error(result.error || '导入失败');
      } else result = await post('/api/import/path', { path: importPath });
      const item: Media = result.media;
      setData(old => old ? { ...old, media: [...old.media.filter(m => m.id !== item.id), item] } : old);
      setJob(null); setActiveId(item.id); setRunId(''); setCandidateId(''); setOutput(null); setDelivery(null); setArtifactId(''); setQuery(''); setVisualEn(''); setTrim([0, item.duration]); setPosition(0);
      importDialog.current?.close(); setNotice('素材已加入本地工作台');
    } catch (e) { setError((e as Error).message); }
    finally { setImporting(false); }
  }

  function dragHandle(e: React.PointerEvent<HTMLButtonElement>, side: 0 | 1) {
    if (busy) return;
    e.stopPropagation(); e.currentTarget.setPointerCapture(e.pointerId); trimHistory.current.push(trim);
    const element = e.currentTarget;
    const move = (event: PointerEvent) => {
      const box = timeline.current?.getBoundingClientRect(); if (!box) return;
      const value = Math.max(0, Math.min(total, (event.clientX - box.left) / box.width * total));
      setTrim(old => side === 0 ? [Math.min(value, old[1] - .05), old[1]] : [old[0], Math.max(value, old[0] + .05)]);
    };
    const end = () => { element.removeEventListener('pointermove', move); element.removeEventListener('pointerup', end); element.removeEventListener('pointercancel', end); };
    element.addEventListener('pointermove', move); element.addEventListener('pointerup', end); element.addEventListener('pointercancel', end);
  }

  const modal = (ref: React.RefObject<HTMLDialogElement | null>, title: string, content: ReactNode, onClose?: () => void) => <dialog ref={ref} onClose={onClose} onClick={e => { if (e.target === e.currentTarget) e.currentTarget.close(); }}><div className="dialog-head"><h2>{title}</h2><IconButton label="关闭窗口" onClick={() => ref.current?.close()}><X size={20}/></IconButton></div>{content}</dialog>;

  function toggleLibrary(next: 'media' | 'history') {
    const open = !libraryOpen || section !== next;
    setLibraryOpen(open);
    if (section !== next) { setSection(next); setLibraryQuery(''); }
    if (open) revealNarrowPanel('library-panel');
  }

  function revealNarrowPanel(id: string) {
    if (window.matchMedia('(max-width: 760px)').matches) {
      window.requestAnimationFrame(() => document.getElementById(id)?.scrollIntoView({ block: 'start' }));
    }
  }

  const transportPlaying = viewingOutput ? outputPlaying : playing;
  const transportPosition = viewingOutput ? outputPosition : position;
  const transportDuration = viewingOutput ? output?.duration || 0 : media?.duration ? total : 0;
  const playable = Boolean(viewingOutput ? output : media);

  return <div className={`studio mode-${mode} ${glass ? '' : 'glass-off'}`}>
    <Suspense fallback={<div className="flow-world" aria-hidden="true"><div className="flow-fallback"/></div>}><FlowBackground paused={backgroundPaused} palette={palette}/></Suspense>
    <a className="skip-link" href="#workspace">跳到剪辑工作台</a>
    <div className="app-body">
      <header className="topbar">
        <div className="workspace-navigation">
          <a className="wordmark" href="#workspace">cutliner<span className="wordmark-dot"/></a>
          <nav className="library-switches" aria-label="素材与历史导航">
            <IconButton label="导入素材" disabled={busy || !data} onClick={() => importDialog.current?.showModal()}><Plus size={17}/></IconButton>
            <IconButton id="library-toggle-media" label="素材库" aria-controls="library-panel" aria-expanded={libraryOpen && section === 'media'} onClick={() => toggleLibrary('media')}><Layers3 size={17}/></IconButton>
            <IconButton id="library-toggle-history" label="检索历史" aria-controls="library-panel" aria-expanded={libraryOpen && section === 'history'} onClick={() => toggleLibrary('history')}><History size={17}/></IconButton>
          </nav>
        </div>
        <div className="edit-modes" role="group" aria-label="剪辑模式"><button className="lg" aria-pressed={mode === 'agent'} disabled={busy} onClick={() => { pendingDraft.current(); setMode('agent'); setShowSource(false); }}><Workflow size={15}/>Agent 自动剪辑</button><button className="lg" aria-pressed={mode === 'manual'} disabled={busy} onClick={enterFineTune}><SlidersHorizontal size={15}/>手动精调</button></div>
        <div className="top-actions">
          <span className={`local-status ${connection !== 'connected' ? 'disconnected' : ''}`} title={connection === 'connected' ? '本地已连接' : '本地服务未连接'} aria-label={connection === 'connected' ? '本地已连接' : '本地服务未连接'} role="img"><i/></span>
          <IconButton id="task-panel-toggle" label="剪辑任务面板" aria-controls="task-panel" aria-expanded={searchOpen} onClick={() => { setSearchOpen(!searchOpen); if (!searchOpen) revealNarrowPanel('task-panel'); }}><PanelRight size={17}/></IconButton>
          <IconButton label="工作台设置" onClick={() => settingsDialog.current?.showModal()}><Settings2 size={17}/></IconButton>
          {output && <a className="export-button lg" aria-label="下载成片" title="下载成片" href={output.url} download={output.filename}><span className="export-label">下载成片</span><ArrowDownToLine size={15}/></a>}
          {mode === 'manual' && <button className="export-button lg" aria-label="导出" title="导出" disabled={busy || !candidate} onClick={() => { setCloudReview(false); exportDialog.current?.showModal(); }}><span className="export-label">导出</span><ArrowUpRight size={16}/></button>}
        </div>
      </header>

      <div className="task-control-bar" aria-label="任务控制">
        <div className="task-status" role="status">
          {connection !== 'connected' ? <WifiOff size={13}/> : jobActive ? <LoaderCircle size={13} className="spin"/> : delivery?.reviewed ? <ShieldCheck size={13}/> : <Workflow size={13}/>}
          <span title={job?.stage}>{connection === 'reconnecting' ? '正在重新连接' : connection === 'connecting' ? '正在连接' : jobActive || job?.status && job.status !== 'complete' ? job.stage : delivery?.reviewed ? '成片已复核' : currentRun ? historyLabel(currentRun) : '就绪'}</span>
        </div>
        <div className="task-control-actions">
          {connection === 'reconnecting' && <button className="lg" onClick={() => void reconnect()}>重连</button>}
          {jobActive && <button className="lg job-stop" disabled={stopping || job?.status === 'cancelling' || connection !== 'connected'} title="停止后续工作；已发送的 API 请求可能仍计费" onClick={() => void stopJob()}><Square size={12}/>{stopping || job?.status === 'cancelling' ? '正在停止' : '停止任务'}</button>}
          {job?.can_retry && !busy && <button className="lg" disabled={connection !== 'connected'} onClick={() => { setRetryConsent(false); retryDialog.current?.showModal(); }}><RotateCcw size={13}/>重试任务</button>}
          <button className="lg records-trigger" aria-haspopup="dialog" aria-controls="agent-records" onClick={() => { recordsDialog.current?.showModal(); setRecordsOpen(true); }}><Workflow size={13}/>执行记录</button>
        </div>
      </div>

      <main id="workspace" className={`workspace ${libraryOpen ? 'library-open' : ''} ${searchOpen ? 'search-open' : ''}`}>
        <aside id="library-panel" className="library" aria-label="素材与历史" hidden={!libraryOpen}>
          <div className="library-heading"><h1>{section === 'history' ? '检索历史' : '素材库'}<span className="count">{section === 'history' ? libraryRuns.length : libraryMedia.length}</span></h1><IconButton label="收起素材与历史" onClick={() => { setLibraryOpen(false); document.getElementById(`library-toggle-${section}`)?.focus(); }}><X size={16}/></IconButton></div>
          <label className="library-filter"><Search size={14}/><input type="search" aria-label={section === 'history' ? '搜索历史' : '搜索素材'} placeholder={section === 'history' ? '搜索历史' : '搜索素材'} value={libraryQuery} onChange={e => setLibraryQuery(e.target.value)}/></label>
          {section === 'history' ? <div className="history-list">{libraryRuns.map(r => <button className={`history-item ${r.id === runId ? 'active' : ''}`} key={r.id} onClick={() => { if (!busy) { setJob(null); selectRun(r); } }} disabled={busy}><span>{r.created}<ArrowUpRight size={14}/></span><strong>{r.request || r.task.query}</strong><small>{historyLabel(r)}</small></button>)}{libraryRuns.length === 0 && <p className="empty-state">{filter ? '无匹配结果' : '暂无记录'}</p>}</div> : <>
            <div className="asset-list">{libraryMedia.map((item, i) => <div className="asset-row" key={item.id}><button className={`asset ${activeId === item.id ? 'active' : ''}`} onClick={() => chooseMedia(item)} disabled={busy}>
              <div className="asset-image"><img src={item.poster} alt={item.name} loading={i > 2 ? 'lazy' : 'eager'}/><span>{timecode(item.duration)}</span>{activeId === item.id && <i className="asset-check"><Check size={12}/></i>}</div>
              <div className="asset-info"><strong title={item.name}>{item.name}</strong></div><div className="asset-meta">{item.width} × {item.height}<span>{item.fps.toFixed(0)} fps</span></div>
            </button><IconButton label={item.favorite ? '取消收藏' : '收藏素材'} className="asset-favorite" aria-pressed={Boolean(item.favorite)} disabled={savingFavorite !== null} onClick={() => void toggleFavorite(item)}><Star size={14} fill={item.favorite ? 'currentColor' : 'none'}/></IconButton></div>)}</div>
            {filter && libraryMedia.length === 0 && <p className="empty-state">无匹配结果</p>}
            {!filter && libraryMedia.length === 0 && <button className="import-tile lg" disabled={busy || !data} onClick={() => importDialog.current?.showModal()}><Plus size={22}/><span>导入素材</span></button>}
          </>}
        </aside>

        <section className="viewer-zone" aria-label="视频预览">
          <div className="viewer-header">
            <div className="viewer-name">
              <div className="viewer-tabs" role="group" aria-label="预览来源">
                <button aria-pressed={!viewingOutput} onClick={() => setShowSource(true)}>原片</button>
                {mode === 'agent' && hasOutputView && <button aria-pressed={viewingOutput} onClick={() => setShowSource(false)}>成片{selectedArtifact ? ` v${selectedArtifact.version}` : ''}</button>}
              </div>
              <span className="viewer-file" title={viewingOutput ? output?.filename : media?.filename}>{viewingOutput ? output?.filename : media?.filename}</span>
              {viewingOutput && delivery?.reviewed && <span className="verified-stamp"><ShieldCheck size={12}/>已复核</span>}
            </div>
            <div className="viewer-mode">
              {media && !viewingOutput && <button className="lg" onClick={openRangePreview}>选段预览</button>}
              <button className="lg" onClick={() => setFit(!fit)} aria-label="填充画面" aria-pressed={!fit}>{fit ? '适应' : '填充'}</button>
            </div>
          </div>
          <div className="preview-stage" ref={stage}>
            {viewingOutput && output ?
              <video ref={outputVideo} key={`output-${output.id}`} src={output.url} muted={muted} playsInline preload="metadata" style={{ objectFit: fit ? 'contain' : 'cover' }}
                onLoadedMetadata={e => { e.currentTarget.playbackRate = speed; }} onTimeUpdate={e => setOutputPosition(e.currentTarget.currentTime)}
                onPlay={() => setOutputPlaying(true)} onPause={() => setOutputPlaying(false)} onEnded={() => setOutputPlaying(false)} onClick={() => void togglePlay()}
                onError={() => { setError('成片不可用，请重新选择版本。'); if (artifactId) void chooseArtifact(artifactId, true); }}/> :
              viewingOutput && selectedArtifact ? <div className="preview-empty"><Clapperboard size={38}/><p>{artifactStatus(selectedArtifact)}</p></div> :
              media && unsupportedMedia === media.id ? <img className="range-preview-poster" src={media.poster} alt={media.name} style={{ objectFit: fit ? 'contain' : 'cover' }}/> :
              media ? <video ref={video} key={`source-${media.id}`} src={media.url} poster={media.poster} muted={muted} playsInline preload="metadata" style={{ objectFit: fit ? 'contain' : 'cover' }}
                onLoadedMetadata={e => { const player = e.currentTarget; setDuration(player.duration || media.duration); player.currentTime = Math.min(position, media.duration); player.playbackRate = speed; }}
                onTimeUpdate={e => { const player = e.currentTarget; setPosition(player.currentTime); if (player.currentTime >= sourceRange[1] && !player.paused) { player.pause(); player.currentTime = sourceRange[0]; } }}
                onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)} onError={() => setUnsupportedMedia(media.id)} onClick={() => void togglePlay()}/> :
              <div className="preview-empty"><Clapperboard size={40}/><button onClick={() => importDialog.current?.showModal()} disabled={!data}>导入视频 <Plus size={16}/></button></div>}
          </div>
          <div className="transport" aria-label="播放控制">
            <div className="transport-buttons">
              <IconButton label={viewingOutput ? '跳到成片开头' : '跳到选段开头'} onClick={() => viewingOutput ? seekOutput(0) : seek(sourceRange[0])} disabled={!playable}><SkipBack size={16}/></IconButton>
              <IconButton label={transportPlaying ? '暂停' : '播放'} className="play-button" onClick={() => !viewingOutput && media?.id === unsupportedMedia ? openRangePreview() : void togglePlay()} disabled={!playable}>{transportPlaying ? <Pause size={17} fill="currentColor"/> : <Play size={17} fill="currentColor"/>}</IconButton>
              <IconButton label={viewingOutput ? '跳到成片结尾' : '跳到选段结尾'} onClick={() => viewingOutput ? seekOutput(transportDuration) : seek(sourceRange[1])} disabled={!playable}><SkipForward size={16}/></IconButton>
            </div>
            <input className="transport-progress" type="range" min={0} max={transportDuration || 1} step="0.01" value={Math.min(transportPosition, transportDuration)} disabled={!playable} aria-label={viewingOutput ? '成片播放进度' : '原片播放进度'} aria-valuetext={timecode(transportPosition, true)} onChange={e => viewingOutput ? seekOutput(e.currentTarget.valueAsNumber) : seek(e.currentTarget.valueAsNumber)}/>
            <span className="transport-time">{timecode(transportPosition, true)}<small> / {timecode(transportDuration, true)}</small></span>
            <div className="transport-buttons">
              <IconButton label={muted ? '打开声音' : '静音'} aria-pressed={!muted} disabled={!playable} onClick={() => setMuted(!muted)}>{muted ? <VolumeX size={16}/> : <Volume2 size={16}/>}</IconButton>
              <IconButton label="全屏预览" disabled={!playable} onClick={() => { void stage.current?.requestFullscreen().catch(() => setNotice('当前窗口不支持全屏')); }}><Expand size={16}/></IconButton>
            </div>
          </div>
        </section>

        <aside id="task-panel" className="search-panel" aria-label="检索与候选片段" hidden={!searchOpen}>
          <form className="search-composer" onSubmit={startRetrieval}>
            <div className="task-composer-heading"><Workflow size={16}/><h2>{mode === 'agent' ? '剪辑任务' : '检索任务'}</h2><IconButton label="收起剪辑任务" onClick={() => { setSearchOpen(false); document.getElementById('task-panel-toggle')?.focus(); }}><X size={16}/></IconButton></div>
            <label className="sr-only" htmlFor="query">描述需要剪辑的视频片段</label><textarea id="query" value={query} onChange={e => setQuery(e.target.value)} placeholder="想剪出什么片段？" maxLength={2000} required rows={4} disabled={busy}/>
            {mode === 'manual' && <><div className="route-options" aria-label="本地检索路径">{Object.entries(routeNames).map(([value,label]) => <button className="lg" type="button" key={value} aria-pressed={route === value} onClick={() => setRoute(value)} disabled={busy || cloudPlanning}>{value === 'visual' ? <ScanLine size={13}/> : value === 'audio_visual' ? <AudioLines size={13}/> : <Layers3 size={13}/>} {label.replace('定位','')}</button>)}</div>
            <details className="model-inputs"><summary><SlidersHorizontal size={12}/> 检索设置 <ChevronDown size={13}/></summary><div>
              <label className="check-line"><input type="checkbox" checked={cloudPlanning} onChange={e => setCloudPlanning(e.target.checked)} disabled={busy}/>智能解析需求（云端服务，可能计费）</label>
              {!cloudPlanning && <>{route !== 'audio_visual' && <label className="field-label">英文视觉查询<input value={visualEn} onChange={e => setVisualEn(e.target.value)} placeholder="A racing car driving on a track"/></label>}{route !== 'visual' && <label className="field-label">声音事件<select value={events[0] || 'Truck'} onChange={e => setEvents([e.target.value])}>{data?.events.map(event => <option key={event.value} value={event.value}>{event.label}</option>)}</select></label>}</>}
              {route !== 'visual' && <p>仅支持所列声音事件，不支持具体台词。</p>}
            </div></details></>}
            <button className="find-button lg" disabled={!data || !media || busy || !query.trim()} type="submit"><span>{busy ? <><LoaderCircle size={16} className="spin"/> 正在执行</> : mode === 'agent' ? <><Workflow size={17}/>开始自动剪辑</> : '检索片段'}</span>{!busy && <ArrowUpRight size={21}/>}</button>
          </form>
          <div className="candidate-section"><div className="candidate-heading"><h2>候选片段 <span>{String(visibleCandidates.length).padStart(2,'0')}</span></h2><IconButton label={onlyReviewed ? '显示全部候选' : '仅显示已审核'} aria-pressed={onlyReviewed} onClick={() => setOnlyReviewed(!onlyReviewed)}><SlidersHorizontal size={16}/></IconButton></div>
            <div className="candidates">{visibleCandidates.map((c,i) => <button key={c.candidate_id} className={`candidate ${candidateId === c.candidate_id ? 'active' : ''}`} onClick={() => chooseCandidate(c)} disabled={busy}><div className="candidate-image"><img src={`/api/thumb/${activeId}?t=${(c.start_seconds + c.end_seconds) / 2}`} alt={`候选片段 ${i+1} 的中间画面`}/><span>{String(i+1).padStart(2,'0')}</span></div><div className="candidate-body"><strong>{timecode(c.start_seconds, true)}<span>—</span>{timecode(c.end_seconds, true)}</strong><p>{c.model_evidence.map(e => data?.events.find(v => v.value === e.label)?.label || (e.modality === 'visual' ? '视觉匹配' : e.label)).filter((v,j,a) => a.indexOf(v) === j).join(' · ')}</p><div><span className={c.review_verdict === 'pass' ? 'verdict pass' : 'verdict'}>{c.review_verdict === 'pass' ? '已审核' : c.review_verdict === 'reject' ? '未通过' : '待审核'}</span><span className="model-score">{(c.end_seconds - c.start_seconds).toFixed(1)} 秒</span></div></div></button>)}
              {!visibleCandidates.length && <div className="candidate-empty"><ScanLine size={22}/><strong>{onlyReviewed ? '暂无已审核片段' : '暂无候选'}</strong></div>}
            </div>
          </div>
        </aside>

        <section className="timeline-panel" aria-label="剪辑时间线">
          <div className="timeline-toolbar">
            <div className="timeline-title"><Film size={15}/><h2>原片时间线</h2></div>
            <div className="timeline-tools">
              {mode === 'manual' && <>
                <IconButton label="撤销裁剪（Ctrl+Z）" disabled={busy || !media} onClick={undo}><History size={16}/></IconButton>
                <IconButton label="标记当前入点（I）" disabled={busy || !media || position >= trim[1] - .05} onClick={() => changeTrim([position, trim[1]])}><span className="keymark">[</span></IconButton>
                <IconButton label="标记当前出点（O）" disabled={busy || !media || position <= trim[0] + .05} onClick={() => changeTrim([trim[0], position])}><span className="keymark">]</span></IconButton>
              </>}
              <button className="timeline-speed lg" aria-label={`播放速度 ${speed} 倍，切换速度`} onClick={() => setSpeed(speed === 1 ? 1.5 : speed === 1.5 ? 2 : speed === .5 ? 1 : .5)}>{speed}×</button>
              {currentRun && <ArtifactVersions run={currentRun} selectedId={artifactId} candidate={candidate} trim={trim} disabled={busy || loadingArtifact} onSelect={id => void chooseArtifact(id)} onApply={applyArtifactRange}/>}
            </div>
          </div>
          <div className="timeline-body"><div className="track-labels"><span className="ruler-icon"><MousePointer2 size={13}/></span><span><Film size={14}/>V1</span>{mode === 'manual' && <span><AudioLines size={14}/>A1</span>}</div><div className="track-area" ref={timeline}>
            <div className="time-ruler">{Array.from({ length: 9 }, (_, i) => <button key={i} disabled={!media} style={{ left: `${i * 12.5}%` }} onClick={() => seekSource(total * i / 8)} aria-label={`跳到 ${timecode(total * i / 8, true)}`}>{media ? timecode(total * i / 8) : '—'}<i/></button>)}</div>
            <div className="video-track" onClick={e => { const b = e.currentTarget.getBoundingClientRect(); seekSource((e.clientX - b.left) / b.width * total); }}>
              {mode === 'manual' && media ? <div className="filmstrip">{Array.from({length:12}, (_, i) => <img key={`${media.id}-${i}`} src={`/api/thumb/${media.id}?t=${((i + .5) / 12 * total).toFixed(2)}`} alt="" loading="lazy"/>)}</div> : media && <span className="track-source">{media.name}</span>}
              {media && (candidate || mode === 'manual') && <>
                <div className="track-dim left" style={{width:`${trim[0]/total*100}%`}}/><div className="track-dim right" style={{width:`${(total-trim[1])/total*100}%`}}/>
                <div className="trim-selection" style={{left:`${trim[0]/total*100}%`,width:`${(trim[1]-trim[0])/total*100}%`}}>
                  <span className="selection-title">{candidate ? `选段 ${String(candidates.indexOf(candidate)+1).padStart(2,'0')}` : '源素材'} <span>{(trim[1]-trim[0]).toFixed(2)}s</span></span>
                  {mode === 'manual' && ([0,1] as const).map(side => <button key={side} disabled={busy} className={`trim-handle ${side ? 'end' : 'start'}`} aria-label={side ? '调整出点' : '调整入点'} role="slider" aria-valuemin={0} aria-valuemax={total} aria-valuenow={trim[side]} aria-valuetext={timecode(trim[side],true)} onPointerDown={e => dragHandle(e,side)} onClick={e => e.stopPropagation()} onKeyDown={e => { if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.stopPropagation(); e.preventDefault(); const d = e.key === 'ArrowLeft' ? -.1 : .1; const n: [number,number] = [...trim]; n[side] += d; if (n[0] >= 0 && n[0] < n[1]-.02 && n[1] <= total) changeTrim(n); } }}><i/></button>)}
                </div>
              </>}
            </div>
            {mode === 'manual' && <div className="audio-track"><svg viewBox="0 0 1100 48" preserveAspectRatio="none" aria-label="音轨波形">{wave.map((v,i) => <line key={i} x1={i*5+2} x2={i*5+2} y1={24-v*21} y2={24+v*21}/>)}</svg>{media && waveStatus !== 'ready' && <span className="wave-status">{waveStatus === 'loading' ? '正在读取音轨…' : waveStatus === 'silent' ? '素材没有可解码音轨' : '音轨预览暂不可用'}</span>}{waveStatus === 'ready' && media && waveDuration < total - .1 && <span className="wave-status">波形覆盖前 {timecode(waveDuration)}</span>}</div>}
            {media && <div className="playhead" style={{left:`${position/total*100}%`}}><i/></div>}
          </div></div>
          {mode === 'manual' && media && <div className="timeline-editbar">
            <div className="trim-controls">
              <label>入点 <input aria-label="片段开始时间（秒）" disabled={busy} type="number" min="0" max={trim[1] - .05} step="0.1" value={Number(trim[0].toFixed(2))} onChange={e => { const n = Number(e.target.value); if (n >= 0 && n < trim[1]) changeTrim([n, trim[1]]); }}/><span>s</span></label>
              <label>出点 <input aria-label="片段结束时间（秒）" disabled={busy} type="number" min={trim[0] + .05} max={total} step="0.1" value={Number(trim[1].toFixed(2))} onChange={e => { const n = Number(e.target.value); if (n > trim[0] && n <= total) changeTrim([trim[0], n]); }}/><span>s</span></label>
              <span className="trim-duration">{(trim[1] - trim[0]).toFixed(2)} 秒</span>
            </div>
            {candidate && <div className="trim-actions">
              <span>{trimReviewed ? '当前范围已复核' : trimChanged ? '已修改 · 待复核' : '当前范围待复核'}</span>
              <button className="lg" disabled={busy} onClick={() => changeTrim(modelRange(candidate))}>模型建议</button>
              {candidate.boundary_refinement && <button className="lg" disabled={busy} onClick={() => { const refined = candidate.boundary_refinement; if (refined) changeTrim([refined.coarse_start_seconds, refined.coarse_end_seconds]); }}>恢复粗定位</button>}
            </div>}
          </div>}
        </section>
      </main>
    </div>

    {error && <div className="error-toast" role="alert"><div><strong>需要注意</strong><p>{error}</p></div><IconButton label="关闭错误提示" onClick={() => setError('')}><X size={18}/></IconButton></div>}
    {notice && <div className="notice-toast" role="status"><Check size={17}/><span>{notice}</span><IconButton label="关闭通知" onClick={() => setNotice('')}><X size={15}/></IconButton></div>}

    {modal(importDialog,'导入素材',<><input ref={fileInput} type="file" accept="video/*,.mkv" hidden onChange={e => { const file = e.target.files?.[0]; if(file) void importMedia(file); e.target.value=''; }}/><button className="file-drop" onClick={() => fileInput.current?.click()} disabled={importing} onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); if(e.dataTransfer.files[0]) void importMedia(e.dataTransfer.files[0]); }}><Plus size={28}/><strong>{importing ? '正在导入…' : '选择或拖入视频'}</strong><span>最大 4 GB</span></button><div className="or-divider">或通过本地路径导入</div><form onSubmit={e => { e.preventDefault(); void importMedia(); }}><label className="field-label">文件路径<input value={importPath} onChange={e => setImportPath(e.target.value)} placeholder="D:\Videos\my-video.mp4" required/></label><button className="primary lg" disabled={!importPath.trim() || importing} type="submit">导入 <ArrowRight size={17}/></button></form></>)}
    {modal(exportDialog,'导出当前范围',<><div className="export-summary"><img src={`/api/thumb/${activeId}?t=${(trim[0]+trim[1])/2}`} alt="所选片段预览"/><div><small>{trimReviewed ? '当前范围已复核' : '当前范围待复核'}</small><strong>{timecode(trim[0],true)} — {timecode(trim[1],true)}</strong><span>{(trim[1]-trim[0]).toFixed(2)} 秒 · MP4 / H.264 · 保留音轨</span></div></div><button className="primary lg" disabled={busy || !candidate} onClick={() => void startExport()}><ArrowDownToLine size={17}/>直接导出</button><div className="review-option"><ShieldCheck size={21}/><div><strong>Agent 复核当前范围</strong><p>当前范围的画面将发送至云端模型，可能产生 API 费用。</p><label className="check-line"><input type="checkbox" checked={cloudReview} disabled={busy} onChange={e => setCloudReview(e.target.checked)}/>授权本次云端复核</label><button className="secondary lg" disabled={!cloudReview || busy || !candidate} onClick={() => void startExport(true)}>复核当前范围并导出 <ArrowUpRight size={15}/></button></div></div></>)}
    {modal(previewDialog,previewRange ? `${timecode(previewRange.start, true)} — ${timecode(previewRange.end, true)}` : '选段预览',previewRange && <video key={previewRange.url} className="range-preview-video" src={previewRange.url} playsInline autoPlay muted controls onError={() => setError('选段预览不可用')}/>,() => setPreviewRange(null))}
    {modal(settingsDialog,'设置',<>
      <div className="setting-row"><strong>液态玻璃</strong><button className="switch" role="switch" aria-checked={glass} aria-label="完整液态玻璃" onClick={() => setGlass(!glass)}><span/></button></div>
      <div className="setting-row"><strong>背景配色</strong><button className="background-palette lg" aria-label={`切换背景配色，当前为${flowPalettes[palette].name}`} onClick={() => setPalette((palette + 1) % flowPalettes.length)}><span className={`palette-swatch palette-${palette}`}/><span>{flowPalettes[palette].name}</span></button></div>
      <div className="setting-row"><strong>动态背景</strong><button className="switch" role="switch" aria-checked={!backgroundPaused} aria-label="动态背景" onClick={() => setBackgroundPaused(!backgroundPaused)}><span/></button></div>
      <div className="setting-row"><strong>快捷键</strong><IconButton label="查看快捷键" onClick={() => { settingsDialog.current?.close(); helpDialog.current?.showModal(); }}><CircleHelp size={18}/></IconButton></div>
    </>)}
    {modal(helpDialog,'快捷键',<div className="shortcut-list">{[['Space','播放 / 暂停'],['← / →','前后移动一秒'],['I','将当前帧设为入点'],['O','将当前帧设为出点'],['Ctrl / ⌘ + Z','撤销上次裁剪'],['Tab','在控件之间导航'],['Esc','关闭当前窗口']].map(([key,text]) => <div key={key}><span>{text}</span><kbd>{key}</kbd></div>)}</div>)}
    {modal(autoDialog,'开始自动剪辑',<><div className="auto-confirm"><Workflow size={28}/><div><strong>{media?.name}</strong><p>{query}</p></div></div><p className="dialog-description">需求文本与候选画面将发送至已配置的云端模型，可能产生 API 费用。原视频保留本机。本次交付一段成片。</p><label className="check-line auto-authorization"><input type="checkbox" checked={autoConsent} onChange={e => setAutoConsent(e.target.checked)}/>授权本次云端解析与画面复核</label><button className="primary lg" disabled={!autoConsent || busy} onClick={() => void startAutoEdit()}>{submitting ? <LoaderCircle size={17} className="spin"/> : <Workflow size={17}/>}确认开始 <ArrowRight size={17}/></button></>)}
    {modal(retryDialog,'重试原任务',<><div className="auto-confirm"><RotateCcw size={25}/><div><strong>{job?.scope?.media_name}</strong><p>{job?.scope?.query || job?.request}</p>{job?.scope?.ranges?.map(range => <p key={range.candidate_id}>{timecode(range.start, true)} — {timecode(range.end, true)}</p>)}</div></div><p className="dialog-description">按原任务与原范围重新执行，保留已有记录。</p>{job?.requires_cloud && <label className="check-line auto-authorization"><input type="checkbox" checked={retryConsent} onChange={e => setRetryConsent(e.target.checked)}/>重新授权本次云端调用，可能产生 API 费用</label>}<button className="primary lg" disabled={busy || !job?.can_retry || job.requires_cloud && !retryConsent} onClick={() => void retryJob()}><RotateCcw size={16}/>确认重试</button></>)}
    <dialog id="agent-records" className="records-dialog" ref={recordsDialog} aria-labelledby="records-title" onClose={() => setRecordsOpen(false)} onClick={e => { if (e.target === e.currentTarget) e.currentTarget.close(); }}>
      <div className="dialog-head"><h2 id="records-title">Agent 执行记录</h2><IconButton label="关闭执行记录" onClick={() => recordsDialog.current?.close()}><X size={19}/></IconButton></div>
      {recordsOpen && <AgentConsole job={contextJob} run={currentRun} delivery={delivery} onFineTune={() => { recordsDialog.current?.close(); enterFineTune(); }}/>}
    </dialog>
  </div>;
}
