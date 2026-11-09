import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { flowPalettes } from './flowPalettes';
import { ArrowDownToLine, ArrowLeft, ArrowRight, ArrowUpRight, AudioLines, Check, ChevronDown, ChevronLeft, ChevronRight, CircleHelp, Clapperboard, Command, Expand, Film, FolderOpen, History, Layers3, LoaderCircle, Monitor, MoreHorizontal, MousePointer2, Pause, Play, Plus, ScanLine, Scissors, Settings2, ShieldCheck, SkipBack, SkipForward, SlidersHorizontal, Volume2, VolumeX, X } from 'lucide-react';

type Media = { id: string; name: string; filename: string; category?: string; path: string; duration: number; width: number; height: number; fps: number; size: number; url: string; poster: string };
type Evidence = { model_name: string; modality: string; label: string; score: number };
type Candidate = { candidate_id: string; start_seconds: number; end_seconds: number; relevance_score: number; evidence_source: string; model_evidence: Evidence[]; boundary_note: string; review_verdict: string };
type Run = { id: string; media_id: string; request: string; task: { query?: string; visual_query_en?: string; audio_event_classes?: string[] }; candidates: Candidate[]; route: string; created: string; review_status: string; path: string };
type Job = { id: string; kind: string; status: string; stage: string; error?: string; started: number; result?: { run?: Run; media?: Media; reviewed?: boolean; message?: string; candidate_count?: number } };
type Bootstrap = { token: string; media: Media[]; runs: Run[]; events: { value: string; label: string }[]; active_job: Job | null };
declare global { interface Window { LiquidGlass?: { refresh: () => void; rebuild: () => void } } }

const routeNames: Record<string, string> = { visual: '视觉', audio_visual: '声画', hybrid: '联合' };
const FlowBackground = lazy(() => import('./FlowBackground'));
function timecode(seconds: number, precise = false) {
  if (!Number.isFinite(seconds)) return '00:00';
  const mins = Math.floor(seconds / 60), secs = Math.floor(seconds % 60);
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}${precise ? '.' + String(Math.floor((seconds % 1) * 100)).padStart(2, '0') : ''}`;
}
const IconButton = ({ label, children, className = '', ...props }: { label: string; children: ReactNode; className?: string } & React.ButtonHTMLAttributes<HTMLButtonElement>) => <button type="button" className={`icon-button lg ${className}`} aria-label={label} title={label} {...props}>{children}</button>;
const MiniLabel = ({ children, number }: { children: ReactNode; number?: string }) => <div className="mini-label"><span>{children}</span>{number && <span>{number}</span>}</div>;

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
  const [position, setPosition] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [trim, setTrim] = useState<[number, number]>([0, 10]);
  const [wave, setWave] = useState<number[]>([]);
  const [waveStatus, setWaveStatus] = useState('loading');
  const [waveDuration, setWaveDuration] = useState(0);
  const [job, setJob] = useState<Job | null>(null);
  const [section, setSection] = useState('media');
  const [importPath, setImportPath] = useState('');
  const [importing, setImporting] = useState(false);
  const [output, setOutput] = useState<Media | null>(null);
  const [onlyReviewed, setOnlyReviewed] = useState(false);
  const [fit, setFit] = useState(true);
  const [glass, setGlass] = useState(true);
  const [backgroundPaused, setBackgroundPaused] = useState(false);
  const [palette, setPalette] = useState(0);
  const [duration, setDuration] = useState(0);
  const video = useRef<HTMLVideoElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const importDialog = useRef<HTMLDialogElement>(null);
  const exportDialog = useRef<HTMLDialogElement>(null);
  const settingsDialog = useRef<HTMLDialogElement>(null);
  const helpDialog = useRef<HTMLDialogElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const timeline = useRef<HTMLDivElement>(null);
  const trimHistory = useRef<[number, number][]>([]);
  const handledJob = useRef('');
  const media = data?.media.find(m => m.id === activeId);
  const currentRun = data?.runs.find(r => r.id === runId && r.media_id === activeId);
  const candidates = currentRun?.candidates ?? [];
  const candidate = candidates.find(c => c.candidate_id === candidateId);
  const visibleCandidates = onlyReviewed ? candidates.filter(c => c.review_verdict === 'pass') : candidates;
  const busy = job?.status === 'running' || importing;
  const sourceDuration = media?.duration || 10;
  const total = duration || sourceDuration;

  const post = useCallback(async (path: string, body: unknown) => {
    const response = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Studio-Token': data?.token || '' }, body: JSON.stringify(body) });
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || '请求失败');
    return value;
  }, [data?.token]);

  const selectRun = useCallback((run: Run) => {
    setActiveId(run.media_id); setRunId(run.id); setQuery(run.request || run.task.query || '');
    setVisualEn(run.task.visual_query_en || ''); setRoute(run.route);
    setEvents(run.task.audio_event_classes?.length ? run.task.audio_event_classes : ['Truck']);
    setCandidateId(run.candidates[0]?.candidate_id || '');
    const c = run.candidates[0];
    setTrim(c ? [c.start_seconds, c.end_seconds] : [0, 10]);
    setOutput(null); setPosition(c?.start_seconds || 0); trimHistory.current = [];
  }, []);

  useEffect(() => {
    fetch('/api/bootstrap').then(r => { if (!r.ok) throw new Error('本地服务暂未连接'); return r.json(); }).then((b: Bootstrap) => {
      setData(b); if (b.runs[0]) selectRun(b.runs[0]); else if (b.media[0]) setActiveId(b.media[0].id);
      if (b.active_job) setJob(b.active_job);
    }).catch(e => setError(e.message));
  }, [selectRun]);

  useEffect(() => {
    const refresh = () => window.LiquidGlass?.refresh();
    if (!document.querySelector('script[data-glass]')) {
      const script = document.createElement('script'); script.src = '/liquid-glass.js'; script.dataset.glass = 'true'; script.onload = refresh; document.body.appendChild(script);
    }
    const timer = window.setTimeout(refresh, 80); return () => clearTimeout(timer);
  }, [data, activeId, glass, playing, output]);

  useEffect(() => {
    if (!media) return;
    setDuration(0); setWave([]); setWaveStatus('loading');
    const controller = new AbortController();
    fetch(`/api/waveform/${media.id}`, { signal: controller.signal }).then(r => r.json()).then(w => {
      if (w.error) throw new Error(w.error);
      setWave(w.values); setWaveDuration(w.duration); setWaveStatus(w.has_audio ? 'ready' : 'silent');
    }).catch(e => { if (e.name !== 'AbortError') setWaveStatus('error'); });
    return () => controller.abort();
  }, [media?.id]);

  useEffect(() => {
    if (video.current) video.current.playbackRate = speed;
  }, [speed]);

  useEffect(() => {
    if (!notice) return; const timer = setTimeout(() => setNotice(''), 6000); return () => clearTimeout(timer);
  }, [notice]);

  useEffect(() => {
    if (!job || job.status !== 'running') return;
    let active = true;
    const poll = async () => {
      try {
        const response = await fetch(`/api/jobs/${job.id}`); if (!response.ok) return;
        const result: Job = await response.json(); if (active) setJob(result);
      } catch { /* Retain running state until the loopback service reconnects. */ }
    };
    const timer = setInterval(poll, 1500); return () => { active = false; clearInterval(timer); };
  }, [job?.id, job?.status]);

  useEffect(() => {
    if (!job || job.status === 'running' || handledJob.current === job.id) return;
    handledJob.current = job.id;
    if (job.status === 'failed') { setError(job.error || '任务未完成'); return; }
    if (job.result?.run) {
      const next = job.result.run;
      setData(old => old ? { ...old, runs: [next, ...old.runs.filter(r => r.id !== next.id)] } : old);
      selectRun(next);
    }
    if (job.result?.media) {
      const out = job.result.media;
      setData(old => old ? { ...old, media: [...old.media.filter(m => m.id !== out.id), out] } : old);
      setOutput(out);
    }
    setNotice(job.result?.message || '任务已完成');
    if (job.kind === 'review' && job.result?.reviewed) {
      setData(old => old ? { ...old, runs: old.runs.map(r => r.id === runId ? { ...r, candidates: r.candidates.map(c => c.candidate_id === candidateId ? { ...c, review_verdict: 'pass' } : c) } : r) } : old);
    }
  }, [job, selectRun, runId, candidateId]);

  const seek = useCallback((at: number) => {
    const value = Math.max(0, Math.min(total, at));
    if (video.current) video.current.currentTime = value; setPosition(value);
  }, [total]);

  const togglePlay = useCallback(async () => {
    if (!video.current || !media) return;
    try { if (video.current.paused) { if (video.current.currentTime >= trim[1] - .05) seek(trim[0]); await video.current.play(); } else video.current.pause(); }
    catch { setError('浏览器无法播放该视频编码。可尝试 MP4（H.264）素材。'); }
  }, [media, seek, trim]);

  const changeTrim = useCallback((next: [number, number], save = true) => {
    if (save) trimHistory.current.push(trim);
    setTrim([Math.max(0, next[0]), Math.min(total, next[1])]);
  }, [trim, total]);

  const undo = useCallback(() => {
    const previous = trimHistory.current.pop(); if (previous) setTrim(previous); else setNotice('没有可撤销的裁剪操作');
  }, []);

  useEffect(() => {
    const keydown = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest('input, textarea, select, dialog') || document.querySelector('dialog[open]')) return;
      if (e.code === 'Space') { e.preventDefault(); void togglePlay(); }
      if (e.code === 'ArrowLeft') { e.preventDefault(); seek(position - 1); }
      if (e.code === 'ArrowRight') { e.preventDefault(); seek(position + 1); }
      if (e.code === 'KeyI' && position < trim[1] - .05) changeTrim([position, trim[1]]);
      if (e.code === 'KeyO' && position > trim[0] + .05) changeTrim([trim[0], position]);
      if ((e.ctrlKey || e.metaKey) && e.code === 'KeyZ') { e.preventDefault(); undo(); }
    };
    window.addEventListener('keydown', keydown); return () => window.removeEventListener('keydown', keydown);
  }, [togglePlay, position, seek, trim, changeTrim, undo]);

  function chooseMedia(item: Media) {
    if (busy) return;
    setActiveId(item.id); setOutput(null); setPosition(0); setCandidateId(''); setTrim([0, item.duration]); trimHistory.current = [];
    const existing = data?.runs.find(r => r.media_id === item.id);
    if (existing) selectRun(existing); else { setRunId(''); setQuery(''); setVisualEn(''); if (item.category) { setRoute('audio_visual'); setEvents([item.category]); } }
  }

  function chooseCandidate(c: Candidate) {
    setCandidateId(c.candidate_id); changeTrim([c.start_seconds, c.end_seconds]); seek(c.start_seconds); setOutput(null);
  }

  async function startRetrieval(e: React.FormEvent) {
    e.preventDefault(); setError(''); setOutput(null);
    if (!cloudPlanning && route !== 'audio_visual' && !visualEn.trim()) { setError('本地视觉模型需要英文查询，请在「检索设置」中填写，或启用智能解析。'); return; }
    try {
      const { job_id } = await post('/api/retrieve', { media_id: activeId, query, route, visual_query_en: visualEn.trim(), events, cloud_planning: cloudPlanning });
      setJob({ id: job_id, kind: 'retrieval', status: 'running', stage: '正在准备检索', started: Date.now() / 1000 });
    } catch (e) { setError((e as Error).message); }
  }

  async function startExport(review = false) {
    setError('');
    try {
      const { job_id } = await post(review ? '/api/review' : '/api/export', { run_id: runId, candidate_id: candidateId, start: trim[0], end: trim[1], allow_cloud: cloudReview });
      setJob({ id: job_id, kind: review ? 'review' : 'export', status: 'running', stage: review ? 'Agent 正在读取候选证据' : '准备导出', started: Date.now() / 1000 });
      exportDialog.current?.close();
    } catch (e) { setError((e as Error).message); }
  }

  async function importMedia(file?: File) {
    setImporting(true); setError('');
    try {
      let result;
      if (file) {
        const response = await fetch('/api/import/upload', { method: 'POST', headers: { 'X-Studio-Token': data?.token || '', 'X-Filename': file.name.replace(/[^\x20-\x7e]/g, '_') }, body: file });
        result = await response.json(); if (!response.ok) throw new Error(result.error || '导入失败');
      } else result = await post('/api/import/path', { path: importPath });
      const item: Media = result.media;
      setData(old => old ? { ...old, media: [...old.media.filter(m => m.id !== item.id), item] } : old);
      setActiveId(item.id); setRunId(''); setCandidateId(''); setOutput(null); setQuery(''); setVisualEn(''); setTrim([0, item.duration]); setPosition(0);
      importDialog.current?.close(); setNotice('素材已加入本地工作台');
    } catch (e) { setError((e as Error).message); }
    finally { setImporting(false); }
  }

  function dragHandle(e: React.PointerEvent<HTMLButtonElement>, side: 0 | 1) {
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

  const modal = (ref: React.RefObject<HTMLDialogElement | null>, title: string, content: ReactNode) => <dialog ref={ref} onClick={e => { if (e.target === e.currentTarget) e.currentTarget.close(); }}><div className="dialog-head"><h2>{title}</h2><IconButton label="关闭窗口" onClick={() => ref.current?.close()}><X size={20}/></IconButton></div>{content}</dialog>;

  return <div className={`studio ${glass ? '' : 'glass-off'}`}>
    <Suspense fallback={<div className="flow-world" aria-hidden="true"><div className="flow-fallback"/></div>}><FlowBackground paused={backgroundPaused} palette={palette}/></Suspense>
    <a className="skip-link" href="#workspace">跳到剪辑工作台</a>
    <nav className="app-rail" aria-label="工作区导航">
      <a className="rail-mark lg" href="#workspace" aria-label="cutliner 工作台"><Scissors size={25}/></a>
      <div className="rail-middle">
        <IconButton label="素材与片段" className={section === 'media' ? 'selected' : ''} onClick={() => setSection('media')}><Layers3 size={21}/></IconButton>
        <IconButton label="检索历史" className={section === 'history' ? 'selected' : ''} onClick={() => setSection('history')}><History size={21}/></IconButton>
        <IconButton label="处理流程" className={section === 'model' ? 'selected' : ''} onClick={() => setSection('model')}><ScanLine size={21}/></IconButton>
      </div>
      <div className="rail-bottom"><IconButton label="快捷键说明" onClick={() => helpDialog.current?.showModal()}><CircleHelp size={20}/></IconButton><IconButton label="工作台设置" onClick={() => settingsDialog.current?.showModal()}><Settings2 size={20}/></IconButton><span className="user-initial">ME</span></div>
    </nav>
    <div className="app-body">
      <header className="topbar">
        <a className="wordmark" href="#workspace">cutliner<span className="wordmark-dot"/></a>
        <div className="project-crumb"><span>你的私人剪辑室</span><ChevronRight size={14}/><strong>{media ? '我的剪辑项目' : '新建项目'}</strong></div>
        <div className="top-actions"><button className="background-palette lg" title="切换背景配色" onClick={() => setPalette((palette + 1) % flowPalettes.length)}><span className={`palette-swatch palette-${palette}`}/><span>{flowPalettes[palette].name}</span></button><IconButton label={backgroundPaused ? '播放动态背景' : '暂停动态背景'} aria-pressed={!backgroundPaused} className="background-toggle" onClick={() => setBackgroundPaused(!backgroundPaused)}>{backgroundPaused ? <Play size={16}/> : <Pause size={16}/>}</IconButton><span className="local-status"><i/> 本地{data ? '已连接' : '连接中'}</span><button className="export-button lg" disabled={!candidate || busy} onClick={() => exportDialog.current?.showModal()}>导出片段 <ArrowUpRight size={17}/></button></div>
      </header>

      <main id="workspace" className="workspace">
        <aside className="library" aria-label="素材与历史">
          <div className="library-heading"><h1>{section === 'history' ? '检索历史' : section === 'model' ? '处理流程' : '素材库'}<span className="count">{section === 'history' ? data?.runs.length || 0 : data?.media.length || 0}</span></h1><IconButton label="导入素材" disabled={busy || !data} onClick={() => importDialog.current?.showModal()}><Plus size={20}/></IconButton></div>
          {section === 'model' ? <div className="model-explainer"><MiniLabel>剪辑流程</MiniLabel>{[['01','描述片段','告诉 cutliner 你想寻找的内容'],['02','定位内容','按画面或声音寻找候选片段'],['03','预览与精剪','选择片段，微调入点和出点'],['04','内容复核','按需核验片段是否满足要求'],['05','导出成片','保留所选画面与原始音轨']].map(([n,t,d]) => <div className="flow-row" key={n}><span>{n}</span><div><strong>{t}</strong><p>{d}</p></div></div>)}</div> : section === 'history' ? <div className="history-list">{data?.runs.map(r => <button className={`history-item ${r.id === runId ? 'active' : ''}`} key={r.id} onClick={() => { if (!busy) selectRun(r); }} disabled={busy}><span>{r.created}<ArrowUpRight size={14}/></span><strong>{r.request || r.task.query}</strong><small>{routeNames[r.route]} · {r.candidates.length} 个候选</small></button>)}{data?.runs.length === 0 && <p className="empty-state">完成第一次检索后，结果会保存在这里。</p>}</div> : <>
            <div className="library-collection"><FolderOpen size={15}/><span>项目素材</span><ChevronDown size={14}/></div>
            <div className="asset-list">{data?.media.map((item, i) => <button className={`asset ${activeId === item.id ? 'active' : ''}`} key={item.id} onClick={() => chooseMedia(item)} disabled={busy}>
              <div className="asset-image"><img src={item.poster} alt={item.name} loading={i > 2 ? 'lazy' : 'eager'}/><span>{timecode(item.duration)}</span>{activeId === item.id && <i className="asset-check"><Check size={12}/></i>}</div>
              <div className="asset-info"><span>{String(i + 1).padStart(2,'0')}</span><strong title={item.name}>{item.name}</strong><MoreHorizontal size={15}/></div><div className="asset-meta">{item.width} × {item.height}<span>{item.fps.toFixed(0)} fps</span></div>
            </button>)}</div>
            <button className="import-tile lg" disabled={busy || !data} onClick={() => importDialog.current?.showModal()}><Plus size={22}/><span>导入你的素材</span><small>MP4 · MOV · MKV · WebM</small></button>
          </>}
          <div className="library-foot"><span className="tiny-cross">＋</span><div>你的素材，留在本机。<small>云端复核始终由你决定</small></div><ArrowUpRight size={15}/></div>
        </aside>

        <section className="viewer-zone" aria-label="视频预览">
          <div className="viewer-header"><div><h2>画面预览</h2></div><div className="viewer-mode"><button className={`lg ${fit ? 'active' : ''}`} onClick={() => setFit(true)}>适应</button><button className={`lg ${!fit ? 'active' : ''}`} onClick={() => setFit(false)}>填充</button></div></div>
          <div className="preview-stage" ref={stage}>
            {media ? <video ref={video} key={media.id} src={media.url} poster={media.poster} muted={muted} playsInline preload="metadata" style={{ objectFit: fit ? 'contain' : 'cover' }} onLoadedMetadata={() => { setDuration(video.current?.duration || media.duration); if (video.current) video.current.currentTime = Math.min(position, media.duration); }} onTimeUpdate={() => { if (!video.current) return; const t = video.current.currentTime; setPosition(t); if (t >= trim[1] && !video.current.paused) { video.current.pause(); video.current.currentTime = trim[0]; } }} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)} onError={() => setError('当前编码可能无法在浏览器播放。检索仍支持源文件，预览建议使用 H.264 MP4。')} onClick={() => void togglePlay()}/> : <div className="preview-empty"><Clapperboard size={48}/><p>{error ? '等待本地素材服务' : '导入素材，开始一次剪辑'}</p><button onClick={() => importDialog.current?.showModal()} disabled={!data}>导入视频 <Plus size={16}/></button></div>}
            <div className="preview-badge lg"><span className="record-dot"/> 原片<span>{media?.height || '—'}p</span></div>
            
            <div className="transport" aria-label="播放控制">
              <IconButton label="跳到选段开头" onClick={() => seek(trim[0])} disabled={!media}><SkipBack size={17}/></IconButton>
              <IconButton label={playing ? '暂停' : '播放'} className="play-button" onClick={() => void togglePlay()} disabled={!media}>{playing ? <Pause size={18} fill="currentColor"/> : <Play size={18} fill="currentColor"/>}</IconButton>
              <IconButton label="跳到选段结尾" onClick={() => seek(trim[1])} disabled={!media}><SkipForward size={17}/></IconButton>
              <span className="transport-rule"/><span className="transport-time">{timecode(position, true)}<small> / {timecode(total, true)}</small></span>
              <span className="transport-rule"/><IconButton label={muted ? '打开声音' : '静音'} aria-pressed={!muted} onClick={() => setMuted(!muted)}>{muted ? <VolumeX size={18}/> : <Volume2 size={18}/>}</IconButton><IconButton label="全屏预览" onClick={() => { void stage.current?.requestFullscreen().catch(() => setNotice('当前窗口不支持全屏')); }}><Expand size={17}/></IconButton>
            </div>
          </div>
          <div className="source-info"><div><Film size={15}/><span title={media?.path}>{media?.filename || '尚未选择视频'}</span></div><span>{media ? `${media.width} × ${media.height} / ${media.fps.toFixed(0)} FPS` : 'LOCAL VIDEO'}</span></div>
          <div className="cut-inspector"><div className="cut-head"><span className="cut-symbol">[&nbsp;]</span><div><strong>{candidate ? `选段 ${String(candidates.indexOf(candidate) + 1).padStart(2,'0')}` : '片段范围'}</strong><small>{candidate ? '拖动时间线两侧手柄微调' : '先检索，再挑选值得留下的片段'}</small></div><span className="duration-chip">{(trim[1] - trim[0]).toFixed(2)}<small> 秒</small></span></div>
            <div className="trim-controls"><label>入点 <input aria-label="片段开始时间（秒）" type="number" min="0" max={trim[1] - .05} step="0.1" value={Number(trim[0].toFixed(2))} onChange={e => { const n = Number(e.target.value); if (n >= 0 && n < trim[1]) changeTrim([n, trim[1]]); }}/><span>s</span></label><ArrowRight size={16}/><label>出点 <input aria-label="片段结束时间（秒）" type="number" min={trim[0] + .05} max={total} step="0.1" value={Number(trim[1].toFixed(2))} onChange={e => { const n = Number(e.target.value); if (n > trim[0] && n <= total) changeTrim([trim[0], n]); }}/><span>s</span></label><button className="text-button lg" onClick={() => { if (candidate) changeTrim([candidate.start_seconds, candidate.end_seconds]); else changeTrim([0,total]); }}><History size={14}/>重置</button></div>
          </div>
          <div className="evidence-line"><span><i/>{currentRun ? `${routeNames[currentRun.route]}检索` : '等待选择片段'}</span><span>{candidate?.review_verdict === 'pass' ? <><ShieldCheck size={14}/> 已审核</> : <><ScanLine size={14}/> {candidate ? '待审核' : '等待检索'}</>}</span></div>
        </section>

        <aside className="search-panel" aria-label="检索与候选片段">
          <form className="search-composer" onSubmit={startRetrieval}>
            <MiniLabel>文字找片段</MiniLabel><div className="composer-title"><h2>先找到，<br/>再剪好<span>。</span></h2><span className="composer-glyph" aria-hidden="true">↗</span></div>
            <label className="sr-only" htmlFor="query">描述需要检索的视频片段</label><textarea id="query" value={query} onChange={e => setQuery(e.target.value)} placeholder="比如：找到赛车驶过，能听到引擎轰鸣的片段……" maxLength={2000} required rows={2}/>
            <div className="route-options" aria-label="本地检索路径">{Object.entries(routeNames).map(([value,label]) => <button className="lg" type="button" key={value} aria-pressed={route === value} onClick={() => setRoute(value)} disabled={busy || cloudPlanning}>{value === 'visual' ? <ScanLine size={13}/> : value === 'audio_visual' ? <AudioLines size={13}/> : <Layers3 size={13}/>} {label.replace('定位','')}</button>)}</div>
            <details className="model-inputs"><summary><SlidersHorizontal size={12}/> 检索设置 <ChevronDown size={13}/></summary><div>
              <label className="check-line"><input type="checkbox" checked={cloudPlanning} onChange={e => setCloudPlanning(e.target.checked)} disabled={busy}/>智能解析需求（云端服务，可能计费）</label>
              {!cloudPlanning && <>{route !== 'audio_visual' && <label className="field-label">英文视觉查询<input value={visualEn} onChange={e => setVisualEn(e.target.value)} placeholder="A racing car driving on a track"/></label>}{route !== 'visual' && <label className="field-label">声音事件<select value={events[0] || 'Truck'} onChange={e => setEvents([e.target.value])}>{data?.events.map(event => <option key={event.value} value={event.value}>{event.label}</option>)}</select></label>}</>}
              <p>声音检索支持列表中的事件，暂不支持具体台词。</p>
            </div></details>
            <button className="find-button lg" disabled={!data || !media || busy || !query.trim()} type="submit"><span>{busy ? <><LoaderCircle size={16} className="spin"/> 正在处理</> : '寻找片段'}</span>{busy ? <span>处理中</span> : <ArrowUpRight size={21}/>}</button>
          </form>
          <div className="candidate-section"><div className="candidate-heading"><h2>候选片段 <span>{String(visibleCandidates.length).padStart(2,'0')}</span></h2><IconButton label={onlyReviewed ? '显示全部候选' : '仅显示已审核'} aria-pressed={onlyReviewed} onClick={() => setOnlyReviewed(!onlyReviewed)}><SlidersHorizontal size={16}/></IconButton></div>
            {busy && job && <div className="job-progress" role="status"><LoaderCircle size={15} className="spin"/><div><strong>{job.stage}</strong></div></div>}
            <div className="candidates">{visibleCandidates.map((c,i) => <button key={c.candidate_id} className={`candidate ${candidateId === c.candidate_id ? 'active' : ''}`} onClick={() => chooseCandidate(c)} disabled={busy}><div className="candidate-image"><img src={`/api/thumb/${activeId}?t=${(c.start_seconds + c.end_seconds) / 2}`} alt={`候选片段 ${i+1} 的中间画面`}/><span>{String(i+1).padStart(2,'0')}</span></div><div className="candidate-body"><strong>{timecode(c.start_seconds, true)}<span>—</span>{timecode(c.end_seconds, true)}</strong><p>{c.model_evidence.map(e => data?.events.find(v => v.value === e.label)?.label || (e.modality === 'visual' ? '视觉匹配' : e.label)).filter((v,j,a) => a.indexOf(v) === j).join(' · ')}</p><div><span className={c.review_verdict === 'pass' ? 'verdict pass' : 'verdict'}>{c.review_verdict === 'pass' ? '已审核' : c.review_verdict === 'reject' ? '未通过' : '待审核'}</span><span className="model-score">{(c.end_seconds - c.start_seconds).toFixed(1)} 秒</span></div></div></button>)}
              {!visibleCandidates.length && <div className="candidate-empty"><ScanLine size={26}/><strong>{onlyReviewed ? '暂无已审核片段' : '等那一幕出现'}</strong><p>{onlyReviewed ? '关闭筛选查看全部模型候选。' : '输入需求后，候选片段会显示在这里。'}</p></div>}
            </div>
            {currentRun && <button className="run-origin" onClick={() => setSection('history')}><History size={12}/><span>检索记录 · {currentRun.created}</span><ArrowUpRight size={13}/></button>}
          </div>
        </aside>

        <section className="timeline-panel" aria-label="剪辑时间线">
          <div className="timeline-toolbar"><div className="timeline-title"><span className="section-index">03 /</span><h2>剪辑时间线</h2></div><div className="timeline-tools"><IconButton label="撤销裁剪（Ctrl+Z）" onClick={undo}><History size={16}/></IconButton><span className="tool-rule"/><button className="timeline-speed lg" onClick={() => setSpeed(speed === 1 ? 1.5 : speed === 1.5 ? 2 : .5 === speed ? 1 : .5)}>{speed}×</button><IconButton label="标记当前入点（I）" disabled={position >= trim[1] - .05} onClick={() => changeTrim([position, trim[1]])}><span className="keymark">[</span></IconButton><IconButton label="标记当前出点（O）" disabled={position <= trim[0] + .05} onClick={() => changeTrim([trim[0], position])}><span className="keymark">]</span></IconButton><span className="timeline-time">{timecode(position, true)}</span></div></div>
          <div className="timeline-body"><div className="track-labels"><span className="ruler-icon"><MousePointer2 size={13}/></span><span><Film size={14}/>V1</span><span><AudioLines size={14}/>A1</span></div><div className="track-area" ref={timeline}>
            <div className="time-ruler">{Array.from({ length: 9 },(_,i) => <button key={i} style={{ left: `${i * 12.5}%` }} onClick={() => seek(total * i / 8)} aria-label={`跳到 ${timecode(total * i / 8, true)}`}>{timecode(total * i / 8)}<i/></button>)}</div>
            <div className="video-track" onClick={e => { const b = e.currentTarget.getBoundingClientRect(); seek((e.clientX - b.left) / b.width * total); }}><div className="filmstrip">{media && Array.from({length:12},(_,i) => <img key={`${media.id}-${i}`} src={`/api/thumb/${media.id}?t=${((i + .5) / 12 * total).toFixed(2)}`} alt="" loading="lazy"/>)}</div><div className="track-dim left" style={{width:`${trim[0]/total*100}%`}}/><div className="track-dim right" style={{width:`${(total-trim[1])/total*100}%`}}/>
              <div className="trim-selection" style={{left:`${trim[0]/total*100}%`,width:`${(trim[1]-trim[0])/total*100}%`}}><span className="selection-title">{candidate ? `选段 ${String(candidates.indexOf(candidate)+1).padStart(2,'0')}` : '源素材'} <span>{(trim[1]-trim[0]).toFixed(2)}s</span></span>{([0,1] as const).map(side => <button key={side} className={`trim-handle ${side ? 'end' : 'start'}`} aria-label={side ? '调整出点' : '调整入点'} role="slider" aria-valuemin={0} aria-valuemax={total} aria-valuenow={trim[side]} aria-valuetext={timecode(trim[side],true)} onPointerDown={e => dragHandle(e,side)} onClick={e => e.stopPropagation()} onKeyDown={e => { if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.stopPropagation();e.preventDefault(); const d = e.key === 'ArrowLeft' ? -.1 : .1; const n: [number,number] = [...trim]; n[side] += d; if(n[0] >= 0 && n[0] < n[1]-.02 && n[1] <= total) changeTrim(n); } }}><i/></button>)}</div>
            </div>
            <div className="audio-track"><svg viewBox="0 0 1100 48" preserveAspectRatio="none" aria-label="音轨波形">{wave.map((v,i) => <line key={i} x1={i*5+2} x2={i*5+2} y1={24-v*21} y2={24+v*21}/>)}</svg>{waveStatus !== 'ready' && <span className="wave-status">{waveStatus === 'loading' ? '正在读取音轨…' : waveStatus === 'silent' ? '素材没有可解码音轨' : '音轨预览暂不可用'}</span>}{waveStatus === 'ready' && waveDuration < total - .1 && <span className="wave-status">波形仅覆盖前 {timecode(waveDuration)}</span>}</div>
            <div className="playhead" style={{left:`${position/total*100}%`}}><i/></div>
          </div></div>
          <div className="timeline-footer"><span><span className="live-dot"/> {job?.status === 'running' ? job.stage : '工作台就绪'}<span className="footer-divider">/</span>{candidate ? '精剪模式' : '原片预览'}</span><span>SPACE 播放 <span>·</span> I / O 标记 <span>·</span> ← → 逐秒定位</span></div>
        </section>
      </main>
    </div>

    {error && <div className="error-toast" role="alert"><div><strong>需要注意</strong><p>{error}</p></div><IconButton label="关闭错误提示" onClick={() => setError('')}><X size={18}/></IconButton></div>}
    {notice && <div className="notice-toast" role="status"><Check size={17}/><span>{notice}</span><IconButton label="关闭通知" onClick={() => setNotice('')}><X size={15}/></IconButton></div>}

    {modal(importDialog,'导入本地素材',<><p className="dialog-description">视频保存在本机，不会上传到外部服务器。大文件建议直接填写本地路径。</p><input ref={fileInput} type="file" accept="video/*,.mkv" hidden onChange={e => { const file = e.target.files?.[0]; if(file) void importMedia(file); e.target.value=''; }}/><button className="file-drop" onClick={() => fileInput.current?.click()} disabled={importing} onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); if(e.dataTransfer.files[0]) void importMedia(e.dataTransfer.files[0]); }}><Plus size={28}/><strong>{importing ? '正在导入…' : '选择视频，或拖放到这里'}</strong><span>复制到项目素材目录 · 最大 4 GB</span></button><div className="or-divider">或使用本地文件路径</div><form onSubmit={e => { e.preventDefault(); void importMedia(); }}><label className="field-label">视频绝对路径<input value={importPath} onChange={e => setImportPath(e.target.value)} placeholder="D:\Videos\my-video.mp4" required/></label><button className="primary lg" disabled={!importPath.trim() || importing} type="submit">导入素材 <ArrowRight size={17}/></button></form></>)}
    {modal(exportDialog,'导出片段',<><div className="export-summary"><img src={`/api/thumb/${activeId}?t=${(trim[0]+trim[1])/2}`} alt="所选片段预览"/><div><small>手动剪辑范围</small><strong>{timecode(trim[0],true)} — {timecode(trim[1],true)}</strong><span>{(trim[1]-trim[0]).toFixed(2)} 秒 · MP4 / H.264 · 保留音轨</span></div></div><p className="dialog-description">直接导出保留当前入点和出点。智能复核会检查原始候选，输出范围可能与手动调整不同。</p><button className="primary lg" disabled={busy || !candidate} onClick={() => void startExport()}><ArrowDownToLine size={17}/> 导出当前剪辑</button><div className="review-option"><ShieldCheck size={21}/><div><strong>先进行内容复核</strong><p>复核会将候选画面发送至云端服务，可能产生费用。</p><label className="check-line"><input type="checkbox" checked={cloudReview} onChange={e => setCloudReview(e.target.checked)}/>我同意发送候选画面并进行云端复核</label><button className="secondary lg" disabled={!cloudReview || busy || !candidate} onClick={() => void startExport(true)}>复核后导出 <ArrowUpRight size={15}/></button></div></div></>)}
    {modal(settingsDialog,'工作台设置',<><div className="setting-row"><div><strong>完整液态玻璃</strong><p>开启折射与边缘高光<br/>推荐使用 Chrome 或 Edge</p></div><button className="switch" role="switch" aria-checked={glass} aria-label="完整液态玻璃" onClick={() => setGlass(!glass)}><span/></button></div><div className="setting-row"><div><strong>素材与隐私</strong><p>视频与剪辑记录保存在本机<br/>云端复核只会在你授权后执行</p></div><Monitor size={22}/></div><p className="small-note">开启系统“减少动态效果”后，背景动画会自动暂停。</p></>)}
    {modal(helpDialog,'手感，来自快捷键',<div className="shortcut-list">{[['Space','播放 / 暂停'],['← / →','前后移动一秒'],['I','将当前帧设为入点'],['O','将当前帧设为出点'],['Ctrl / ⌘ + Z','撤销上次裁剪'],['Tab','在控件之间导航'],['Esc','关闭当前窗口']].map(([key,text]) => <div key={key}><span>{text}</span><kbd>{key}</kbd></div>)}</div>)}
    {output && <div className="output-panel"><div><Check size={17}/><strong>成片已保存</strong><IconButton label="关闭成片面板" onClick={() => setOutput(null)}><X size={16}/></IconButton></div><video src={output.url} controls playsInline/><p>{output.filename} · {timecode(output.duration,true)}</p><a href={output.url} download={output.filename}>下载 MP4 <ArrowDownToLine size={15}/></a></div>}
  </div>;
}
