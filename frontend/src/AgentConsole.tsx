import { useEffect, useRef } from 'react';
import { ArrowDownToLine, ArrowRight, Check, Circle, CircleAlert, Clapperboard, LoaderCircle, ScanLine, Scissors, ShieldCheck, SlidersHorizontal, Workflow } from 'lucide-react';
import type { Delivery, Job, Run } from './studio-types';
import { formatTimecode } from './editor-state';

const steps = [
  { id: 'planning', title: '理解目标', icon: Workflow },
  { id: 'retrieval', title: '模型定位', icon: ScanLine },
  { id: 'refinement', title: '边界精修', icon: SlidersHorizontal },
  { id: 'review', title: 'Agent 取证', icon: ShieldCheck },
  { id: 'boundary_repair', title: '边界修正', icon: SlidersHorizontal },
  { id: 'export', title: '自动裁剪', icon: Scissors },
  { id: 'verify', title: '成片复核', icon: Clapperboard },
];
const clock = (seconds: number) => formatTimecode(seconds, true);

export default function AgentConsole({ job, run, delivery, onFineTune }: { job: Job | null; run?: Run; delivery: Delivery | null; onFineTune: () => void }) {
  const running = job?.status === 'running' || job?.status === 'cancelling';
  const historicalView = !running && Boolean(delivery?.id && delivery.id !== run?.latest_artifact_id);
  const events = historicalView ? delivery?.events || [] : job?.events || run?.events || [];
  const delivered = Boolean(delivery?.reviewed);
  const finishedWithoutOutput = !running && !historicalView && Boolean(job && (['failed', 'cancelled', 'interrupted'].includes(job.status) || job.result?.outcome && job.result.outcome !== 'delivered') || run && ['no_match', 'review_blocked', 'needs_semantic_review', 'operation_failed', 'cancelled', 'interrupted', 'output_changed'].includes(run.review_status));
  const activeSteps = steps.filter(step =>
    (step.id !== 'refinement' || run?.candidates.some(candidate => candidate.boundary_refinement) || events.some(event => event.stage_id === 'refinement'))
    && (step.id !== 'boundary_repair' || run?.candidates.some(candidate => candidate.agent_boundary_repair) || events.some(event => event.stage_id === 'boundary_repair')));
  const last = useRef<HTMLDivElement>(null);
  useEffect(() => { const el = last.current; if (el) el.scrollTop = el.scrollHeight; }, [events.length]);
  const status = running ? job.stage : historicalView ? `正在查看成片 v${delivery?.version} 的记录` : finishedWithoutOutput ? job?.error || job?.result?.message || job?.stage || (run?.review_status === 'review_blocked' ? '复核受阻，等待人工确认' : '当前范围未交付') : delivered ? '复核通过' : delivery?.kind === 'export' ? '手动导出 · 未复核' : '';

  return <section className="agent-console" aria-label="Agent 自动剪辑执行记录">
    <div className="agent-console-head"><div><span className={`agent-signal ${running ? 'working' : delivered && !finishedWithoutOutput ? 'delivered' : ''}`}>{running ? <LoaderCircle size={16} className="spin"/> : <Workflow size={16}/>}</span><h2>Agent 行动台</h2><span className="agent-state">{historicalView ? '历史版本' : job?.status === 'cancelling' ? '正在停止' : job?.status === 'cancelled' ? '已停止' : job?.status === 'interrupted' ? '已中断' : running ? '正在执行' : finishedWithoutOutput ? '待调整' : delivered ? '已交付' : delivery?.kind === 'export' ? '手动导出' : '待命'}</span></div><span className="agent-status" role="status">{status}</span></div>
    <div className="agent-console-body">
      <div className="agent-execution">
        <ol className="agent-steps" aria-label="自动剪辑流程" style={{ gridTemplateColumns: `repeat(${activeSteps.length},minmax(0,1fr))` }}>{activeSteps.map(({ id, title, icon: Icon }, i) => {
          const active = running && job?.stage_id === id;
          const visited = events.some(e => e.stage_id === id);
          const finished = delivered && !finishedWithoutOutput && visited;
          return <li key={id} className={`${active ? 'active' : ''} ${visited ? 'visited' : ''} ${finished ? 'finished' : ''}`} aria-current={active ? 'step' : undefined}><span>{finished ? <Check size={14}/> : active ? <LoaderCircle size={14} className="spin"/> : <Icon size={14}/>}</span><div><small>0{i + 1}</small><strong>{title}</strong></div>{i < activeSteps.length - 1 && <ArrowRight className="step-arrow" size={12}/>}</li>;
        })}</ol>
        <div className="agent-event-list" ref={last} role="log" aria-label="实际执行记录" aria-live="off">
          {events.length ? events.map(event => <div className={`agent-event ${event.kind || ''}`} key={event.seq}><time>{new Date(event.at * 1000).toLocaleTimeString('zh-CN', { hour12: false })}</time>{event.kind === 'delivered' ? <Check size={12}/> : event.kind === 'attention' || event.kind === 'no_match' ? <CircleAlert size={12}/> : <Circle size={7}/>}<p>{event.message}</p>{event.tool && <span className="event-tool">{{inspect_frames:'画面取证',inspect_av_events:'声画检测',inspect_context:'邻近补看',propose_boundary:'边界修正',verify_export_audio:'音轨校验'}[event.tool] || event.tool}</span>}</div>) : <div className="agent-log-empty"><span>暂无执行记录</span></div>}
        </div>
      </div>
      <div className={`agent-delivery ${delivered ? 'is-delivered' : ''}`}>
        <div className="delivery-label"><Clapperboard size={15}/><span>{delivery?.version ? `成片 v${delivery.version}` : finishedWithoutOutput && delivery ? '保留上一成片' : delivery && !delivery.reviewed ? '当前导出' : '自动交付'}</span>{delivered && <span className="delivery-check"><ShieldCheck size={12}/> 双重复核</span>}</div>
        {delivery ? <><strong>{delivery.media.duration.toFixed(2)}<small> 秒成片</small></strong><p>{delivery.start !== undefined && delivery.end !== undefined ? `原片 ${clock(delivery.start)} — ${clock(delivery.end)}` : delivery.reviewed ? '已通过内容复核' : '手动导出 · 未复核'}</p><div className="delivery-actions"><a className="lg" href={delivery.media.url} download={delivery.version ? `cutliner_v${String(delivery.version).padStart(2, '0')}.mp4` : delivery.media.filename}><ArrowDownToLine size={14}/>下载成片</a><button className="lg" onClick={onFineTune}><SlidersHorizontal size={14}/>手动精调</button></div></> : <strong className="delivery-waiting">{running ? '生成中…' : '暂无成片'}</strong>}
      </div>
    </div>
  </section>;
}
