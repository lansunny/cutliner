import { ArrowDownToLine, ArrowRight, History, Layers3 } from 'lucide-react';
import type { Artifact, Candidate, Run } from './studio-types';
import { formatTimecode, modelRange } from './editor-state';
import type { ClipRange } from './editor-state';

export function artifactStatus(artifact: Artifact) {
  if (artifact.file_status === 'missing') return '文件缺失';
  if (artifact.file_status === 'changed') return '文件已改变';
  if (artifact.source_current === false) return '源素材已改变';
  if (artifact.file_status === 'legacy') return '旧版记录';
  return artifact.reviewed ? '已复核' : '未复核';
}

const clock = (seconds: number) => formatTimecode(seconds, true);
const interval = (range?: ClipRange | null) => range ? `${clock(range[0])} — ${clock(range[1])}` : '—';

export default function ArtifactVersions({ run, selectedId, candidate, trim, disabled, onSelect, onApply }: {
  run: Run; selectedId: string; candidate?: Candidate; trim: ClipRange; disabled: boolean;
  onSelect: (id: string) => void; onApply: (artifact: Artifact) => void;
}) {
  const versions = run.deliveries || [];
  if (!versions.length) return null;
  const selected = versions.find(version => version.id === selectedId) || versions[0];
  const base = run.candidates.find(item => item.candidate_id === selected.candidate_id);
  const repair = selected.agent_boundary_repair;
  const currentNumber = candidate ? String(run.candidates.indexOf(candidate) + 1).padStart(2, '0') : '';

  return <div className="artifact-versions" aria-label="成片版本与范围">
    <Layers3 size={14}/>
    <select aria-label="成片版本" value={selected.id} disabled={disabled} onChange={event => onSelect(event.target.value)}>
      {versions.map(version => <option value={version.id} key={version.id}>v{String(version.version).padStart(2, '0')} · {version.kind === 'export' ? '手动导出' : 'Agent 成片'} · {artifactStatus(version)}</option>)}
    </select>
    <details className="artifact-range-details"><summary><History size={13}/>范围对照</summary>
      <div className="artifact-range-panel">
        <dl>
          <div><dt>模型建议</dt><dd>{interval(base ? modelRange(base) : null)}</dd></div>
          <div><dt>Agent 修正</dt><dd>{interval(repair ? [repair.proposed_start_seconds, repair.proposed_end_seconds] : null)}</dd></div>
          <div><dt>当前选段 {currentNumber}</dt><dd>{interval(trim)}</dd></div>
          <div className="artifact-selected-range"><dt>成片 v{String(selected.version).padStart(2, '0')}</dt><dd>{interval(selected.start !== undefined && selected.end !== undefined ? [selected.start, selected.end] : null)}</dd></div>
        </dl>
        <div className="artifact-range-actions"><span>{artifactStatus(selected)}</span>
          <button className="lg" disabled={disabled || !base || selected.source_current === false} onClick={() => onApply(selected)}>应用此范围<ArrowRight size={12}/></button>
          {selected.media && <a className="lg" href={selected.media.url} download={`cutliner_v${String(selected.version).padStart(2, '0')}.mp4`}><ArrowDownToLine size={12}/>下载 v{selected.version}</a>}
        </div>
      </div>
    </details>
  </div>;
}
