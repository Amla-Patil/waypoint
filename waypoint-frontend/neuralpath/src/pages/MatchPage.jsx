import { useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { ArrowLeft, CheckCircle2, CircleDashed, XCircle, ShieldCheck, ShieldAlert, Loader2, ExternalLink } from 'lucide-react'
import { fairnessCheck } from '../api'

const VERDICT = {
  met:     { label: 'Met',     cls: 'bg-mintbg text-accentDark border-accent/30', Icon: CheckCircle2 },
  partial: { label: 'Partial', cls: 'bg-amber-50 text-amber-700 border-amber-200', Icon: CircleDashed },
  not_met: { label: 'Not met', cls: 'bg-red-50 text-red-700 border-red-200',       Icon: XCircle },
}
const PRIO = { 'CORE GAP': 'text-red-700 bg-red-50 border-red-200', REFRESH: 'text-amber-700 bg-amber-50 border-amber-200', PREREQUISITE: 'text-skyDark bg-panel2 border-sky/30' }

function Section({ title, children, right }) {
  return (<div className="frosted rounded-2xl p-5 mb-5"><div className="flex items-center mb-3"><h2 className="font-display font-700 text-ink">{title}</h2><div className="ml-auto">{right}</div></div>{children}</div>)
}

export default function MatchPage() {
  const { state } = useLocation(); const navigate = useNavigate()
  const saved = state || (() => { try { return JSON.parse(sessionStorage.getItem('waypoint_match') || 'null') } catch { return null } })()
  const [fair, setFair] = useState(null); const [fairLoading, setFairLoading] = useState(false); const [fairErr, setFairErr] = useState('')
  if (!saved?.result) return (<div className="max-w-xl mx-auto py-24 text-center"><p className="text-dim mb-4">No result yet.</p>
    <button onClick={() => navigate('/upload')} className="text-accentDark underline text-sm">Analyze a resume</button></div>)
  const { result: r, resumeText, jdText, useLlm } = saved

  const runFair = async () => {
    setFairErr(''); setFairLoading(true)
    try { setFair(await fairnessCheck({ resumeText, jobDescription: jdText, useLlm })) }
    catch (e) { setFairErr(e.message) } finally { setFairLoading(false) }
  }

  return (
    <div className="max-w-5xl mx-auto px-6 py-10">
      <button onClick={() => navigate('/upload')} className="flex items-center gap-2 text-muted hover:text-ink text-sm mb-6"><ArrowLeft size={14} /> New analysis</button>

      <div className="frosted rounded-2xl p-6 mb-5 grid md:grid-cols-4 gap-4 items-center">
        <div className="md:col-span-1"><div className="text-muted text-xs font-mono">{r.role_title}</div>
          <div className="font-display text-5xl font-800 text-gradient">{r.match_score}</div><div className="text-dim text-xs">match score / 100 (weighted, computed in code)</div></div>
        <div><div className="font-display text-2xl font-700 text-ink">{r.baseline_keyword_score}</div><div className="text-dim text-xs">keyword-only baseline</div></div>
        <div><div className="font-display text-2xl font-700 text-ink">{r.semantic_similarity}</div><div className="text-dim text-xs">whole-document cosine similarity</div></div>
        <div className="text-xs font-mono text-muted">backend: {r.embedding_backend}<br />mode: {r.mode}</div>
      </div>
      {r.embedding_backend === 'tfidf' && <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg p-3 mb-5">
        Running without the sentence-embedding model: matching is stem-aware lexical only. Install <code>requirements-sbert.txt</code> for semantic matching.</p>}

      <Section title="Requirement-by-requirement evidence">
        <div className="overflow-x-auto"><table className="w-full text-sm">
          <thead><tr className="text-left text-muted text-xs font-mono"><th className="py-2 pr-3">Requirement</th><th className="pr-3">Weight</th><th className="pr-3">Verdict</th><th className="pr-3">Evidence from resume</th><th>Why</th></tr></thead>
          <tbody>{r.requirements.map((q) => { const v = VERDICT[q.verdict]; return (
            <tr key={q.requirement} className="border-t border-softborder align-top">
              <td className="py-2 pr-3 font-700 text-ink">{q.requirement}</td>
              <td className="pr-3 text-dim">{(q.weight * 100).toFixed(0)}%</td>
              <td className="pr-3"><span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-lg border text-xs font-mono ${v.cls}`}><v.Icon size={12} />{v.label}</span></td>
              <td className="pr-3 text-ink max-w-xs">{q.evidence ? <>“{q.evidence}” {q.evidence_verified && <span title="Quote found verbatim in the resume" className="text-accentDark text-xs">✓ verified</span>}</> : <span className="text-muted">—</span>}</td>
              <td className="text-dim text-xs">{q.reason}</td></tr>) })}</tbody></table></div>
      </Section>

      <Section title="Grounding: closest O*NET occupations (retrieved)">
        <div className="grid md:grid-cols-3 gap-3">{r.grounding.map((g) => (
          <div key={g.code} className="rounded-xl border border-softborder bg-white/60 p-3"><div className="font-700 text-ink text-sm">{g.title}</div>
            <div className="text-muted text-xs font-mono mb-1">{g.code} · similarity {g.similarity}</div>
            <div className="text-dim text-xs">{g.top_skills.slice(0, 4).join(', ')}</div></div>))}</div>
      </Section>

      <Section title="Skill-gap pathway" right={<span className="text-xs font-mono text-muted">{r.total_hours}h planned · {r.full_hours}h uncompressed</span>}>
        {r.modules.length === 0 ? <p className="text-dim text-sm">No gaps: every requirement is met.</p> :
          <ol className="space-y-3">{r.modules.map((m, i) => (
            <li key={m.module_id} className="rounded-xl border border-softborder bg-white/60 p-3">
              <div className="flex items-center gap-2 flex-wrap"><span className="text-muted font-mono text-xs">{i + 1}</span>
                <span className="font-700 text-ink">{m.title}</span><span className="text-dim text-xs">{m.hours}h</span>
                <span className={`px-2 py-0.5 rounded-lg border text-xs font-mono ${PRIO[m.priority]}`}>{m.priority}</span></div>
              <div className="text-dim text-xs mt-1">{m.why_included}{m.prerequisites.length > 0 && <> Needs: {m.prerequisites.join(', ')}.</>}</div>
              {m.course_url ? <a href={m.course_url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-xs text-accentDark mt-1 hover:underline">
                  {m.course_title} <ExternalLink size={11} /> <span className="text-muted">(retrieved, similarity {m.course_similarity})</span></a>
                : <div className="text-muted text-xs mt-1">No sufficiently similar course in the dataset.</div>}
            </li>))}</ol>}
        {r.resources.length > 0 && <div className="mt-4 text-xs text-dim">{r.resources.map((x) => (
          <div key={x.skill}><b>{x.skill}</b> (no catalogue module): {x.courses.map((c) => <a key={c.url} href={c.url} target="_blank" rel="noreferrer" className="text-accentDark hover:underline mr-2">{c.title}</a>)}</div>))}</div>}
      </Section>

      <Section title="Fairness and robustness check" right={<button onClick={runFair} disabled={fairLoading || !resumeText}
        className="px-3 py-1.5 btn-gradient text-white text-xs font-mono rounded-lg disabled:opacity-50">{fairLoading ? <Loader2 size={12} className="animate-spin inline" /> : 'Run check'}</button>}>
        {!resumeText && <p className="text-muted text-xs">Paste the resume as text on the Analyze page to enable this check.</p>}
        {fairErr && <p className="text-red-700 text-xs">{fairErr}</p>}
        {fair && <div className="text-sm space-y-3">
          <div className="flex items-center gap-2">{fair.identity_pass ? <ShieldCheck size={16} className="text-accentDark" /> : <ShieldAlert size={16} className="text-red-700" />}
            <span>Identity variants (name + gendered words): max shift <b>{fair.identity_max_shift}</b> points (tolerance {fair.tolerance}) — {fair.identity_pass ? 'pass' : 'FAIL'}</span></div>
          <table className="text-xs w-full"><thead><tr className="text-left text-muted font-mono"><th>Variant</th><th>Score</th><th>Shift vs {fair.base_score}</th></tr></thead><tbody>
            {fair.identity.map((x) => <tr key={x.group} className="border-t border-softborder"><td>{x.group}: {x.name}</td><td>{x.score}</td><td>{(x.score - fair.base_score).toFixed(1)}</td></tr>)}
            {fair.robustness.map((x) => <tr key={x.variant} className="border-t border-softborder"><td>{x.variant}</td><td>{x.score}</td><td className={x.shift > fair.tolerance ? 'text-red-700 font-700' : ''}>{x.shift > 0 ? '+' : ''}{x.shift}</td></tr>)}</tbody></table>
          {fair.stuffing_vulnerable && <p className="text-amber-700 text-xs">Keyword stuffing raised the score: this matcher is vulnerable to it (known limitation).</p>}
          <ul className="text-muted text-xs list-disc pl-5">{fair.notes.map((n) => <li key={n}>{n}</li>)}</ul></div>}
      </Section>

      <Section title="Reasoning trace (generated by the program from computed values)">
        <pre className="text-xs font-mono text-dim whitespace-pre-wrap">{r.reasoning_trace.join('\n')}</pre>
      </Section>
    </div>
  )
}
