import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, FileText, Briefcase, ArrowRight, Loader2, AlertCircle } from 'lucide-react'
import { matchResume } from '../api'

const SAMPLE_JD = `ML Engineer
Required: Python, PyTorch, REST APIs
Must have experience with Docker and Kubernetes
Nice to have: MLflow, Spark`

export default function UploadPage() {
  const [resumeText, setResumeText] = useState('')
  const [jdText, setJdText] = useState('')
  const [file, setFile] = useState(null)
  const [useLlm, setUseLlm] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const fileRef = useRef()
  const navigate = useNavigate()

  const onFile = (e) => {
    e.preventDefault()
    const f = e.dataTransfer?.files?.[0] || e.target.files?.[0]
    if (!f) return
    if (/\.(pdf|docx)$/i.test(f.name)) { setFile(f); setResumeText('') }
    else { setFile(null); const r = new FileReader(); r.onload = (ev) => setResumeText(ev.target.result); r.readAsText(f); setFile({ name: f.name, textOnly: true }) }
  }

  const submit = async () => {
    const realFile = file && !file.textOnly ? file : null
    if (!realFile && !resumeText.trim()) return setError('Provide a resume: upload a file or paste text.')
    if (!jdText.trim()) return setError('Provide the job description.')
    setError(''); setLoading(true)
    try {
      const result = await matchResume({ resumeText, file: realFile, jobDescription: jdText, useLlm })
      // keep inputs so the fairness check can re-run on the same resume text
      sessionStorage.setItem('waypoint_match', JSON.stringify({ result, resumeText: realFile ? '' : resumeText, jdText, useLlm }))
      navigate('/match', { state: { result, resumeText: realFile ? '' : resumeText, jdText, useLlm } })
    } catch (e) { setError(e.message || 'Request failed. Is the backend running?') }
    finally { setLoading(false) }
  }

  return (
    <div className="max-w-5xl mx-auto px-6 py-10">
      <h1 className="font-display text-3xl font-800 text-ink mb-2">Analyze a resume against a job</h1>
      <p className="text-dim text-sm mb-8">List the required skills in the job description (a line such as “Required: Python, SQL” works well). “Must / required” lines weigh more than “nice to have” lines.</p>
      <div className="grid md:grid-cols-2 gap-5">
        <div className="frosted rounded-2xl p-5">
          <div className="flex items-center gap-2 mb-3 text-ink font-display font-700"><FileText size={16} className="text-accentDark" /> Resume</div>
          <div onClick={() => fileRef.current.click()} onDragOver={(e) => e.preventDefault()} onDrop={onFile}
            className="border-2 border-dashed border-softborder rounded-xl p-4 text-center cursor-pointer hover:border-accent/60 mb-3">
            <Upload size={18} className="mx-auto text-muted mb-1" />
            <div className="text-sm text-dim">{file ? file.name : 'Drop PDF / DOCX / TXT, or click'}</div>
            <input ref={fileRef} type="file" accept=".pdf,.docx,.txt" className="hidden" onChange={onFile} />
          </div>
          <textarea value={resumeText} onChange={(e) => { setResumeText(e.target.value); if (file) setFile(null) }} rows={11}
            placeholder="…or paste resume text" className="w-full rounded-xl border border-softborder bg-white/70 p-3 text-sm font-mono outline-none focus:border-accent" />
        </div>
        <div className="frosted rounded-2xl p-5">
          <div className="flex items-center gap-2 mb-3 text-ink font-display font-700"><Briefcase size={16} className="text-accentDark" /> Job description
            <button onClick={() => setJdText(SAMPLE_JD)} className="ml-auto text-xs font-mono text-accentDark hover:underline">use sample</button></div>
          <textarea value={jdText} onChange={(e) => setJdText(e.target.value)} rows={16} placeholder="Paste the job description"
            className="w-full rounded-xl border border-softborder bg-white/70 p-3 text-sm font-mono outline-none focus:border-accent" />
        </div>
      </div>
      <label className="flex items-center gap-2 mt-5 text-sm text-dim">
        <input type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} />
        Also ask the LLM to judge (needs GROQ_API_KEY on the server; its quotes are verified and rejected if not found in the resume)
      </label>
      {error && <div className="mt-4 flex items-start gap-2 text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg p-3"><AlertCircle size={16} className="mt-0.5" /> {error}</div>}
      <button onClick={submit} disabled={loading}
        className="mt-6 inline-flex items-center gap-2 px-7 py-3 btn-gradient text-white font-display font-700 rounded-xl shadow disabled:opacity-60">
        {loading ? <><Loader2 size={16} className="animate-spin" /> Matching…</> : <>Run match <ArrowRight size={16} /></>}
      </button>
    </div>
  )
}
