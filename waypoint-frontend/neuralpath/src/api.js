// All backend calls for WayPoint (explainable RAG-grounded resume-job matching)
export const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

async function check(res) {
  if (!res.ok) {
    let msg = await res.text()
    try { msg = JSON.parse(msg).detail || msg } catch { /* plain text */ }
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg))
  }
  return res.json()
}

/** Explainable match. Pass either resumeText or a File (PDF/DOCX/TXT). */
export async function matchResume({ resumeText, file, jobDescription, useLlm = false }) {
  if (file) {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('job_description', jobDescription)
    fd.append('use_llm', String(useLlm))
    return check(await fetch(`${BASE_URL}/match`, { method: 'POST', body: fd }))
  }
  return check(await fetch(`${BASE_URL}/match/text`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ resume_text: resumeText, job_description: jobDescription, use_llm: useLlm }),
  }))
}

export async function fairnessCheck({ resumeText, jobDescription, useLlm = false }) {
  return check(await fetch(`${BASE_URL}/match/fairness`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ resume_text: resumeText, job_description: jobDescription, use_llm: useLlm }),
  }))
}

export async function ragSearch(q, kind = null, k = 8) {
  const p = new URLSearchParams({ q, k: String(k) })
  if (kind) p.set('kind', kind)
  return check(await fetch(`${BASE_URL}/rag/search?${p}`))
}

export async function healthCheck() {
  try { const r = await fetch(`${BASE_URL}/health`); return r.ok ? r.json() : null } catch { return null }
}
