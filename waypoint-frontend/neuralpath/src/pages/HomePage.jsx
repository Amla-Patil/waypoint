import { useNavigate } from 'react-router-dom'
import { ArrowRight, Search, Database, FileCheck2, Scale, GitBranch } from 'lucide-react'

const PIPELINE = [
  { label: 'Parse',      sub: 'resume + JD',          color: 'text-accentDark' },
  { label: 'Embed',      sub: 'semantic similarity',  color: 'text-skyDark' },
  { label: 'Retrieve',   sub: 'O*NET + courses (RAG)', color: 'text-lavDark' },
  { label: 'Explain',    sub: 'verified evidence',    color: 'text-accentDark' },
  { label: 'Audit',      sub: 'fairness + robustness', color: 'text-skyDark' },
  { label: 'Recommend',  sub: 'ordered skill-gap path', color: 'text-lavDark' },
]
const FEATURES = [
  { icon: Search,     title: 'Semantic matching',          desc: 'Requirements are compared with resume sentences using embeddings and a stem-aware lexical check, not keywords alone.' },
  { icon: Database,   title: 'Retrieval-grounded',         desc: 'A hybrid index over O*NET occupations, the course catalogue and Coursera courses supplies sources for every recommendation.' },
  { icon: FileCheck2, title: 'Evidence you can verify',    desc: 'Each verdict quotes the resume. A verifier rejects any quote that is not literally in the resume, and the score is computed in code.' },
  { icon: Scale,      title: 'Fairness and robustness',    desc: 'Counterfactual name/gender variants, typos, reordering and keyword stuffing show how stable the score really is.' },
  { icon: GitBranch,  title: 'Prerequisite-ordered gaps',  desc: 'Missing skills map to catalogue modules; prerequisites are added and sorted topologically so the order is valid.' },
]

export default function HomePage() {
  const navigate = useNavigate()
  return (
    <div className="min-h-screen">
      <section className="max-w-4xl mx-auto px-6 pt-16 pb-12 text-center">
        <h1 className="font-display text-5xl md:text-6xl font-800 leading-[1.05] tracking-tight mb-4 text-ink">
          Match resumes to roles.<br /><span className="text-gradient">Show the evidence.</span>
        </h1>
        <p className="text-dim text-base md:text-lg max-w-2xl mx-auto mb-8 leading-relaxed">
          WayPoint is an explainable, retrieval-grounded resume–job matching prototype. It scores fit requirement by requirement,
          quotes the supporting resume text, audits its own stability, and lists the skills to learn next.
        </p>
        <button onClick={() => navigate('/upload')}
          className="group inline-flex items-center gap-2 px-7 py-3.5 btn-gradient text-white font-display font-700 text-base rounded-xl transition-all hover:scale-105 shadow-lg">
          Analyze a resume <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
        </button>
      </section>

      <section className="max-w-6xl mx-auto px-6 pb-14">
        <div className="frosted rounded-2xl p-6 shadow-sm overflow-x-auto">
          <div className="text-muted text-xs font-mono mb-4">Pipeline</div>
          <div className="flex items-center gap-2 min-w-max">
            {PIPELINE.map(({ label, sub, color }, i) => (
              <div key={label} className="flex items-center gap-2">
                <div className="flex flex-col items-center">
                  <span className={`font-mono text-sm font-700 ${color}`}>{label}</span>
                  <span className="text-muted text-xs">{sub}</span>
                </div>
                {i < PIPELINE.length - 1 && <div className="flex items-center gap-1 text-muted mx-2"><div className="w-5 h-px bg-softborder" /><ArrowRight size={10} /></div>}
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="max-w-6xl mx-auto px-6 pb-24">
        <div className="grid md:grid-cols-2 gap-4">
          {FEATURES.map(({ icon: Icon, title, desc }) => (
            <div key={title} className="frosted rounded-xl p-6 hover:shadow-md transition-all">
              <div className="w-10 h-10 rounded-lg bg-accent/20 flex items-center justify-center mb-4"><Icon size={18} className="text-accentDark" /></div>
              <h3 className="font-display font-700 text-base mb-2 text-ink">{title}</h3>
              <p className="text-dim text-sm leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
        <p className="text-muted text-xs font-mono text-center mt-8">
          Research prototype using public or synthetic data. Not a production applicant-tracking system; a human makes the hiring decision.
        </p>
      </section>
    </div>
  )
}
