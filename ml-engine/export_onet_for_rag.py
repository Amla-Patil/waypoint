"""Export O*NET occupations (title, description, top skills) to JSON for the backend RAG index.
Run once after build_skill_dag.py has produced onet_data/skill_dag.pkl:
    python ml-engine/export_onet_for_rag.py
"""
import json, os, pickle
HERE = os.path.dirname(__file__)
d = pickle.load(open(os.path.join(HERE, 'onet_data', 'skill_dag.pkl'), 'rb'))
occ, mat = d['occ_df'], d['skill_matrix']
out = []
for _, r in occ.iterrows():
    code = r['O*NET-SOC Code']
    top = []
    if code in mat.index:
        top = [f"{k} ({v:.1f})" for k, v in mat.loc[code].sort_values(ascending=False).head(6).items()]
    out.append({'code': code, 'title': r['Title'], 'description': r['Description'], 'top_skills': top})
dst = os.path.join(HERE, '..', 'waypoint-backend', 'app', 'onet_occupations.json')
json.dump(out, open(dst, 'w'), ensure_ascii=False)
print('wrote', len(out), 'occupations ->', os.path.abspath(dst))
