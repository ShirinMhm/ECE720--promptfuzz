"""
get_coverage_metric.py  —  with --plot support
"""

import argparse, os, sys, json, ast
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'PromptFuzz', 'Fuzzer')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'PromptFuzz')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_defenses(f):
    with open(f, encoding='utf-8') as fp:
        return [json.loads(l) for l in fp if l.strip()]

def build_cluster_labels(defenses, k, openai_key, cache=None):
    from sklearn.cluster import KMeans
    from gptfuzzer.llm import OpenAIEmbeddingLLM
    n = len(defenses); k = min(k, n)
    emb = None
    if cache and os.path.exists(cache):
        emb = np.load(cache)
        if len(emb) != n: emb = None
    if emb is None:
        print(f"Computing {n} embeddings...")
        m = OpenAIEmbeddingLLM("text-embedding-ada-002", openai_key)
        texts = [(d.get("pre_prompt","")+" "+d.get("post_prompt","")).strip() for d in defenses]
        emb = np.array([m.get_embedding(t) for t in texts], dtype=np.float32)
        if cache: np.save(cache, emb); print(f"Saved to {cache}")
    km = KMeans(n_clusters=k, random_state=42, n_init=10); km.fit(emb)
    return km.labels_

def load_csv(path):
    try: df = pd.read_csv(path)
    except: df = pd.read_csv(path, on_bad_lines='skip').dropna()
    df['results_list'] = df['results'].apply(lambda x: ast.literal_eval(str(x)))
    return df

def compute(df, labels, ndef, top_k=5):
    nc = int(labels.max())+1
    mat = np.array([( list(r)+[0]*ndef )[:ndef] for r in df['results_list']], dtype=int)
    asr = mat.sum(1)/ndef
    best_asr = float(asr.max())
    ens = np.bitwise_or.reduce(mat[np.argsort(asr)[-top_k:]])
    ens_asr = float(ens.sum()/ndef)
    def_cov = float(mat.any(0).sum()/ndef)
    cr = {}
    for c in range(nc):
        idx = np.where(labels==c)[0]
        cr[c] = {'covered': bool(mat[:,idx].any()),
                 'defenses_in_cluster': len(idx),
                 'defenses_broken': int(mat[:,idx].any(0).sum())}
    fam_cov = sum(v['covered'] for v in cr.values())/nc
    fam_int = sum(v['covered'] for v in cr.values())
    tq = int(df['query'].max()) if 'query' in df.columns else len(df)
    eff = fam_int/(tq/100) if tq>0 else 0.0
    covered_set=set(); cq=[]; cc=[]
    for _,row in df.iterrows():
        q=row.get('query',len(cq)+1)
        for i,h in enumerate(row['results_list']):
            if h==1 and i<len(labels): covered_set.add(int(labels[i]))
        cq.append(q); cc.append(len(covered_set)/nc)
    def qth(t):
        for q,c in zip(cq,cc):
            if c>=t: return q
        return None
    return dict(best_asr=round(best_asr,4), ensemble_asr=round(ens_asr,4),
                defense_coverage=round(def_cov,4), family_coverage=round(fam_cov,4),
                families_broken=f"{fam_int}/{nc}", families_broken_int=fam_int,
                n_clusters=nc, total_queries=tq,
                efficiency_per_100q=round(eff,4),
                queries_to_25pct=qth(0.25), queries_to_50pct=qth(0.50), queries_to_75pct=qth(0.75),
                cluster_report=cr, coverage_curve=(cq,cc), results_matrix=mat)

def print_single(label, m):
    cr=m['cluster_report']
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    for k in ['best_asr','ensemble_asr','defense_coverage','family_coverage',
              'families_broken','total_queries','efficiency_per_100q',
              'queries_to_25pct','queries_to_50pct','queries_to_75pct']:
        print(f"  {k:<44} {m[k]}")
    print(f"\n  Per-cluster breakdown:")
    print(f"  {'Cluster':>8}  {'Covered':>8}  {'In cluster':>12}  {'Broken':>8}")
    for c,v in sorted(cr.items()):
        print(f"  {c:>8}  {'✓' if v['covered'] else '✗':>8}  {v['defenses_in_cluster']:>12}  {v['defenses_broken']:>8}")

def print_comparison(labels, mlist):
    keys=['best_asr','ensemble_asr','defense_coverage','family_coverage',
          'families_broken','total_queries','efficiency_per_100q',
          'queries_to_25pct','queries_to_50pct','queries_to_75pct']
    w=max(len(l) for l in labels)+4
    print(f"\n{'Metric':<46}"+"".join(f"{l:>{w}}" for l in labels))
    print("-"*(46+w*len(labels)))
    for k in keys:
        print(f"  {k:<44}"+"".join(f"{str(m.get(k,'N/A')):>{w}}" for m in mlist))

def make_plots(labels, all_metrics, plot_dir):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(plot_dir, exist_ok=True)
    COLS = ['#2E5FAB','#059669','#D97706','#DC2626','#7C3AED']
    cols = COLS[:len(labels)]

    # ── 1. Efficiency ────────────────────────────────────────────────────────
    fig,ax=plt.subplots(figsize=(7,4.5))
    vals=[m['efficiency_per_100q'] for m in all_metrics]
    bars=ax.bar(labels,vals,color=cols,width=0.5,edgecolor='white',linewidth=1.2)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.03,f'{v:.2f}',
                ha='center',va='bottom',fontsize=13,fontweight='bold')
    ax.set_ylabel('Families Broken per 100 Queries',fontsize=12)
    ax.set_title('Query Efficiency Comparison',fontsize=14,fontweight='bold',pad=12)
    ax.set_ylim(0,max(vals)*1.35); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); p=os.path.join(plot_dir,'1_efficiency.png'); plt.savefig(p,dpi=150,bbox_inches='tight'); plt.close(); print(f"Saved: {p}")

    # ── 2. Total queries ─────────────────────────────────────────────────────
    fig,ax=plt.subplots(figsize=(7,4.5))
    vals=[m['total_queries'] for m in all_metrics]
    bars=ax.bar(labels,vals,color=cols,width=0.5,edgecolor='white',linewidth=1.2)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+5,str(v),
                ha='center',va='bottom',fontsize=13,fontweight='bold')
    ax.set_ylabel('Total API Queries',fontsize=12)
    ax.set_title('Total Query Cost Comparison',fontsize=14,fontweight='bold',pad=12)
    ax.set_ylim(0,max(vals)*1.25); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); p=os.path.join(plot_dir,'2_queries.png'); plt.savefig(p,dpi=150,bbox_inches='tight'); plt.close(); print(f"Saved: {p}")

    # ── 3. Grouped metrics ───────────────────────────────────────────────────
    mkeys=['best_asr','ensemble_asr','defense_coverage','family_coverage']
    mnames=['Best ASR','Ensemble ASR','Defense\nCoverage','Family\nCoverage']
    x=np.arange(len(mkeys)); w=0.8/len(labels)
    fig,ax=plt.subplots(figsize=(9,5))
    for i,(lbl,m,col) in enumerate(zip(labels,all_metrics,cols)):
        vs=[m[k] for k in mkeys]; off=(i-len(labels)/2+0.5)*w
        bars=ax.bar(x+off,vs,w,label=lbl,color=col,edgecolor='white',linewidth=0.8)
        for b,v in zip(bars,vs):
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.005,f'{v:.2f}',
                    ha='center',va='bottom',fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(mnames,fontsize=11)
    ax.set_ylabel('Score',fontsize=12); ax.set_title('Metric Comparison',fontsize=14,fontweight='bold',pad=12)
    ax.set_ylim(0,1.18); ax.legend(fontsize=10,framealpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); p=os.path.join(plot_dir,'3_metrics.png'); plt.savefig(p,dpi=150,bbox_inches='tight'); plt.close(); print(f"Saved: {p}")

    # ── 4. Radar — per cluster ───────────────────────────────────────────────
    nc=all_metrics[0]['n_clusters']
    angles=np.linspace(0,2*np.pi,nc,endpoint=False).tolist(); angles+=angles[:1]
    fig,ax=plt.subplots(figsize=(6,6),subplot_kw=dict(polar=True))
    for lbl,m,col in zip(labels,all_metrics,cols):
        cr=m['cluster_report']
        vs=[1 if cr[c]['covered'] else 0 for c in range(nc)]; vs+=vs[:1]
        ax.plot(angles,vs,'o-',linewidth=2,color=col,label=lbl)
        ax.fill(angles,vs,alpha=0.12,color=col)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels([f'C{c}' for c in range(nc)],fontsize=10)
    ax.set_yticks([0,1]); ax.set_yticklabels(['Not Broken','Broken'],fontsize=8); ax.set_ylim(0,1.25)
    ax.set_title('Per-Cluster Coverage\n(Defense Families)',fontsize=13,fontweight='bold',pad=20)
    ax.legend(loc='upper right',bbox_to_anchor=(1.35,1.1),fontsize=10)
    plt.tight_layout(); p=os.path.join(plot_dir,'4_radar.png'); plt.savefig(p,dpi=150,bbox_inches='tight'); plt.close(); print(f"Saved: {p}")

    # ── 5. Coverage curve ────────────────────────────────────────────────────
    fig,ax=plt.subplots(figsize=(8,5))
    for lbl,m,col in zip(labels,all_metrics,cols):
        qs,cs=m['coverage_curve']
        ax.step(qs,[c*100 for c in cs],where='post',color=col,linewidth=2.5,label=lbl)
        if qs: ax.scatter(qs[-1],cs[-1]*100,color=col,s=80,zorder=5)
    ax.axhline(y=100,color='gray',linestyle='--',linewidth=1,alpha=0.4,label='100% coverage')
    ax.set_xlabel('Total Queries (API Calls)',fontsize=12)
    ax.set_ylabel('Defense Families Covered (%)',fontsize=12)
    ax.set_title('Coverage Curve: Family Coverage vs Query Budget',fontsize=13,fontweight='bold',pad=12)
    ax.set_ylim(0,115); ax.legend(fontsize=10,framealpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); p=os.path.join(plot_dir,'5_coverage_curve.png'); plt.savefig(p,dpi=150,bbox_inches='tight'); plt.close(); print(f"Saved: {p}")

    # ── 6. Stage savings ─────────────────────────────────────────────────────
    fig,ax=plt.subplots(figsize=(8,4.5))
    stages=['Stage 1\n(20% defenses)','Stage 2\n(50% defenses)','Stage 3\n(100% defenses)']
    promoted=[2,1,1]; eliminated=[3,1,0]; savings=[360,75,0]
    x=np.arange(3); w=0.35
    b1=ax.bar(x-w/2,promoted,w,label='Promoted',color='#059669',edgecolor='white')
    b2=ax.bar(x+w/2,eliminated,w,label='Eliminated',color='#DC2626',edgecolor='white')
    for b,v in zip(b1,promoted):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.05,str(v),ha='center',va='bottom',fontsize=12,fontweight='bold')
    for b,v,s in zip(b2,eliminated,savings):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.05,str(v),ha='center',va='bottom',fontsize=12,fontweight='bold',color='#DC2626')
        if s>0:
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.4,f'~{s}q saved',
                    ha='center',va='bottom',fontsize=9,color='#7B3F00',style='italic')
    ax.set_xticks(x); ax.set_xticklabels(stages,fontsize=11)
    ax.set_ylabel('Number of Mutants',fontsize=12)
    ax.set_title('Multi-Fidelity Scheduler: Stage-by-Stage Elimination\n(~435 queries saved in one iteration)',
                 fontsize=13,fontweight='bold',pad=12)
    ax.set_ylim(0,6); ax.legend(fontsize=10,framealpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); p=os.path.join(plot_dir,'6_stage_savings.png'); plt.savefig(p,dpi=150,bbox_inches='tight'); plt.close(); print(f"Saved: {p}")

    print(f"\nAll 6 plots saved to: {os.path.abspath(plot_dir)}")

def main(args):
    defenses=load_defenses(args.defense_file)
    ndef=len(defenses)
    labels_arr=build_cluster_labels(defenses,args.cluster_k,args.openai_key,args.embedding_cache)

    if args.compare:
        assert args.result_csvs and args.labels
        assert len(args.result_csvs)==len(args.labels)
        all_m=[]
        for path,lbl in zip(args.result_csvs,args.labels):
            df=load_csv(path); m=compute(df,labels_arr,ndef,args.top_k); all_m.append(m)
        printable=[{k:v for k,v in m.items() if k not in ('cluster_report','coverage_curve','results_matrix')} for m in all_m]
        print_comparison(args.labels,printable)
        if args.save_path:
            pd.DataFrame([{'condition':l,**p} for l,p in zip(args.labels,printable)]).to_csv(args.save_path,index=False)
            print(f"\nSaved to {args.save_path}")
        if args.plot: make_plots(args.labels,all_m,args.plot_dir)
    else:
        assert args.result_csv
        df=load_csv(args.result_csv); m=compute(df,labels_arr,ndef,args.top_k)
        lbl=os.path.basename(args.result_csv); print_single(lbl,m)
        if args.save_path:
            pd.DataFrame([{k:v for k,v in m.items() if k not in ('cluster_report','coverage_curve','results_matrix')}]).to_csv(args.save_path,index=False)
        if args.plot: make_plots([lbl],[m],args.plot_dir)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--result_csv',type=str,default=None)
    p.add_argument('--compare',action='store_true')
    p.add_argument('--result_csvs',type=str,nargs='+',default=None)
    p.add_argument('--labels',type=str,nargs='+',default=None)
    p.add_argument('--defense_file',type=str,required=True)
    p.add_argument('--cluster_k',type=int,default=8)
    p.add_argument('--openai_key',type=str,default=None)
    p.add_argument('--embedding_cache',type=str,default=None)
    p.add_argument('--top_k',type=int,default=5)
    p.add_argument('--save_path',type=str,default=None)
    p.add_argument('--plot',action='store_true')
    p.add_argument('--plot_dir',type=str,default='./plots')
    args=p.parse_args()
    if not args.openai_key:
        try:
            sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..','PromptFuzz')))
            from utils import constants; args.openai_key=constants.openai_key
        except: pass
    if not args.openai_key: print("ERROR: --openai_key required"); sys.exit(1)
    main(args)
