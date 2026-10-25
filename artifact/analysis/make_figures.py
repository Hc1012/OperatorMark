"""Figures from FINAL_ANALYSIS.json only; no scientific computation reruns."""
import json,pathlib,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
HERE=pathlib.Path(__file__).resolve().parent
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=pathlib.Path,required=True)
OUT=parser.parse_args().output.resolve()
A=json.loads((OUT/'FINAL_ANALYSIS.json').read_text());D=json.loads((HERE/'inputs/MODEL_ANALYSIS_INPUTS.json').read_text())
F=OUT/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none'})
navy='#172d43';teal='#16877d';red='#bc4749';amber='#d79c39';gray='#d7dce1'
def save(fig,name):
    for ext in ['png','svg','pdf']:fig.savefig(F/(name+'.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10.8,5.7),sharey=True)
for ax,(k,title) in zip(axes,[('random_jvp','Random-direction JVP error'),('task','Inverse-task error')]):
    x=A['primary']['advantage'][k];ys=np.arange(10,0,-1)
    ax.axvspan(.8,1.25,color=gray,alpha=.6);ax.axvline(1,color='#87929e',lw=1);ax.axvline(1.25,color='#87929e',lw=1,ls='--')
    ax.scatter(x['ratios'],ys,color=navy,s=36,zorder=3)
    ax.errorbar(x['geometric_mean'],-.25,xerr=[[x['geometric_mean']-x['lower']],[x['upper']-x['geometric_mean']]],fmt='D',color=teal,capsize=5,lw=2,ms=7)
    ax.set_xscale('log');ax.set_xlim(.7,11);ax.set_xticks([1,2,4,8]);ax.set_xticklabels(['1','2','4','8']);ax.set_title(title,loc='left',fontweight='bold',pad=12)
    ax.set_xlabel('Replacement error / attacked-copy error');ax.grid(axis='x',alpha=.14)
    ax.text(.02,-.18,f"GM {x['geometric_mean']:.2f} · 97.5% CI [{x['lower']:.2f}, {x['upper']:.2f}]",transform=ax.transAxes,color=teal)
axes[0].set_yticks(list(range(10,0,-1))+[-.25]);axes[0].set_yticklabels(A['qualification_roster']+['Geometric mean'])
fig.suptitle('Copying retains a measured advantage over the selected replacement',x=.08,ha='left',fontweight='bold',fontsize=15,y=1.02)
fig.text(.08,-.03,'All 10 planned owners; frozen ordinary P2 comparison. The ordinary attack fails the full utility gate.\nGrey band: [0.8, 1.25]. Intervals use owner log ratios (df = 9); tasks and keys are not extra owners.',fontsize=9,color='#45566a')
fig.tight_layout();save(fig,'01_primary_copy_advantage')

co=A['cohorts']['cohorts'];labels=list(co);matrix=[];texts=[]
for group in labels:
    v=co[group]['primary'];row=[];tx=[]
    for o in A['qualification_roster']:
        r=A['model_readouts'][o+'/'+v]
        val=0 if not r['same_test_intact_prerequisite'] else 1 if r['qualified_removal'] else 2 if r['retained_detection'] else 3
        row.append(val);tx.append(['—','R','S','U'][val])
    matrix.append(row);texts.append(tx)
fig,ax=plt.subplots(figsize=(10,12.8));cmap=ListedColormap([gray,red,teal,'#f1d6a2'])
ax.imshow(matrix,cmap=cmap,norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4),aspect='auto')
for i,row in enumerate(texts):
    for j,t in enumerate(row):ax.text(j,i,t,ha='center',va='center',fontsize=8,color='white' if matrix[i][j] in [1,2] else navy)
ax.set_xticks(range(10));ax.set_xticklabels(A['qualification_roster']);ax.xaxis.tick_top()
ax.set_yticks(range(len(labels)));ax.set_yticklabels([co[x]['primary'].replace('_M3','') for x in labels],fontsize=9)
ax.set_title('Active-mark outcomes at the prescribed endpoints',loc='left',fontweight='bold',fontsize=15,pad=35)
ax.set_xticks(np.arange(-.5,10,1),minor=True);ax.set_yticks(np.arange(-.5,len(labels),1),minor=True);ax.grid(which='minor',color='white',lw=1);ax.tick_params(which='minor',length=0)
fig.legend(handles=[Patch(color=red,label='R: qualified removal'),Patch(color=teal,label='S: detected + utility retained'),Patch(color='#f1d6a2',label='U: utility fails'),Patch(color=gray,label='Intact prerequisite absent')],loc='lower center',bbox_to_anchor=(.62,.035),ncol=2,frameon=False,fontsize=9)
fig.text(.08,.002,'O05 remains in the locked ten-owner roster but cannot count as attack-induced removal.\nIntermediate checkpoints also enter cohort survival rules; this figure shows fixed terminal endpoints only.',fontsize=9)
fig.subplots_adjust(left=.33,right=.98,top=.94,bottom=.105);save(fig,'02_active_mark_outcomes')

names=['M3','Aord_M3_raw1920','A4_M3_cooled20480','A5B3_M3_cooled20480','W256_M3','W256_Aord_M3_raw1920','mix25_M3','mix50_M3','overwrite_M3','storagefp16_M3']
pretty=['Intact mark','Ordinary · 1,920','A4 · 20,480','A5-B3 · 20,480','Jacobi256 on intact','Jacobi256 after ordinary','Mix 25%','Mix 50%','Overwrite','Storage fp16']
fig,ax=plt.subplots(figsize=(10,6.6));yy=np.arange(len(names));w=.32
active=[A['variants'][v]['qualified_removal'] or 0 for v in names]
passive=[A['B2'][v]['positive_and_utility'] for v in names]
ax.barh(yy-w/2,active,w,color=red,label='Active mark: qualified removal')
ax.barh(yy+w/2,passive,w,color=teal,label='Passive fingerprint: positive + utility')
ax.set_yticks(yy);ax.set_yticklabels(pretty);ax.invert_yaxis();ax.set_xlim(0,10.7);ax.set_xticks(range(0,11,2));ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
for y,v in zip(yy,active):ax.text(v+.1,y-w/2,str(v),va='center',fontsize=9)
for y,v in zip(yy,passive):ax.text(v+.1,y+w/2,str(v),va='center',fontsize=9)
ax.set_xlabel('Owners / 10 planned');ax.set_title('Active-mark removal can coexist with passive lineage evidence',loc='left',fontsize=15,fontweight='bold',pad=15)
ax.legend(loc='lower center',bbox_to_anchor=(.5,-.24),ncol=1,frameon=False)
fig.text(.08,-.005,'Different decisions: B3 removal needs a detected intact parent; B2 positivity is relative to the fixed 9 + 19 comparators.\nSelected illustrative arms; all 124 B2 variants and 36 active cohorts are retained in the tables. No population FPR claim.',fontsize=9)
fig.tight_layout();save(fig,'03_active_and_passive')

fig,ax=plt.subplots(figsize=(8.7,5.2))
variants=['O','M3','Aord_M3_raw1920','A4_M3_cooled20480','Rstar']
for v,lab,c in zip(variants,['Clean owners','Intact marked','Ordinary copies','A4 copies','Scratch replacements'],[navy,'#6982a0',red,teal,amber]):
    vals=[D[o+'/'+v]['metrics']['T'] for o in A['qualification_roster']]
    x=variants.index(v);ax.scatter([x]*10,vals,color=c,s=28,alpha=.8);ax.plot([x-.18,x+.18],[np.median(vals)]*2,color=c,lw=3,label=lab)
for i,k in enumerate(['F64','P256'],5):ax.scatter(i,A['native'][k]['means']['T'],marker='D',color='#6c517d',s=70)
ax.set_xticks(range(7));ax.set_xticklabels(['Clean','Intact\nmarked','Ordinary\ncopy','A4\ncopy','Scratch\nR*','Native\nF64','Native\nP256'])
ax.set_ylabel('Mean T128 inverse-task error');ax.set_ylim(0,.039);ax.grid(axis='y',alpha=.15)
ax.set_title('Task performance against the declared discrete reference',loc='left',fontweight='bold',fontsize=14,pad=15)
fig.text(.10,-.015,'Circles: 10 owner-level means; bars: medians. Diamonds: one shared 64-target solver procedure each.\nNative solvers are not 10 new owners; these results do not establish physical or continuum accuracy.',fontsize=9)
fig.tight_layout();save(fig,'04_discrete_task_context')
print('Saved four figures in PNG, SVG and PDF.')
