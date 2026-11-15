"""Scientific figure for the audited numeric-argument comparison."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest
from scripts import modal_numeric_followup as study


def main():
    root=study.ROOT;r=study.read(root/'evaluation/results.json');audit=study.read(root/'completion_audit.json')
    assert audit['passed'] and audit['results_sha256']==digest(root/'evaluation/results.json')
    pairs=[('neural_numeric','neural_blank','History transformer · 320 questions'),
           ('numeric320','matched_duration320','Tree · 320 questions'),('numeric512','matched_duration512','Tree · 512 questions')]
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,6.4),gridspec_kw={'width_ratios':[1.25,1]})
    fig.subplots_adjust(left=.28,right=.97,bottom=.23,top=.78,wspace=.28)
    for i,(new,old,label) in enumerate(pairs):
        y=2-i;a=100*r['summary'][old]['validation']['accuracy'];b=100*r['summary'][new]['validation']['accuracy']
        axes[0].plot([a,b],[y,y],color='#9da7ac',lw=1.5)
        axes[0].scatter(a,y-.07,color='#7c8791',s=48,label='Matched control' if i==0 else None)
        axes[0].scatter(b,y+.07,color='#196c87',s=52,marker='s',label='Add numeric argument features' if i==0 else None)
        axes[0].text(a-.07,y-.21,f'{a:.2f}%',ha='right',color='#56616b',fontsize=10)
        axes[0].text(b+.07,y+.18,f'{b:.2f}%',ha='left',color='#196c87',fontsize=10)
        c=next(z for z in r['contrasts'] if (z['new'],z['reference'],z['metric'])==(new,old,'error'))
        value=100*c['reduction'];lo,hi=100*np.asarray(c['ci95'])
        axes[1].errorbar(value,y,xerr=[[value-lo],[hi-value]],fmt='s',color='#196c87',capsize=4)
        axes[1].text(value,y+.21,f'{value:+.3f} pp',ha='center',fontsize=10)
    minimum=min(100*r['summary'][name]['validation']['accuracy'] for pair in pairs for name in pair[:2])
    axes[0].set(xlim=(min(87.5,minimum-.6),95.5),ylim=(-.5,2.6),yticks=[2,1,0],yticklabels=[z[2] for z in pairs],xlabel='Validation accuracy (%)')
    axes[0].axvline(95,ls='--',color='#b15745',lw=1);axes[0].text(95.08,2.35,'95% target',rotation=90,color='#b15745',va='top',fontsize=9)
    axes[1].axvline(0,color='#88949b',lw=1)
    axes[1].set(ylim=axes[0].get_ylim(),yticks=[],xlabel='Gain from numeric features (percentage points)',title='Paired 95% question intervals')
    for ax in axes:ax.grid(axis='x',alpha=.15)
    fig.suptitle('Does explicit numeric argument information transfer?',x=.59,y=.95,fontsize=16)
    fig.text(.59,.88,'All updates on the same 80 validation questions · models frozen before scoring',ha='center',fontsize=11)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.60,.10),ncol=2,frameon=False,fontsize=10)
    fig.text(.28,.035,'Neural scores average three individual seeds; both arms have 2,370 learned parameters.\nTrees share 155 iterations. No answer keys are inputs. Intervals are descriptive on reused validation.',fontsize=9,color='#4a5963')
    fig.savefig('reports/modal_numeric_followup_320.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
