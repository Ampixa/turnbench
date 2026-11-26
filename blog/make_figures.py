"""Render aggregate evidence from data/results.json; never read benchmark media."""
from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
D=json.loads((ROOT/'data/results.json').read_text());P=D['palette']
def v(k):return D['metrics'][k]['value']
def font(n,mono=False):return ImageFont.truetype(str(ROOT/'fonts'/('ibm-plex-mono.ttf' if mono else 'cmu-serif.ttf')),n)
def new(h=900,w=1600):
 im=Image.new('RGB',(w,h),P['paper']);return im,ImageDraw.Draw(im)
def txt(d,xy,t,n=36,color='ink',mono=False):d.text(xy,str(t),font=font(n,mono),fill=P.get(color,color))
def save(im,name):im.save(ROOT/name,quality=86,subsampling=0)
def title(d,t,sub):txt(d,(88,55),t,61);txt(d,(90,135),sub,26,'others',True)
def footer(d,t,y=832):txt(d,(90,y),t,25,'others',True)
for name,w,h in [('hero-16x9.jpg',1600,900),('og-1200x630.jpg',1200,630)]:
 im,d=new(h,w);s=w/1600
 def t(x,y,text,n=36,c='ink',m=False):txt(d,(int(x*s),int(y*s)),text,int(n*s),c,m)
 t(88,55,'How far can timing rules go?',65)
 t(90,143,'TURNBENCH / HELD-OUT DEV / NO NEW WEIGHTS',26,'others',True)
 for x,track in [(90,'eot'),(850,'int')]:
  t(x,240,track.upper()+' recall',36)
  t(x,290,f'{v(track+"_held_recall_mean"):.3f}',132,'ours',True)
  t(x,473,f'FPR {v(track+"_held_fpr_mean"):.3f}',34,'ink',True)
  t(x,537,f'Full dev {v(track+"_full_recall"):.3f}',29,'others',True)
 t(90,666,'A wide scoring window rewards late certainty.',42)
 t(90,742,f'Median delay: {v("eot_latency_ms"):.0f} / {v("int_latency_ms"):.0f} ms (EOT / INT)',29,'ours',True)
 t(90,820,'Dev estimates; no measured hidden-test ranking.',25,'others',True)
 save(im,name)
# Figure 1: explicitly a constructed mechanism diagram, not synthetic chart data.
im,d=new();title(d,'A second stamp can be free','SYNTHETIC TIMING DIAGRAM / NOT A DEV EXAMPLE')
T=D['diagram'];x0=440;x1=1460
X=lambda z:x0+(x1-x0)*z/T['window_end_s']
for label,y in [('After-anchor window',250),('Partner activity',410),('Emissions',570)]:txt(d,(88,y),label,30)
d.rectangle((x0,225,x1,320),fill=P['tint'])
d.line((x0,225,x0,650),fill=P['ink'],width=3);d.line((x1,225,x1,650),fill=P['others'],width=3)
d.rectangle((X(T['partner_onset_s']),405,x1,451),fill=P['others'])
for z,lab in [(T['first_emission_s'],'first'),(T['confirmation_s'],'confirmation')]:
 x=X(z);d.line((x,510,x,630),fill=P['ours'],width=5);d.ellipse((x-12,555,x+12,579),fill=P['ours'])
 txt(d,(x-65,470),lab,28,'ours');txt(d,(x-65,642),f'{z:g} s',26,'ours',True)
txt(d,(x0-10,335),'anchor',28);txt(d,(x1-70,335),f'+{T["window_end_s"]:g} s',28,mono=True)
txt(d,(88,735),'First emission matches; confirmation stays inside its positive window.',35)
footer(d,'Policy-derived illustration. Latency changes; recall need not.');save(im,'fig-timing.jpg')
# Figure 2: matched removal and re-selection, same budget and labelled FPRs.
im,d=new();title(d,'Confirmation is part of the result',f'EOT / HELD-OUT DEV / A-SELECTION FPR BUDGET {v("ablation_budget"):.2f}')
rows=[('Full policy','eot_ablation_baseline','ours'),('Confirmation removed','eot_ablation_no_confirm_matched','others'),('Removed, then retuned','eot_ablation_no_confirm_retuned','others'),('Partner evidence removed, retuned','eot_ablation_no_partner_retuned','others')]
lo=.90;hi=1.0;left=750;right=1320
for tick in [.90,.95,1.0]:
 x=left+(tick-lo)/(hi-lo)*(right-left);d.line((x,240,x,675),fill=P['pale'],width=2);txt(d,(x-35,701),f'{tick:.2f}',26,'others',True)
for i,(label,key,c) in enumerate(rows):
 y=260+i*110;txt(d,(88,y-20),label,31)
 x=left+(v(key+'_recall_mean')-lo)/(hi-lo)*(right-left)
 d.line((left,y,x,y),fill=P[c],width=3);d.ellipse((x-11,y-11,x+11,y+11),fill=P[c])
 txt(d,(1360,y-28),f'{v(key+"_recall_mean"):.3f}',31,c,True)
 txt(d,(1360,y+15),f'{v(key+"_fpr_mean"):.3f}',23,'others',True)
txt(d,(1360,195),'recall / FPR',22,'others',True)
footer(d,f'Matched removal: {100*v("confirmation_matched_delta"):.3f} recall points.'.replace("-", "−"));save(im,'fig-confirmation.jpg')
# Figure 3: selection variance; do not portray the oracle as held-out selection.
im,d=new();title(d,'The selector loses recall','INT / EARLIER ENERGY RULE / SAME SPLIT-HALVES')
for x,key,label,c in [(90,'int_selected','Selected on A, scored on B','ours'),(860,'int_oracle','Full-dev oracle, scored on B','others')]:
 txt(d,(x,260),f'{v(key):.4f}',112,c,True);txt(d,(x,430),label,33,c)
 txt(d,(x,495),'Eligible held-out selection' if key=='int_selected' else 'Uses B labels: diagnostic only',27,'others',True)
txt(d,(90,660),f'{100*v("int_selection_gap"):.3f} recall points separate these estimates.',43)
footer(d,'This diagnoses selection variance; it is not a hidden-test gain.');save(im,'fig-selection.jpg')
print('Rendered result card, social card and three figures from data/results.json.')
