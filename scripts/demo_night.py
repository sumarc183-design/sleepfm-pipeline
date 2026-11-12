"""Page HTML autonome d'une nuit de test : stades, scores et événements respiratoires."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cache_contract import file_sha256
from src.data.subject_ids import validate_subject_ids


def night_data(
    stages: Path,
    subject: str | None,
    respiratory: Path | None,
    labels: Path | None,
    comparison: Path | None = None,
) -> dict:
    with np.load(stages) as data:
        subjects = sorted(set(data["subject"].tolist()))
        subject = subject or subjects[0]
        validate_subject_ids([subject])
        rows = np.flatnonzero(data["subject"] == subject)
        if not len(rows):
            raise ValueError("Nuit absente de l'archive de test")
        rows = rows[np.argsort(data["onset_s"][rows])]
        result = {
            "subject": subject,
            "onset_s": data["onset_s"][rows].tolist(),
            "epoch_idx": data["epoch_idx"][rows].tolist(),
            "reference": data["y_true"][rows].tolist(),
            "probes": {},
            "events": [],
            "respiratory": None,
            "sources": {"stages_sha256": file_sha256(stages)},
        }
        for probe in ("logistic", "sequence"):
            probability = data[f"proba_{probe}"][rows]
            classes = data[f"classes_{probe}"]
            if not np.isfinite(probability).all() or probability.shape != (len(rows), 5):
                raise ValueError("Probabilités de stade invalides")
            result["probes"][probe] = {
                "pred": data[f"pred_{probe}"][rows].tolist(),
                "probability": probability.tolist(),
                "classes": classes.tolist(),
            }
    if not np.isfinite(result["onset_s"]).all():
        raise ValueError("Temps d'origine absents")
    if respiratory is not None:
        report = json.loads(respiratory.read_text())
        if report.get("status") != "terminé" or subject not in report["partitions"]["test"]:
            raise ValueError("Nuit absente du test respiratoire terminé")
        archive = respiratory.with_suffix(".predictions.npz")
        if file_sha256(archive) != report["predictions_sha256"]:
            raise ValueError("Archive respiratoire différente de son empreinte")
        with np.load(archive) as data:
            lookup = {int(data["epoch_idx"][i]): i for i in np.flatnonzero(data["subject"] == subject)}
            probabilities, targets = [], []
            for index, onset in zip(result["epoch_idx"], result["onset_s"], strict=True):
                i = lookup.get(index)
                if i is not None and not np.isclose(data["onset_s"][i], onset, atol=0.001, rtol=0):
                    raise ValueError("Fenêtres des deux tâches non alignées")
                probabilities.append(None if i is None else float(data["probability"][i]))
                targets.append(None if i is None else int(data["y_true"][i]))
        result["respiratory"] = {
            "probability": probabilities,
            "target": targets,
            "threshold": report["threshold"]["value"],
        }
        result["sources"]["respiratory_sha256"] = file_sha256(respiratory)
        if labels is not None:
            label_manifest = json.loads((labels / "manifest.json").read_text())
            if file_sha256(labels / "manifest.json") != report["label_manifest_sha256"]:
                raise ValueError("Manifeste des labels différent du run")
            label_path = labels / f"{subject}.npz"
            if file_sha256(label_path) != label_manifest["sources"][subject]["labels_sha256"]:
                raise ValueError("Labels de la nuit modifiés")
            with np.load(label_path) as target:
                result["events"] = target["intervals_s"].tolist()
    if comparison is not None:
        if respiratory is None:
            raise ValueError("La comparaison exige la baseline respiratoire de référence")
        report = json.loads(comparison.read_text())
        if (
            report.get("status") != "terminé"
            or subject not in report["partitions"]["test"]
            or report["baseline_sha256"] != file_sha256(respiratory)
        ):
            raise ValueError("Comparaison incomplète ou baseline différente")
        archive = comparison.with_suffix(".predictions.npz")
        if file_sha256(archive) != report["artifacts"]["predictions.npz"]["sha256"]:
            raise ValueError("Archive comparative différente de son empreinte")
        with np.load(archive) as data:
            lookup = {int(data["epoch_idx"][i]): i for i in np.flatnonzero(data["subject"] == subject)}
            osf = []
            for j, (index, onset) in enumerate(zip(result["epoch_idx"], result["onset_s"], strict=True)):
                i = lookup.get(index)
                if (i is None) != (result["respiratory"]["probability"][j] is None):
                    raise ValueError("Couvertures respiratoires différentes")
                if i is not None:
                    if not np.isclose(data["onset_s"][i], onset, atol=0.001, rtol=0):
                        raise ValueError("Fenêtres comparatives non alignées")
                    if not np.isclose(
                        data["probability_sleepfm"][i], result["respiratory"]["probability"][j], atol=1e-7, rtol=0
                    ):
                        raise ValueError("Scores SleepFM différents")
                osf.append(None if i is None else float(data["probability_osf"][i]))
        result["respiratory"]["osf_probability"] = osf
        result["respiratory"]["osf_threshold"] = report["thresholds"]["osf"]
        result["sources"]["comparison_sha256"] = file_sha256(comparison)
    return result


def render(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    title = html.escape(data["subject"])
    return PAGE.replace("__TITLE__", title).replace("__DATA__", payload)


PAGE = """<!doctype html>
<html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nuit __TITLE__ — SleepFM</title>
<style>
body{font:16px system-ui,sans-serif;background:#f5f7fb;color:#192c42;margin:0}
main{max-width:1180px;margin:auto;padding:28px}h1{margin:0 0 8px}h2{font-size:19px}
section,.controls{background:white;border:1px solid #dce3ec;border-radius:12px;padding:18px;margin:16px 0}
svg{width:100%;height:auto;display:block}label{display:inline-block;margin:8px 20px 8px 0}
input[type=range]{width:min(600px,75vw)}.muted{color:#526579}.blue{color:#2563eb}.orange{color:#ba5a0a}
#detail{min-height:48px;background:#edf2f9;padding:12px;border-radius:8px}summary{cursor:pointer}
</style><main>
<h1>Une nuit, deux tâches</h1><p>__TITLE__ · PhysioNet 2018 · exploration de recherche</p>
<p class="muted">Fenêtres annotées de 30 secondes. Les données viennent d'archives de test ;
aucun diagnostic ni IAH n'est calculé. Les scores ne sont pas des risques cliniques calibrés.</p>
<div class="controls">
<label>Étendue <select id="span"><option value="60">1 heure</option><option value="30">30 minutes</option>
<option value="0">Nuit entière</option></select></label>
<label>Stade <select id="stage"><option value="0">Éveil</option><option value="1">N1</option>
<option value="2">N2</option><option value="3">N3</option><option value="4">REM</option></select></label>
<p><label for="offset">Début de la vue</label><input id="offset" type="range" min="0" step="1" value="0">
<output id="time"></output></p>
</div>
<section><h2>Hypnogramme</h2><p>Expert : gris · <span class="blue">logistique : bleu</span> ·
<span class="orange">BiGRU : orange</span>. Les bandes roses signalent un désaccord entre les deux sondes.</p>
<svg id="hypno" viewBox="0 0 1100 220" role="img" aria-label="Hypnogramme comparé"></svg></section>
<section><h2>Score du stade sélectionné</h2>
<svg id="scores" viewBox="0 0 1100 180" role="img" aria-label="Probabilités des deux sondes"></svg></section>
<section><h2>Apnées et hypopnées</h2><p id="resp-note"></p>
<svg id="resp" viewBox="0 0 1100 180" role="img" aria-label="Scores respiratoires et annotations"></svg></section>
<p id="detail" aria-live="polite">Clique sur un graphique pour consulter une fenêtre.</p>
<details><summary>Limites et provenance</summary>
<p>Le BiGRU utilise les époques futures : traitement après enregistrement. Les intervalles respiratoires
annotés sont grisés. Cible de la baseline : au moins 10 s d'apnée ou d'hypopnée dans la fenêtre ;
couverture entre 0 et 10 s exclue, donc score respiratoire absent ici. La page n'affiche pas de signal brut.
La nuit est choisie par identifiant, sans sélection du meilleur résultat.</p><pre id="sources"></pre></details>
</main><script id="night-data" type="application/json">__DATA__</script>
<script>
'use strict';
const D=JSON.parse(document.getElementById('night-data').textContent);
const names=['Éveil','N1','N2','N3','REM'], positions=[0,2,3,4,1], colors={logistic:'#2563eb',sequence:'#ba5a0a'};
const $=id=>document.getElementById(id), min=D.onset_s[0], max=D.onset_s[D.onset_s.length-1]+30;
let left=min,right=max;
function svg(tag,attrs,parent){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);
for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);parent.appendChild(e);return e;}
function text(parent,x,y,s){svg('text',{x,y,fill:'#526579','font-size':12},parent).textContent=s;}
function x(t){return 65+(t-left)/(right-left)*1015;}
function axis(id,height,stage){const el=$(id);el.replaceChildren();
for(let i=0;i<=4;i++){const y=20+i*(height-50)/4;
svg('line',{x1:65,x2:1080,y1:y,y2:y,stroke:'#e3e9f1'},el);
text(el,6,y+4,stage?['Éveil','REM','N1','N2','N3'][i]:(1-i/4).toFixed(2));}
for(let i=0;i<=6;i++){const t=left+(right-left)*i/6;text(el,x(t)-12,height-7,(t/3600).toFixed(1)+' h');}
return el;}
function line(el,values,height,stage,color){let path='',last=-Infinity;
for(let i=0;i<values.length;i++){const t=D.onset_s[i],v=values[i];
if(t+30<left||t>right||v===null){last=-Infinity;continue;}
const y=20+(stage?positions[v]/4:1-v)*(height-50),xx=x(Math.max(t,left));
path+=(t-last>31?'M':'L')+xx+','+y+'L'+x(Math.min(t+30,right))+','+y;last=t;}
svg('path',{d:path,fill:'none',stroke:color,'stroke-width':2},el);}
function draw(){const minutes=Number($('span').value),width=minutes?minutes*60:max-min;
$('offset').max=Math.max(0,Math.floor((max-min-width)/60));$('offset').disabled=!minutes;
left=min+Number($('offset').value)*60;right=Math.min(max,left+width);
$('time').textContent=(left/3600).toFixed(2)+' — '+(right/3600).toFixed(2)+' h';
const h=axis('hypno',220,true),p=axis('scores',180,false),r=axis('resp',180,false);
for(let i=0;i<D.onset_s.length;i++){const t=D.onset_s[i];if(t+30<left||t>right)continue;
if(D.probes.logistic.pred[i]!==D.probes.sequence.pred[i])
svg('rect',{x:x(Math.max(t,left)),y:14,width:Math.max(0,x(Math.min(t+30,right))-x(Math.max(t,left))),
height:165,fill:'#ffe5e5'},h);}
line(h,D.reference,220,true,'#8893a1');
for(const kind of ['logistic','sequence']){line(h,D.probes[kind].pred,220,true,colors[kind]);
const c=D.probes[kind].classes.indexOf(Number($('stage').value));
line(p,D.probes[kind].probability.map(row=>row[c]),180,false,colors[kind]);}
if(D.respiratory){for(const [a,b] of D.events){if(b<=left||a>=right)continue;
svg('rect',{x:x(Math.max(a,left)),y:20,width:x(Math.min(b,right))-x(Math.max(a,left)),height:130,fill:'#dce2e9'},r);}
const yy=20+(1-D.respiratory.threshold)*130;
svg('line',{x1:65,x2:1080,y1:yy,y2:yy,stroke:'#d15454','stroke-dasharray':'6 4'},r);
line(r,D.respiratory.probability,180,false,'#087d70');}
if(D.respiratory?.osf_probability){
const yy=20+(1-D.respiratory.osf_threshold)*130;
svg('line',{x1:65,x2:1080,y1:yy,y2:yy,stroke:'#8b40cc','stroke-dasharray':'3 4'},r);
line(r,D.respiratory.osf_probability,180,false,'#8b40cc');}
$('resp-note').textContent=D.respiratory?.osf_probability?
'SleepFM en vert, OSF en violet ; seuils de validation en pointillés ; annotations grisées.':D.respiratory?
'Score en vert ; seuil de validation en rouge ; événements annotés grisés.':
'Aucune baseline respiratoire fournie.';}
for(const id of ['span','stage','offset'])$(id).addEventListener('input',draw);
for(const id of ['hypno','scores','resp'])$(id).addEventListener('click',event=>{
const rect=$(id).getBoundingClientRect(),pixel=(event.clientX-rect.left)/rect.width*1100;
const t=left+(pixel-65)/1015*(right-left);let best=0;
for(let i=1;i<D.onset_s.length;i++)if(Math.abs(D.onset_s[i]-t)<Math.abs(D.onset_s[best]-t))best=i;
const resp=D.respiratory?.probability[best];
const osf=D.respiratory?.osf_probability?.[best];
$('detail').textContent='Fenêtre '+D.epoch_idx[best]+' · '+(D.onset_s[best]/3600).toFixed(3)+' h · Expert : '+
names[D.reference[best]]+' · Logistique : '+names[D.probes.logistic.pred[best]]+' · BiGRU : '+
names[D.probes.sequence.pred[best]]+(resp===null||resp===undefined?
' · score respiratoire absent':' · score SleepFM : '+resp.toFixed(3))+
(osf===null||osf===undefined?'':' · score OSF : '+osf.toFixed(3));});
$('sources').textContent=JSON.stringify(D.sources,null,2);draw();
</script></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", type=Path, default=Path("results/results_physionet2018_500.predictions.npz"))
    parser.add_argument("--subject", help="Par défaut : premier identifiant de l'archive de test")
    parser.add_argument("--respiratory", type=Path, help="JSON du run respiratoire terminé")
    parser.add_argument("--labels", type=Path, help="Labels respiratoires préparés, pour les intervalles annotés")
    parser.add_argument("--comparison", type=Path, help="JSON de comparaison OSF/SleepFM respiratoire terminée")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("La sortie existe déjà ; choisir un nouveau nom")
    if args.out.resolve() in {p.resolve() for p in (args.stages, args.respiratory, args.comparison) if p is not None}:
        parser.error("La sortie doit être différente des sources")
    data = night_data(args.stages, args.subject, args.respiratory, args.labels, args.comparison)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(data), encoding="utf-8")
    print(f"{data['subject']} : {args.out}")


if __name__ == "__main__":
    main()
