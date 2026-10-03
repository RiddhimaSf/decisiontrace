"""The decision screen. Rendered on the server as plain HTML/CSS (tables, no modern layout features) with a few
lines of ES5 JavaScript for the 'ask the simulation' panel, so it works in a browser, offline as a saved file,
and in old rendering engines. The same renderer produces the live page and the static demo."""
from __future__ import annotations

import html
import json

from . import config as C
from .analysis import fmt

STATE_STYLE = {
    "ROBUST": ("#2E8B57", "ROBUST"), "MODERATE": ("#C98A00", "MODERATE"), "LOW": ("#D9622B", "FRAGILE"),
    "CONTESTED": ("#7B4FA3", "CONTESTED"), "UNSTABLE": ("#C0392B", "UNSTABLE"),
    "INSUFFICIENT_EVIDENCE": ("#C0392B", "INSUFFICIENT EVIDENCE"), "STALE": ("#6B7280", "OUT OF DATE"),
}
PRESETS = ["Which option performed best?", "Which segment contributed most to B's advantage?", "Did enterprise buyers prefer B?",
           "Why did enterprise buyers prefer B?", "Why did SMB buyers prefer B?", "What would have to change for the recommendation to flip?",
           "Is B robust to changes in price sensitivity?", "Did real customers prefer B?", "What will B do to revenue next year?",
           "What assumptions does this depend on?"]

CSS = """
body{margin:0;background:#F4F6FA;font-family:Helvetica,Arial,sans-serif;color:#1F2A44;font-size:13px}
.wrap{width:1080px;margin:0 auto;padding:14px 0 30px}
h1{font-size:12px;letter-spacing:1px;color:#5B657A;margin:0 0 2px;font-weight:bold}
.q{font-size:20px;font-weight:bold;margin:0 0 10px}
.card{background:#fff;border:1px solid #DDE2EC;border-radius:6px;padding:12px 14px;margin-bottom:10px}
.card h2{font-size:11px;letter-spacing:.8px;color:#1B7F79;margin:0 0 8px;text-transform:uppercase}
.banner{border-radius:6px;padding:12px 16px;color:#fff;margin-bottom:10px}
.banner .tag{font-size:11px;letter-spacing:1px;font-weight:bold;opacity:.95}
.banner .head{font-size:19px;font-weight:bold;margin-top:3px;line-height:1.25}
.banner .sub{font-size:12px;margin-top:5px;opacity:.95}
table.l{border-collapse:separate;border-spacing:0;width:100%}
td.col{vertical-align:top}
.barrow{margin:6px 0}
.barlab{display:inline-block;width:150px;vertical-align:middle}
.bartrack{display:inline-block;width:300px;height:15px;background:#EEF1F6;position:relative;vertical-align:middle}
.barfill{height:15px;background:#1B7F79}
.barfill.win{background:#1F2A44}
.barval{display:inline-block;margin-left:8px;vertical-align:middle;font-weight:bold}
.muted{color:#5B657A}.small{font-size:11px}
.pill{display:inline-block;border-radius:10px;padding:1px 8px;font-size:10px;font-weight:bold;color:#fff}
.meter{width:100%;height:10px;background:#EEF1F6;position:relative;margin:6px 0}
.meter div{height:10px}
.div-track{display:inline-block;width:300px;height:15px;background:#EEF1F6;position:relative;vertical-align:middle}
.div-mid{position:absolute;left:150px;top:0;width:1px;height:15px;background:#1F2A44}
.warn{background:#FFF4E5;border-left:3px solid #D9622B;padding:7px 10px;margin:6px 0;font-size:12px}
.ok{color:#2E8B57;font-weight:bold}.bad{color:#C0392B;font-weight:bold}.amb{color:#C98A00;font-weight:bold}
ul.f{margin:0;padding-left:16px}ul.f li{margin:4px 0;line-height:1.35}
.cite{display:inline-block;background:#E4F2F1;color:#145E59;border-radius:3px;padding:0 4px;font-size:10px;cursor:pointer;margin:0 1px}
.ans{background:#F8FAFD;border:1px solid #DDE2EC;border-radius:5px;padding:9px 11px;margin-top:8px;line-height:1.5}
.prov{background:#fff;border:1px dashed #9AA3B2;padding:6px 9px;margin-top:6px;font-size:11px;display:none}
.chip{display:inline-block;padding:0 5px;border-radius:3px;font-size:9px;font-weight:bold;color:#fff;margin-right:3px}
.c-INPUT{background:#5B657A}.c-GENERATED{background:#D9622B}.c-DERIVED{background:#1B7F79}.c-OBSERVED{background:#2E8B57}
input.askin,select.askin{width:700px;padding:6px;font-size:13px;border:1px solid #9AA3B2;border-radius:4px}
button.askbtn{padding:6px 14px;background:#1F2A44;color:#fff;border:0;border-radius:4px;font-size:13px;cursor:pointer}
.scn a{display:inline-block;margin:2px 4px 2px 0;padding:2px 8px;border-radius:10px;background:#E8ECF4;color:#1F2A44;text-decoration:none;font-size:11px}
.scn a.cur{background:#1F2A44;color:#fff}
"""

JS = """
function dtShow(name){var els=document.getElementsByTagName('div');for(var i=0;i<els.length;i++){if(els[i].className.indexOf('scenario')===0){els[i].style.display=(els[i].id==='s-'+name)?'block':'none';}}
 var as=document.getElementsByTagName('a');for(var j=0;j<as.length;j++){if(as[j].getAttribute('data-sc')){as[j].className=(as[j].getAttribute('data-sc')===name)?'cur':'';}}}
function dtPreset(sel,name){var opts=sel.options;for(var i=0;i<opts.length;i++){var el=document.getElementById('a-'+name+'-'+i);if(el){el.style.display=(i===sel.selectedIndex)?'block':'none';}}}
function dtProv(id,name){var el=document.getElementById('p-'+name+'-'+id);if(el){el.style.display=(el.style.display==='block')?'none':'block';}}
function dtEsc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function dtAsk(name){var q=document.getElementById('askq-'+name).value;if(!q){return;}var out=document.getElementById('askout-'+name);out.innerHTML='<div class="muted small">Asking the ledger...</div>';
 var x=new XMLHttpRequest();x.open('POST','/api/ask',true);x.setRequestHeader('Content-Type','application/json');
 x.onreadystatechange=function(){if(x.readyState===4){if(x.status!==200){out.innerHTML='<div class="warn">Request failed.</div>';return;}var r=JSON.parse(x.responseText);
  var t=dtEsc(r.text).replace(/\\[(EV-\\d{4})\\]/g,function(m,id){return '<span class="cite" onclick="dtProvLive(\\''+id+'\\',\\''+name+'\\')">'+id+'</span>';});
  var g=[];for(var k in r.guardrails){if(r.guardrails.hasOwnProperty(k)&&k!=='intent'&&k!=='all'){g.push((r.guardrails[k]?'<span class="ok">&#10003;</span> ':'<span class="bad">&#10007;</span> ')+k);}}
  out.innerHTML='<div class="ans">'+t+'</div><div class="small muted" style="margin-top:5px">intent: '+dtEsc(r.intent)+' &middot; guardrails: '+g.join(' &middot; ')+'</div><div id="live-prov"></div>';}};
 x.send(JSON.stringify({scenario:name,question:q}));}
function dtProvLive(id,name){var x=new XMLHttpRequest();x.open('GET','/api/evidence/'+name+'/'+id,true);x.onreadystatechange=function(){if(x.readyState===4&&x.status===200){var r=JSON.parse(x.responseText);var h='<div class="prov" style="display:block"><span class="chip c-'+r.cls+'">'+r.cls+'</span><b>'+dtEsc(r.key)+'</b> = '+dtEsc(r.display)+'<br/><span class="muted">'+dtEsc(r.note||'')+'</span>';
 if(r.chain.length){h+='<br/>Provenance: ';for(var i=0;i<r.chain.length;i++){h+='&larr; <span class="chip c-'+r.chain[i].cls+'">'+r.chain[i].cls+'</span>'+dtEsc(r.chain[i].key)+' ';}}h+='</div>';document.getElementById('live-prov').innerHTML=h;}};x.send();}
"""


def e(x):
    return html.escape(str(x))


def evidence_display(row):
    v = row["value"]
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return fmt(v, row["unit"])
    if isinstance(v, bool):
        return "yes" if v else "NO"
    return json.dumps(v) if not isinstance(v, str) else v


def prov_block(exp, evid, name):
    L = exp.ledger
    r = L.by_id(evid)
    chain = L.chain(evid)
    ch = "".join(f'&larr; <span class="chip c-{c["cls"]}">{c["cls"]}</span>{e(c["key"])} ' for c in chain[:8])
    more = f' <span class="muted">(+{len(chain) - 8} more)</span>' if len(chain) > 8 else ""
    return (f'<div class="prov" id="p-{name}-{evid}"><span class="chip c-{r["cls"]}">{r["cls"]}</span><b>{e(r["key"])}</b> = {e(evidence_display(r))}'
            f'<br/><span class="muted">{e(r["note"])}</span>' + (f"<br/>Provenance: {ch}{more}" if chain else "<br/>Provenance: (root input)") + "</div>")


def cite_html(text, name):
    import re
    t = e(text)
    return re.sub(r"\[(EV-\d{4})\]", lambda m: f'<span class="cite" onclick="dtProv(\'{m.group(1)}\',\'{name}\')">{m.group(1)}</span>', t)


def _bar(label, value, maxv, win=False, lo=None, hi=None, ev=None, name=""):
    w = max(0.0, min(100.0, value / maxv * 100))
    band = f'<span class="muted small"> ({lo:.1f}&ndash;{hi:.1f})</span>' if lo is not None else ""
    return (f'<div class="barrow"><span class="barlab">{e(label)}</span><span class="bartrack"><div class="barfill{" win" if win else ""}" style="width:{w:.1f}%"></div></span>'
            f'<span class="barval">{value:.1f}%</span>{band}</div>')


def _div_bar(label, value, maxabs, n=None):
    w = min(150.0, abs(value) / maxabs * 150) if maxabs else 0
    if value >= 0:
        fill = f'<div style="position:absolute;left:150px;top:0;width:{w:.1f}px;height:15px;background:#1B7F79"></div>'
    else:
        fill = f'<div style="position:absolute;left:{150 - w:.1f}px;top:0;width:{w:.1f}px;height:15px;background:#D9622B"></div>'
    return (f'<div class="barrow"><span class="barlab">{e(label)}</span><span class="div-track">{fill}<span class="div-mid"></span></span>'
            f'<span class="barval">{value:+.1f}pp</span></div>')


def render_scenario(exp, name, qa=None, live=False, display="block"):
    d = exp.decision
    color, tag = STATE_STYLE[d["state"]]
    W, R = d["winner"], d["runner_up"]
    s = d["stability"]
    sub = ""
    if d["state"] in ("UNSTABLE",):
        sub = "Why: " + "; ".join(d["unstable_reasons"]) + "."
    elif d["state"] == "INSUFFICIENT_EVIDENCE":
        sub = "The figures below are shown for transparency only; they are not a basis for a decision."
    elif d["state"] == "STALE":
        sub = "Do not act on these figures until the simulation is re-run on current inputs."
    elif d["state"] in ("MODERATE", "LOW", "ROBUST", "CONTESTED"):
        sub = f"Decision stability: <b>{e(s['label'])}</b> &middot; {W} remains preferred under <b>{s['pct']:.0f}%</b> of {s['n_tested']} tested assumption sets &middot; leads in {d['win_rate_pct']:.0f}% of runs."
    banner = (f'<div class="banner" style="background:{color}"><div class="tag">{e(tag)} &middot; CONFIDENCE: {e(d["confidence"]).upper()}</div>'
              f'<div class="head">{e(d["headline"])}</div><div class="sub">{sub}</div></div>')
    if d["state"] in ("UNSTABLE", "INSUFFICIENT_EVIDENCE", "STALE"):
        shares = "".join(_bar(f"{x['option']} {x['label']}", x["mean"], 60, False, x["lo"], x["hi"]) for x in d["shares"])
        rank_note = '<div class="muted small">Shown in fixed order, not ranked. Bands are +/-1.96 sd across runs.</div>'
    else:
        shares = "".join(_bar(f"{x['option']} {x['label']}", x["mean"], 60, x["option"] == W, x["lo"], x["hi"]) for x in sorted(d["shares"], key=lambda x: -x["mean"]))
        rank_note = '<div class="muted small">Simulated preference. Bands are +/-1.96 sd across runs.</div>'
    left = f'<div class="card"><h2>Current result (simulated preference)</h2>{shares}{rank_note}</div>'

    segs = d["segments"]["company_size"] + d["segments"]["buyer_type"]
    maxabs = max([abs(r["contribution_pp"]) for r in segs if r["contribution_pp"] is not None] + [1])
    drv = ""
    for dim, title in (("company_size", "By company size"), ("buyer_type", "By buyer type")):
        drv += f'<div class="muted small" style="margin-top:6px">{title}: contribution to the {W}&ndash;{R} lead</div>'
        for r in d["segments"][dim]:
            if r["contribution_pp"] is None:
                drv += f'<div class="barrow"><span class="barlab">{e(r["segment"])}</span><span class="muted">no simulated agents</span></div>'
            else:
                drv += _div_bar(f"{r['segment']} (pref. {r['winner']})", r["contribution_pp"], maxabs)
    dis = d["disagreement"]
    dis_html = ""
    if dis["flag"]:
        items = "; ".join(f"{x['value']} prefers {x['winner']}, {x['ahead_of_winner_pp']:.1f}pp ahead of {W}" for x in dis["segments"] if x["dim"] != "industry")
        dis_html = f'<div class="warn"><b>The aggregate result masks significant population disagreement.</b> {e(items)}.</div>'
    drivers = f'<div class="card"><h2>What drives the result?</h2>{drv}{dis_html}</div>'

    sens_rows = "".join(
        f'<div class="barrow"><span class="barlab" style="width:190px">{e(t["assumption"].replace("_", " "))}</span>'
        f'<span class="bartrack" style="width:200px"><div class="barfill" style="width:{min(100, t["index"] / max(d["sensitivity"][0]["index"], 1e-9) * 100):.0f}%;background:{"#D9622B" if t["flips_in_range"] else "#1B7F79"}"></div></span>'
        f'<span class="barval small">{t["swing_pp"]:.1f}pp{" &middot; flips in range" if t["flips_in_range"] else ""}</span></div>'
        for t in d["sensitivity"])
    sens = (f'<div class="card"><h2>Most sensitive assumption</h2><div style="font-size:15px;font-weight:bold">{e(d["most_sensitive"].replace("_", " "))}</div>'
            f'<div class="muted small" style="margin-bottom:6px">Swing in {W}\'s lead across each assumption\'s tested range (orange = the winner changes inside the range)</div>{sens_rows}</div>')

    flips = "".join(f'<li>{e(f["text"])} <span class="pill" style="background:{"#C0392B" if f["within_tested_range"] else "#9AA3B2"}">{"IN RANGE" if f["within_tested_range"] else "OUTSIDE RANGE"}</span></li>' for f in d["flips"][:6])
    stress = "".join(f'<li>If {e(s["dropped"].split("=")[1])} agents were missing: <b>{s["winner"]}</b> preferred{"" if s["same_winner"] else " (the decision changes)"}.</li>' for s in d["stress"])
    change = f'<div class="card"><h2>What would change the decision?</h2><ul class="f">{flips or "<li>No single assumption flips the result.</li>"}</ul><div class="muted small" style="margin-top:6px">Missing-segment stress tests</div><ul class="f">{stress}</ul></div>'

    marks = {True: '<span class="ok">&#10003;</span>', False: '<span class="bad">&#9888;</span>', None: '<span class="amb">&#9888;</span>'}
    ev = "".join(f'<li>{marks[c["ok"]]} {e(c["text"])}</li>' for c in d["checks"])
    warn = "".join(f'<li><span class="bad">&#9888;</span> {e(w)}</li>' for w in d["warnings"])
    evid = f'<div class="card"><h2>Evidence</h2><ul class="f">{ev}{warn}</ul></div>'

    # ask panel
    if live:
        ask = (f'<div class="card"><h2>Ask the simulation</h2><input class="askin" id="askq-{name}" placeholder="Why does the result change for SMB customers?" '
               f'onkeydown="if(event.keyCode===13){{dtAsk(\'{name}\')}}"/> <button class="askbtn" onclick="dtAsk(\'{name}\')">Ask</button>'
               f'<div class="muted small" style="margin-top:4px">Answers use only the evidence ledger. Click a citation to see its provenance.</div><div id="askout-{name}"></div></div>')
    else:
        opts = "".join(f'<option>{e(q)}</option>' for q, _ in (qa or []))
        answers = ""
        provs = ""
        for i, (q, a) in enumerate(qa or []):
            answers += f'<div class="ans" id="a-{name}-{i}" style="display:{"block" if i == 0 else "none"}">{cite_html(a.text, name)}<div class="small muted" style="margin-top:5px">intent: {e(a.intent)}{" &middot; refused: not supported by the evidence" if a.refused else ""}</div></div>'
            for cid in a.citations:
                key = (name, cid)
                provs += prov_block(exp, cid, name) if f'id="p-{name}-{cid}"' not in provs else ""
        ask = (f'<div class="card"><h2>Ask the simulation</h2><select class="askin" onchange="dtPreset(this,\'{name}\')">{opts}</select>'
               f'<div class="muted small" style="margin-top:4px">Pre-computed answers from the same grounded responder as the live API. Click a citation to see its provenance.</div>{answers}{provs}</div>')

    two = (f'<table class="l"><tr><td class="col" style="width:50%;padding-right:5px">{left}{drivers}</td><td class="col" style="width:50%;padding-left:5px">{sens}{change}</td></tr></table>')
    return (f'<div class="scenario" id="s-{name}" style="display:{display}">{banner}{two}'
            f'<table class="l"><tr><td class="col" style="width:35%;padding-right:5px">{evid}</td><td class="col" style="width:65%;padding-left:5px">{ask}</td></tr></table>'
            f'<div class="muted small">Scenario: {e(d["title"])}. {e(d["note"])} All figures are generated by a synthetic simulation fixture; no observed customer data.</div></div>')


def render_page(blocks, names, titles, current, live):
    nav = "".join(f'<a href="{"/?scenario=" + n if live else "#"}" data-sc="{n}" class="{"cur" if n == current else ""}" '
                  f'{"" if live else f"onclick=" + chr(34) + f"dtShow(&#39;{n}&#39;);return false;" + chr(34)}>{e(titles[n])}</a>' for n in names)
    return (f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>DecisionTrace</title><style>{CSS}</style></head><body><div class="wrap">'
            f'<h1>DECISION</h1><div class="q">Which messaging strategy should we launch?</div><div class="scn">{nav}</div>{"".join(blocks)}'
            f'</div><script>{JS}</script></body></html>')
