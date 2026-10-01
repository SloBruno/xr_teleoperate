#!/usr/bin/env python3
"""Analisa um diretorio cog_capture_<ts> e gera relatorio markdown. Somente leitura.

Uso: python tools/analyze_cog_capture.py DIR [-o relatorio.md]
"""
import argparse
import json
import sys
from pathlib import Path

PHASE_ORDER = ["inicio", "antes", "mexendo", "depois"]
KNOWN_SET_IDS = {7101, 7102, 7103, 7104, 7105, 7106, 7107, 7108}


def _jl(path):
    out = []
    try:
        for ln in open(path, encoding="utf-8"):
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    except FileNotFoundError:
        pass
    return out


def _js(path, default):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return default


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, "%s%s." % (prefix, k) if prefix else "%s." % k))
    elif isinstance(d, list):
        if d and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in d):
            for i, x in enumerate(d):
                out["%s%d" % (prefix, i)] = x
        else:
            for i, x in enumerate(d[:32]):
                out.update(flatten(x, "%s%d." % (prefix, i)))
    else:
        out[prefix.rstrip(".")] = d
    return {k.rstrip("."): v for k, v in out.items()}


def phase_means(snaps):
    """phase -> {campo: media} para campos numericos dos snapshots."""
    acc = {}
    for s in snaps:
        ph = s.get("phase", "?")
        flat = flatten({k: v for k, v in s.items() if k not in ("t", "phase", "_age")})
        for k, v in flat.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                a = acc.setdefault(ph, {}).setdefault(k, [0.0, 0])
                a[0] += v
                a[1] += 1
    return {ph: {k: a[0] / a[1] for k, a in d.items() if a[1]} for ph, d in acc.items()}


def snapshot_deltas(snaps, a="antes", b="depois", eps=1e-6, top=25):
    m = phase_means(snaps)
    A, B = m.get(a, {}), m.get(b, {})
    rows = [(k, A[k], B[k], B[k] - A[k]) for k in A if k in B and abs(B[k] - A[k]) > eps]
    rows.sort(key=lambda r: -abs(r[3]))
    return rows[:top]


def build_report(d):
    d = Path(d)
    counts = _js(d / "counts.json", {})
    topics = _js(d / "topics.json", {})
    markers = _js(d / "markers.json", [])
    meta = _js(d / "meta.json", {})
    api = _jl(d / "api_events.jsonl")
    snaps = _jl(d / "snapshots.jsonl")
    fchg = _js(d / "file_changes.json", {})
    L = []
    w = L.append
    w("# Relatorio de captura passiva - ajuste de centro de gravidade\n")
    w("Diretorio: `%s`  " % d)
    dur = (meta.get("t_end", 0) - meta.get("t0", 0)) if meta else 0
    w("Duracao: %.1f s | passivo: %s | erros: %s | descartes: %s\n" % (
        dur, meta.get("passive"), meta.get("errors") or "{}", meta.get("dropped")))
    w("## Marcadores\n")
    if markers:
        for m in markers:
            w("- %s: +%.1f s" % (m["name"].upper(), m["rel"]))
    else:
        w("- (nenhum marcador - sem Enter)")
    w("")
    w("## Topicos descobertos: %d\n" % len(topics))
    if topics:
        w("| topico | tipo | pubs | subs |\n|---|---|---|---|")
        for t, e in sorted(topics.items()):
            w("| %s | %s | %s | %s |" % (t, e.get("type"), e.get("pubs"), e.get("subs")))
    w("")
    uns = meta.get("unsupported_type_topics") or {}
    if uns:
        w("Topicos cujo tipo nao tem IDL local (NAO assinados, so listados): " +
          ", ".join("`%s` (%s)" % kv for kv in sorted(uns.items())) + "\n")
    w("## Mensagens por topico e fase\n")
    w("| topico | total | " + " | ".join(PHASE_ORDER) + " |\n|---|---|" + "---|" * len(PHASE_ORDER))
    for t, c in sorted(counts.items()):
        w("| %s | %d | " % (t, c["total"]) + " | ".join(str(c["phase"].get(p, 0)) for p in PHASE_ORDER) + " |")
    w("")
    w("### Topicos com mensagens NOVAS entre os marcadores (mexendo/depois, nao presentes em antes)\n")
    novos = []
    for t, c in sorted(counts.items()):
        ph = c["phase"]
        if c.get("kind") == "api" and (ph.get("mexendo", 0) or ph.get("depois", 0)):
            novos.append("- `%s`: antes=%d mexendo=%d depois=%d" % (t, ph.get("antes", 0), ph.get("mexendo", 0),
                                                                   ph.get("depois", 0)))
        elif ph.get("antes", 0) == 0 and (ph.get("mexendo", 0) or ph.get("depois", 0)):
            novos.append("- `%s` (apareceu so depois do 1o marcador)" % t)
    L.extend(novos or ["- nenhum"])
    w("")
    w("## Requests/Responses de API\n")
    reqs = [e for e in api if e["kind"] == "request"]
    resp = {(e["topic"].rsplit("/", 1)[0], e.get("id")): e for e in api if e["kind"] == "response"}
    w("Total: %d requests, %d responses.\n" % (len(reqs), sum(1 for e in api if e["kind"] == "response")))
    unk = [e for e in reqs if not e.get("known")]
    w("### Requests com api_id DESCONHECIDO: %d\n" % len(unk))
    seen = {}
    for e in unk:
        seen.setdefault((e["topic"], e["api_id"]), []).append(e)
    for (t, aid), es in sorted(seen.items(), key=lambda x: (x[0][0], x[0][1] or 0)):
        e = es[0]
        r = resp.get((t.rsplit("/", 1)[0], e.get("id")))
        w("- `%s` api_id **%s** x%d (fases: %s) param=`%s` -> resp code=%s data=`%s`" % (
            t, aid, len(es), ",".join(sorted({x["phase"] for x in es})), (e.get("parameter") or "")[:300],
            r.get("code") if r else "?", ((r or {}).get("data") or "")[:200]))
    w("")
    strs = [e for e in api if e["kind"] == "string"]
    if strs:
        w("### Mensagens std_msgs/String (canais auxiliares, so tamanho/hash para WebRTC)\n")
        agg = {}
        for e in strs:
            agg.setdefault((e["topic"], e["phase"]), []).append(e)
        for (t, ph), es in sorted(agg.items()):
            w("- `%s` [%s] x%d ex: %s" % (t, ph, len(es), (es[0].get("text") or "len=%s sha=%s" % (es[0]["len"], es[0]["sha1"]))[:160]))
        w("")
    w("### Linha do tempo de requests (exclui repeticoes de GET 7001/7002/7007 e lease)\n")
    n = 0
    last = {}
    for e in api:
        if e["kind"] != "request" or e.get("api_id") in (7001, 7002, 7007, 7109, 102):
            continue
        key = (e["topic"], e["api_id"], e.get("parameter_sha1"), e["phase"])
        last[key] = last.get(key, 0) + 1
        if last[key] > 3:
            continue  # repeticao periodica: mostra so as 3 primeiras por fase
        n += 1
        if n > 200:
            w("- ... (truncado, ver api_events.jsonl)")
            break
        r = resp.get((e["topic"].rsplit("/", 1)[0], e.get("id")))
        w("- +%.2fs [%s] `%s` api_id=%s (%s) param=`%s` -> code=%s" % (
            e["t"] - meta.get("t0", e["t"]), e["phase"], e["topic"], e["api_id"], e.get("api_name") or "DESCONHECIDO",
            (e.get("parameter") or "")[:200], r.get("code") if r else "?"))
    for key, c in sorted(last.items(), key=lambda kv: str(kv[0])):
        if c > 3:
            w("- (x%d no total) `%s` api_id=%s fase=%s" % (c, key[0], key[1], key[3]))
    if n == 0:
        w("- nenhum request relevante registrado")
    w("")
    w("## Deltas dos snapshots (media ANTES -> DEPOIS, maiores)\n")
    rows = snapshot_deltas(snaps)
    if rows:
        w("| campo | antes | depois | delta |\n|---|---|---|---|")
        for k, a, b, dl in rows:
            w("| %s | %.6g | %.6g | %+.4g |" % (k, a, b, dl))
    else:
        w("- sem snapshots em ANTES e DEPOIS (use os 3 Enters) ou sem diferenca")
    w("")
    w("## Arquivos de configuracao\n")
    ch = fchg.get("changes", [])
    w("Candidatos monitorados: %s%s. Alterados: %d\n" % (fchg.get("candidates", "?"),
                                                         " (busca truncada)" if fchg.get("truncated") else "", len(ch)))
    for c in ch:
        w("- %s `%s`" % (c["change"], c["path"]))
    rec = fchg.get("recent_any_name", [])
    if rec:
        w("\nArquivos modificados durante a captura (qualquer nome, ate 300):")
        for p in rec[:60]:
            w("- `%s`" % p)
    dd = d / "diffs"
    if dd.is_dir():
        w("\n### Diffs\n")
        for f in sorted(dd.iterdir()):
            w("```diff\n%s\n```" % f.read_text()[:4000])
    w("")
    w("## Rede (cabecalhos)\n")
    w("Status: %s; conexoes distintas vistas: %s" % (meta.get("net_status", "n/d"), meta.get("net_connections_seen", "n/d")))
    conns = _jl(d / "connections.jsonl")
    for c in conns[:60]:
        w("- [%s] %s %s %s -> %s" % (c["phase"], c["proto"], c["state"], c["local"], c["peer"]))
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("-o", "--out")
    a = ap.parse_args(argv)
    rep = build_report(a.dir)
    if a.out:
        Path(a.out).write_text(rep)
    else:
        sys.stdout.write(rep)


if __name__ == "__main__":
    main()
