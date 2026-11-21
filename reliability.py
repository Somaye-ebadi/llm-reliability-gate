"""LLM Reliability Gate: SLO + canary + automatic rollback for prompt changes.

Usage:
  python reliability.py baseline
  python reliability.py canary v2_bad
  python reliability.py canary v2_good
  python reliability.py canary v2_bad --no-guard
"""
import argparse
import atexit
import contextlib
import hashlib
import logging
import json
import os
import random
import re
import sys
import time

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.environ.get("MODEL", "llama3.2:3b")
LABELS = ["billing", "delivery", "tech"]
STABLE = "v1"          # the prompt version currently in production
STAGES = [10, 30, 60, 100]   # canary traffic steps (percent)
WINDOW = 30            # requests per stage
MIN_SAMPLES = 5        # minimum canary answers before judging against the SLO
CACHE_FILE = os.path.join(HERE, "cache.json")

# ---- optional observability (OpenTelemetry -> Tempo / Loki / Prometheus) ----
OTEL_ON = False
TRACER = M_REQ = M_LAT = M_ROLL = None
PACE = 0.0
STATE = {"slo": 0.0, "pct": 0}
LOG = logging.getLogger("llm-gate")
LOG.propagate = False


def init_otel(endpoint):
    """Send traces, metrics and logs to an OTLP/HTTP endpoint (e.g. grafana/otel-lgtm)."""
    global OTEL_ON, TRACER, M_REQ, M_LAT, M_ROLL
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.metrics import Observation
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        sys.exit("Install first: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http")
    res = Resource.create({"service.name": "llm-reliability-gate"})
    tp = TracerProvider(resource=res)
    tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint + "/v1/traces")))
    trace.set_tracer_provider(tp)
    TRACER = trace.get_tracer("llm-gate")
    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=endpoint + "/v1/metrics"), export_interval_millis=2000)
    mp = MeterProvider(resource=res, metric_readers=[reader])
    meter = mp.get_meter("llm-gate")
    M_REQ = meter.create_counter("llm.requests")
    M_LAT = meter.create_histogram("llm.latency", unit="s")
    M_ROLL = meter.create_counter("llm.rollbacks")
    meter.create_observable_gauge("llm.slo.accuracy", callbacks=[lambda o: [Observation(STATE["slo"])]])
    meter.create_observable_gauge("llm.candidate.traffic.percent", callbacks=[lambda o: [Observation(STATE["pct"])]])
    closers = [tp.shutdown, mp.shutdown]
    try:  # logs API is still marked experimental in some SDK versions
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        lp = LoggerProvider(resource=res)
        lp.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter(endpoint=endpoint + "/v1/logs")))
        LOG.addHandler(LoggingHandler(level=logging.INFO, logger_provider=lp))
        LOG.setLevel(logging.INFO)
        closers.append(lp.shutdown)
    except Exception as e:
        print(f"  (logs export disabled: {e})")
    atexit.register(lambda: [c() for c in closers])
    OTEL_ON = True
    print(f"  OpenTelemetry on -> {endpoint}")


@contextlib.contextmanager
def span(name, **attrs):
    if not OTEL_ON:
        yield None
        return
    with TRACER.start_as_current_span(name, attributes=attrs) as sp:
        yield sp


def note(level, msg):
    if OTEL_ON:
        LOG.log(level, msg)


def path(*p):
    return os.path.join(HERE, *p)


def jload(p, default):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def jsave(p, data):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


CACHE = jload(CACHE_FILE, {})


def classify(prompt_name, message):
    """Ask the model; answers are cached so reruns (and the demo) are instant."""
    template = open(path("prompts", prompt_name + ".txt"), encoding="utf-8").read()
    # the key includes a hash of the prompt text, so editing a prompt never reuses stale answers
    key = f"{MODEL}|{prompt_name}|{hashlib.sha1(template.encode()).hexdigest()[:8]}|{message}"
    if key in CACHE:
        return CACHE[key]
    body = {
        "model": MODEL,
        "prompt": template.replace("{message}", message),
        "stream": False,
        "options": {"temperature": 0, "num_predict": 12},
    }
    t0 = time.time()
    try:
        data = requests.post(OLLAMA_URL, json=body, timeout=300).json()
    except requests.exceptions.RequestException as e:
        sys.exit(f"Cannot reach Ollama ({e}).\nIs it running? Try: ollama run {MODEL}")
    if "error" in data:
        sys.exit(f"Ollama error: {data['error']}")
    raw = data.get("response", "").lower()
    found = {lab for lab in LABELS if re.search(r"\b" + lab + r"\b", raw)}
    label = next(iter(found)) if len(found) == 1 else None   # None = invalid output
    res = {"label": label, "latency": round(time.time() - t0, 2)}
    CACHE[key] = res
    jsave(CACHE_FILE, CACHE)
    return res


def serve(prompt_name, it, pct=0):
    """Answer one request and record it as a span, metrics and (on errors) a log line."""
    with span("classify", **{"prompt.version": prompt_name, "rollout.pct": pct}) as sp:
        r = classify(prompt_name, it["text"])
        ok = r["label"] == it["label"]
        if PACE:
            time.sleep(PACE)
        if OTEL_ON:
            sp.set_attribute("llm.correct", ok)
            sp.set_attribute("llm.valid", r["label"] is not None)
            sp.set_attribute("llm.latency_s", r["latency"])
            sp.set_attribute("llm.expected", it["label"])
            sp.set_attribute("llm.got", str(r["label"]))
            M_REQ.add(1, {"prompt": prompt_name, "correct": str(ok).lower()})
            M_LAT.record(r["latency"], {"prompt": prompt_name})
            if not ok:
                LOG.warning("wrong answer prompt=%s expected=%s got=%s", prompt_name, it["label"], r["label"])
    return r, ok


def stats(rows):
    n = len(rows)
    lat = sorted(r["latency"] for r in rows)
    return {
        "accuracy": round(sum(r["ok"] for r in rows) / n, 3),
        "valid": round(sum(r["valid"] for r in rows) / n, 3),
        "p95": lat[min(n - 1, int(0.95 * n))],
    }


def holdout_acc(prompt_name):
    """Accuracy on messages never used for tuning or as prompt examples."""
    items = jload(path("holdout.json"), [])
    ok = 0
    for i, it in enumerate(items, 1):
        ok += classify(prompt_name, it["text"])["label"] == it["label"]
        print(f"\r  holdout {prompt_name}: {i}/{len(items)}", end="", flush=True)
    print()
    return ok / len(items)


def cmd_baseline():
    items = jload(path("messages.json"), [])
    rows = []
    for i, it in enumerate(items, 1):
        r, ok = serve(STABLE, it)
        rows.append({"ok": ok, "valid": r["label"] is not None, "latency": r["latency"]})
        print(f"\r  baseline {STABLE}: {i}/{len(items)}", end="", flush=True)
    print()
    s = stats(rows)
    # The SLO is set relative to what the current prompt really achieves.
    s["slo"] = round(max(0.5, s["accuracy"] - 0.10), 2)
    jsave(path("baseline.json"), s)
    print(f"  accuracy={s['accuracy']:.0%}  valid_output={s['valid']:.0%}  "
          f"p95_latency={s['p95']}s  ->  SLO (min accuracy) = {s['slo']:.0%}")
    return s


def cmd_canary(candidate, guard=True):
    items = jload(path("messages.json"), [])
    base = jload(path("baseline.json"), None) or cmd_baseline()
    slo = base["slo"]
    STATE["slo"] = slo
    print(f"\nRelease gate: {candidate} vs {STABLE} | SLO accuracy >= {slo:.0%} | "
          f"auto-rollback {'ON' if guard else 'OFF'}\n")
    rng = random.Random(7)
    cand_ok, events, active = [], [], STABLE
    for pct in STAGES:
        k = round(WINDOW * pct / 100)
        sample = rng.sample(items, WINDOW)
        to_cand = set(rng.sample(range(WINDOW), k))
        served = []
        STATE["pct"] = pct
        for i, it in enumerate(sample):
            name = candidate if i in to_cand else STABLE
            _, ok = serve(name, it, pct)
            served.append(ok)
            if i in to_cand:
                cand_ok.append(ok)
        n = len(cand_ok)
        acc = sum(cand_ok) / n if n else 0
        served_acc = sum(served) / len(served)
        if guard and n >= 3 and sum(cand_ok) == 0:
            decision, rollback = "ROLLBACK (0 correct answers)", True
        elif guard and n >= MIN_SAMPLES and acc < slo:
            decision, rollback = "ROLLBACK (below SLO)", True
        elif not guard:
            decision, rollback = "no guard: continue", False
        elif n < MIN_SAMPLES:
            decision, rollback = "collecting more data", False
        else:
            decision, rollback = "healthy: continue", False
        events.append({"pct": pct, "n": n, "acc": acc, "served": served_acc, "decision": decision})
        print(f"  stage {pct:>3}% | candidate answers: {n:>2} | candidate accuracy: {acc:>4.0%} "
              f"| users got: {served_acc:>4.0%} correct | {decision}")
        note(logging.WARNING if rollback else logging.INFO,
             f"stage {pct}% candidate={candidate} n={n} acc={acc:.2f} users={served_acc:.2f} -> {decision}")
        if rollback:
            if OTEL_ON:
                M_ROLL.add(1, {"candidate": candidate})
            STATE["pct"] = 0
            active = STABLE
            print(f"\n  >>> Rolled back. All traffic is on {STABLE} again.")
            break
        active = candidate
    holdout = None
    if active == candidate and guard:
        print("\n  Final gate on unseen holdout messages:")
        h_new, h_old = holdout_acc(candidate), holdout_acc(STABLE)
        holdout = (h_new, h_old)
        print(f"  holdout accuracy: {candidate}={h_new:.0%} vs {STABLE}={h_old:.0%}")
        if h_new < h_old - 0.10:
            active = STABLE
            print(f"  >>> Holdout check FAILED. Rolled back to {STABLE}.")
    if active == candidate:
        print(f"\n  >>> {candidate} reached 100% and is now the production prompt.")
    write_report(candidate, slo, guard, events, active, holdout)
    print("  Report written: report.html")


def write_report(candidate, slo, guard, events, active, holdout=None):
    rows = ""
    for e in events:
        bad = "ROLLBACK" in e["decision"]
        color = "#c8372d" if bad else "#1f8a5b"
        rows += (f"<tr><td>{e['pct']}%</td><td>{e['n']}</td><td>{e['acc']:.0%}</td>"
                 f"<td>{e['served']:.0%}</td><td style='color:{color};font-weight:700'>"
                 f"{e['decision']}</td></tr>")
    hz = (f"<div class='b'>آزمون نهایی روی پیام‌های دیده‌نشده: نسخه‌ی جدید <b>{holdout[0]:.0%}</b> | نسخه‌ی فعلی <b>{holdout[1]:.0%}</b></div>"
          if holdout else "")
    html = f"""<!DOCTYPE html><html lang="fa" dir="rtl"><head><meta charset="utf-8">
<title>گزارش انتشار {candidate}</title>
<style>body{{font-family:Tahoma,sans-serif;max-width:820px;margin:30px auto;padding:0 16px;color:#14213d}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd;padding:8px;text-align:center}}
th{{background:#e6edf7}}.b{{padding:12px;border-radius:10px;background:#e6edf7;margin:14px 0}}</style></head><body>
<h2>گزارش انتشار دستور جدید: {candidate}</h2>
<div class="b">هدف (SLO): دقت حداقل <b>{slo:.0%}</b> | rollback خودکار: <b>{'روشن' if guard else 'خاموش'}</b></div>
<table><tr><th>سهم ترافیک نسخه‌ی جدید</th><th>تعداد جواب نسخه‌ی جدید</th><th>دقت نسخه‌ی جدید</th>
<th>دقتی که کاربران دیدند</th><th>تصمیم</th></tr>{rows}</table>
{hz}<div class="b">نسخه‌ی فعال در پایان: <b>{active}</b></div></body></html>"""
    with open(path("report.html"), "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["baseline", "canary"])
    ap.add_argument("candidate", nargs="?", help="prompt name in prompts/, e.g. v2_bad")
    ap.add_argument("--no-guard", action="store_true", help="disable automatic rollback")
    ap.add_argument("--otel", action="store_true", help="send traces/metrics/logs via OpenTelemetry")
    ap.add_argument("--otlp", default="http://localhost:4318", help="OTLP/HTTP endpoint")
    ap.add_argument("--pace", type=float, default=0.0, help="seconds between requests (nicer live graphs)")
    a = ap.parse_args()
    PACE = a.pace
    if a.otel:
        init_otel(a.otlp.rstrip("/"))
    with span("run", command=a.command, candidate=a.candidate or ""):
        if a.command == "baseline":
            cmd_baseline()
        else:
            if not a.candidate:
                sys.exit("Usage: python reliability.py canary v2_bad")
            cmd_canary(a.candidate, guard=not a.no_guard)
