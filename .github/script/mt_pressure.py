#!/usr/bin/env python3
"""Hy-MT pressure test against one flm binary. Python 3.9+ standard library only.

usage: python mt_pressure.py --flm-bin FLM --model hy-mt3-flash:1.8b --out DIR [--seconds 180] [--seconds2 90]

1. a sequential reference: every request twice at temperature 0 (the answers must be identical),
2. one server, --threads threads in random order for --seconds: every answer equals the reference,
3. two servers at once, 2 threads each, --seconds2: the same.
Prompts translate 60..4000 tokens of text, so prefill runs at every length class.
"""
import argparse, json, os, random, sys, threading, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from npu_test.server import FlmServer  # noqa: E402

SENT = ('The central bank raised interest rates again on Tuesday as inflation stayed above target, '
        'while the home team won the championship after a late goal and researchers announced a new chip. ')


def build_requests(model):
    reqs = []
    for n in (60, 120, 250, 500, 1000, 2000, 3000, 4000):
        words = SENT.split()
        text = ' '.join(words[i % len(words)] for i in range(n * 3 // 4))
        reqs.append(('len%d' % n, {
            'model': model, 'temperature': 0, 'max_tokens': 48, 'stream': False,
            'messages': [{'role': 'user', 'content': 'Translate the following text into Chinese, '
                          'without additional explanation.\n\n' + text}]}))
    return reqs


def post(port, body, timeout=600):
    q = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(body).encode(),
                               {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(q, timeout=timeout))


def perf(port, model, n, iters=3):
    """TTFT, prefill and decode tok/s of one streamed request of about n prompt tokens (median of iters)."""
    words = SENT.split()
    text = ' '.join(words[i % len(words)] for i in range(n * 3 // 4))
    body = {'model': model, 'temperature': 0, 'max_tokens': 128, 'stream': True,
            'stream_options': {'include_usage': True},
            'messages': [{'role': 'user', 'content': 'Translate the following text into Chinese, '
                          'without additional explanation.\n\n' + text}]}
    rows = []
    for _ in range(iters):
        q = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(body).encode(),
                                   {'Content-Type': 'application/json'})
        t0 = time.time()
        first = last = None
        ntok = ptok = 0
        with urllib.request.urlopen(q, timeout=600) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith('data:') or line.endswith('[DONE]'):
                    continue
                d = json.loads(line[5:])
                if d.get('usage'):
                    ptok = d['usage'].get('prompt_tokens', 0)
                    ntok = d['usage'].get('completion_tokens', 0)
                if d.get('choices') and d['choices'][0].get('delta', {}).get('content'):
                    last = time.time()
                    first = first or last
        if first and ntok > 1:
            rows.append((first - t0, ptok, (ntok - 1) / (last - first)))
    rows.sort()
    return rows[len(rows) // 2] if rows else None


def key(a):
    return a['choices'][0]['message']['content']


def hammer(ports, threads, seconds, reqs, ref):
    stop = time.time() + seconds
    cnt, bad, lock = [0, 0], [], threading.Lock()

    def work(seed, port):
        r = random.Random(seed)
        while time.time() < stop:
            name, body = reqs[r.randrange(len(reqs))]
            try:
                ok = key(post(port, body)) == ref[name]
                err = None if ok else 'different answer'
            except Exception as e:  # noqa: BLE001
                ok, err = False, repr(e)
            with lock:
                cnt[0] += 1
                cnt[1] += ok
                if not ok:
                    bad.append((name, err))
    ts = [threading.Thread(target=work, args=(i * 100 + port, port)) for port in ports for i in range(threads)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return cnt[0], cnt[1], bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--flm-bin', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seconds', type=float, default=180)
    ap.add_argument('--seconds2', type=float, default=90)
    ap.add_argument('--perf', action='store_true')
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reqs = build_requests(a.model)
    failed = False
    if a.perf:
        sp = FlmServer(a.flm_bin, name='mt-perf', log_dir=a.out, args=[])
        sp.start()
        try:
            post(sp.port, reqs[0][1])  # load the model
            print(f'== perf {a.model} (median of 3, streamed)', flush=True)
            print('  prompt_tok |  TTFT s | prefill tok/s | decode tok/s', flush=True)
            for n in (1000, 2000, 4000, 6000, 8000):
                r = perf(sp.port, a.model, n)
                if r:
                    print(f'  {r[1]:10d} | {r[0]:7.3f} | {r[1] / r[0]:13.0f} | {r[2]:12.1f}', flush=True)
                else:
                    print(f'  {n}: no result', flush=True)
        except Exception as e:  # noqa: BLE001
            print('perf phase error', repr(e), flush=True)
        finally:
            sp.stop()
        if not a.seconds:
            return 0
    s1 = FlmServer(a.flm_bin, name='mt-pressure-1', log_dir=a.out, args=[])
    s1.start()
    try:
        ref = {}
        for name, body in reqs:
            t = time.time()
            r1, r2 = post(s1.port, body), post(s1.port, body)
            ref[name] = key(r1)
            same = key(r1) == key(r2)
            failed |= not same
            u = r1.get('usage', {})
            print(f'  ref {name}: {"identical twice" if same else "DIFFERS between two runs"} '
                  f'prompt_tokens={u.get("prompt_tokens")} {time.time() - t:.2f}s (2 requests)', flush=True)
        print(f'== single server, {a.threads} threads, {a.seconds:.0f} s', flush=True)
        n, ok, bad = hammer([s1.port], a.threads, a.seconds, reqs, ref)
        print(f'requests {n} identical {ok} bad {len(bad)} {bad[:5]}', flush=True)
        failed |= bool(bad) or n == 0
        s2 = FlmServer(a.flm_bin, name='mt-pressure-2', log_dir=a.out, args=[])
        s2.start()
        try:
            print(f'== two servers, 2 threads each, {a.seconds2:.0f} s', flush=True)
            n, ok, bad = hammer([s1.port, s2.port], 2, a.seconds2, reqs, ref)
            print(f'requests {n} identical {ok} bad {len(bad)} {bad[:5]}', flush=True)
            failed |= bool(bad) or n == 0
        finally:
            s2.stop()
    finally:
        s1.stop()
    print('RESULT', 'FAIL' if failed else 'PASS', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
