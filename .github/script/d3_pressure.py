#!/usr/bin/env python3
"""d3:edge pressure test (text + images) against one flm binary. Python 3.9+ standard library only.

usage: python d3_pressure.py --flm-bin FLM --out DIR [--seconds 300] [--threads 4] [--seconds2 120]

1. a sequential reference: every request once, then a second time (the answers must be bit-identical),
2. one server, --threads threads sending the requests in random order for --seconds: every answer
   must equal the reference bit for bit, no error,
3. two servers at once, 2 threads each, --seconds2: the same.
The requests mix prompts of 20..4000 tokens and 1-3 synthetic images of 224..1344 px, which straddle
the vision host-attention threshold (1024 patches).
"""
import argparse, base64, json, os, random, struct, sys, threading, time, urllib.request, zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from npu_test.server import FlmServer  # noqa: E402


def png(w, h, seed):
    r = random.Random(seed)
    cx, cy, rad = r.randrange(w), r.randrange(h), r.randrange(min(w, h) // 8, min(w, h) // 2)
    c1, c2 = bytes(r.randrange(256) for _ in range(3)), bytes(r.randrange(256) for _ in range(3))
    rows = []
    for y in range(h):
        row = bytearray([0])
        for x in range(w):
            row += c2 if (x - cx) ** 2 + (y - cy) ** 2 < rad * rad else bytes(
                (c1[0] * x // w, c1[1] * y // h, c1[2]))
        rows.append(bytes(row))
    raw = b''.join(rows)

    def chunk(t, d):
        c = struct.pack('>I', len(d)) + t + d
        return c + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    data = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw, 1)) + chunk(b'IEND', b''))
    return 'data:image/png;base64,' + base64.b64encode(data).decode()


Q_IMG = {'scene': {'type': 'choice', 'instructions': 'What is the main subject of the image?',
                   'criteria': {'animal': 'An animal', 'food': 'Food or dessert', 'text': 'Text, logo or graphic',
                                'person': 'A person or people', 'scene': 'A room, street or landscape'}}}
Q_TXT = {'topic': {'type': 'choice', 'instructions': 'What is the topic of this article?',
                   'criteria': {'world': 'World news', 'sports': 'Sports', 'business': 'Business and finance',
                                'scitech': 'Science and technology'}}}
SENT = ('The central bank raised interest rates again on Tuesday as inflation stayed above target, '
        'while the home team won the championship after a late goal and researchers announced a new chip. ')


def build_requests():
    r = random.Random(7)
    reqs = []
    for n in (20, 64, 128, 256, 512, 1024, 2048, 4000):
        words = SENT.split()
        text = ' '.join(words[i % len(words)] for i in range(n * 3 // 4))
        reqs.append(('text%d' % n, {'model': 'd3-edge', 'state': text, 'questions': Q_TXT}))
    sizes = [224, 336, 448, 560, 672, 784, 896, 1120, 1344]
    k = 0
    for s in sizes:
        reqs.append(('img%d' % s, {'model': 'd3-edge', 'questions': Q_IMG, 'images': [png(s, s, k)]}))
        k += 1
    for s in (224, 336, 448):
        for m in (2, 3):
            reqs.append(('img%dx%d' % (s, m), {'model': 'd3-edge', 'questions': Q_IMG,
                                               'images': [png(s, s - 28 * i, k + i) for i in range(m)]}))
            k += m
    reqs.append(('img+text', {'model': 'd3-edge', 'state': SENT * 6, 'questions': Q_IMG,
                              'images': [png(448, 448, 99)]}))
    return reqs


def post(port, body, timeout=300):
    q = urllib.request.Request(f'http://127.0.0.1:{port}/v1/systemone', json.dumps(body).encode(),
                               {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(q, timeout=timeout))


def key(a):
    return json.dumps(a['answers'], sort_keys=True)


def hammer(ports, threads, seconds, reqs, ref):
    stop = time.time() + seconds
    cnt = [0, 0]
    bad = []
    lock = threading.Lock()

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
    ap.add_argument('--out', required=True)
    ap.add_argument('--seconds', type=float, default=300)
    ap.add_argument('--seconds2', type=float, default=120)
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reqs = build_requests()
    print(len(reqs), 'distinct requests', flush=True)
    failed = False
    s1 = FlmServer(a.flm_bin, name='d3-pressure-1', log_dir=a.out, args=['--decision', 'd3:edge'])
    s1.start()
    try:
        ref = {}
        for name, body in reqs:
            t = time.time()
            r1 = post(s1.port, body)
            r2 = post(s1.port, body)
            ref[name] = key(r1)
            same = key(r1) == key(r2)
            failed |= not same
            print(f'  ref {name}: {"identical twice" if same else "DIFFERS between two runs"} '
                  f'tok={r1["usage"]["input_tokens"]} {time.time() - t:.2f}s', flush=True)
        print(f'== single server, {a.threads} threads, {a.seconds:.0f} s', flush=True)
        n, ok, bad = hammer([s1.port], a.threads, a.seconds, reqs, ref)
        print(f'requests {n} identical {ok} bad {len(bad)} {bad[:5]}', flush=True)
        failed |= bool(bad) or n == 0
        s2 = FlmServer(a.flm_bin, name='d3-pressure-2', log_dir=a.out, args=['--decision', 'd3:edge'])
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
