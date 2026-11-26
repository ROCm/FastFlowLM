#!/usr/bin/env python3
"""Host telemetry and accelerator admission. Read-only cases also run on the HTTP harness.

Set FLM_TEST_DEVICE_LOAD=gpu or npu on a Windows test host with working counters to require
measured work from the running model, rather than accepting the unsupported-host nulls.
"""

import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import flm_api
from fixtures import user
from flm_api import Api, case, expect


def get(api, path):
    url = api.url.removesuffix('/v1/chat/completions') + path
    with urllib.request.urlopen(url, timeout=api.timeout) as r:
        expect(r.status == 200 and r.headers.get_content_type() == 'application/json', path)
        return json.load(r)


@case
def device_load_snapshot(api: Api) -> None:
    data = get(api, '/v1/flm/devices')
    expect(set(data) == {'server', 'devices', 'sampled_at', 'system_memory_available_bytes'}, str(data))
    rows = data['devices']
    expect(len(rows) == 2 and {d['device'] for d in rows} == {'gpu', 'npu'}, str(rows))
    for d in rows:
        expect(set(d) == {'device', 'name', 'utilization_percent', 'own_utilization_percent',
                          'memory_shared_bytes', 'memory_dedicated_bytes'}, str(d))
        expect(d['name'] is None or isinstance(d['name'], str), str(d))
        for k in ('utilization_percent', 'own_utilization_percent'):
            expect(d[k] is None or (type(d[k]) in (float, int) and 0 <= d[k] <= 100), str(d))
        for k in ('memory_shared_bytes', 'memory_dedicated_bytes'):
            expect(d[k] is None or (type(d[k]) is int and d[k] >= 0), str(d))
    ram = data['system_memory_available_bytes']
    expect(ram is None or (type(ram) is int and ram >= 0), str(ram))
    stamp = data['sampled_at']
    if stamp is not None:
        expect(datetime.fromisoformat(stamp.replace('Z', '+00:00')).utcoffset() == timedelta(0), stamp)
    s = data['server']
    expect(set(s) == {'accelerator_busy', 'queued_requests'} and type(s['accelerator_busy']) is bool
           and type(s['queued_requests']) is int and s['queued_requests'] >= 0, str(s))


@case
def activity_belongs_to_the_serving_process(api: Api) -> None:
    kind = os.environ.get('FLM_TEST_DEVICE_LOAD')
    if not kind:
        raise flm_api.Skip('set FLM_TEST_DEVICE_LOAD to the model device on a supported Windows host')
    expect(kind in ('gpu', 'npu'), f'unknown FLM_TEST_DEVICE_LOAD={kind}')
    path = '/v1/flm/devices'
    get(api, path)  # prime the rate counter
    body = api.body(user('List 200 distinct picnic ideas in a numbered list, with details.'), max_tokens=320)
    measured = False
    with ThreadPoolExecutor(max_workers=1) as pool:
        turn = pool.submit(api.post, body)
        while not turn.done():
            data = get(api, path)
            row = next(d for d in data['devices'] if d['device'] == kind)
            own, total = row['own_utilization_percent'], row['utilization_percent']
            if data['server']['accelerator_busy'] and own is not None and total is not None and 50 < own <= total:
                measured = True
                break
            time.sleep(.1)
        expect(turn.result().status == 200, 'generation must complete')
    expect(measured, f'never observed this process using the {kind} under load')


@case
def legacy_npu_status_remains_available(api: Api) -> None:
    s = get(api, '/api/npu/status')
    expect(type(s['npu_available']) is bool and type(s['active_requests']) is int
           and isinstance(s['message'], str), str(s))


CASES = flm_api.collect(globals())

if __name__ == '__main__':
    flm_api.main(CASES, __doc__)
