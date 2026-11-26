#!/usr/bin/env python3
"""Host telemetry and accelerator admission. These read-only cases also run on the HTTP harness."""

import json
import urllib.request
from datetime import datetime, timedelta

import flm_api
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
def legacy_npu_status_remains_available(api: Api) -> None:
    s = get(api, '/api/npu/status')
    expect(type(s['npu_available']) is bool and type(s['active_requests']) is int
           and isinstance(s['message'], str), str(s))


CASES = flm_api.collect(globals())

if __name__ == '__main__':
    flm_api.main(CASES, __doc__)
