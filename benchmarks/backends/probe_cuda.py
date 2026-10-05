import ctypes as c
import json
from pathlib import Path

driver = c.CDLL('/usr/lib/wsl/lib/libcuda.so.1')
result = {'cuInit': driver.cuInit(0)}
dev = c.c_int()
result['cuDeviceGet'] = driver.cuDeviceGet(c.byref(dev), 0)
for key, attr in [('concurrent_managed_access', 89), ('compute_major', 75), ('compute_minor', 76)]:
    value = c.c_int()
    rc = driver.cuDeviceGetAttribute(c.byref(value), attr, dev)
    result[key] = value.value
    result[key + '_status'] = rc
root = Path(__file__).resolve().parents[2] / 'out/results/backends'
root.mkdir(exist_ok=True)
(root / 'cuda_capabilities.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
