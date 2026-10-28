# Paste this whole file into a NEW CELL in your ORIGINAL TRAINING NOTEBOOK.
# Imports, configuration and DATA_DIR must match the completed training run.
# No retraining required. test_data.npy already contains amplitude-scaled data.
from pathlib import Path
import json
import numpy as np

folder=Path(DATA_DIR)
scale=float(json.loads((folder/'meta.json').read_text())['scale'])
if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid amplitude scale')
x=np.load(folder/'test_data.npy',mmap_mode='r')
y=np.load(folder/'test_vel.npy',mmap_mode='r')
raw=np.asarray(x[0],dtype=np.float32)/scale
expected=(int(CFG['shots']),(int(CFG['nt'])+int(CFG['ds'])-1)//int(CFG['ds']),int(NX))
assert raw.shape==expected,(raw.shape,expected)
output=Path.home()/'Downloads'/'matching_test_sample.npz'
output.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(output,seismic=raw,velocity=np.asarray(y[0],dtype=np.float32))
print('Saved:',output,'seismic shape:',raw.shape)
