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
shots=int(CFG['shots']);nx=int(NX);nz=int(NZ);nt=int(CFG['nt']);ds=int(CFG['ds'])
src=np.round(np.linspace(0,nx-1,shots)).astype(int)
metadata=dict(schema='fcnvmb_v2',input_shape=list(raw.shape),model_shape=[nz,nx],
    dx_m=float(DH),dz_m=float(DH),dt_s=float(DT)*ds,dt_sim_s=float(DT),nt=nt,ds=ds,
    source_hz=float(FPEAK),shots=shots,receivers=nx,time_samples=raw.shape[1],
    source_x_m=(src*float(DH)).tolist(),receiver_x_m=(np.arange(nx)*float(DH)).tolist(),
    source_z_m=float(DH),receiver_z_m=float(DH),preprocessing='raw',
    physics='notebook_v2: 2D constant-density acoustic; 8th-order space, 2nd-order time; side/bottom sponge; no physical free surface')
np.savez_compressed(output,seismic=raw,velocity=np.asarray(y[0],dtype=np.float32),metadata=json.dumps(metadata))
print('Saved:',output,'seismic shape:',raw.shape)
