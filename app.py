from pathlib import Path
import io,json,hashlib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from lab.v2_adapter import load_trained,predict_map,make_earth,simulate_earth

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title="Seismic Velocity Lab",layout="wide")
st.title("Seismic Velocity Lab")
st.caption("Notebook v2 · synthetic acoustic velocity prediction · 4 km training grid")
path=ROOT/'models/fcnvmb_sim.pt'
if not path.is_file():
    st.info("Put your trained checkpoint at models/fcnvmb_sim.pt, then restart the app.")
    st.stop()
blob=path.read_bytes();digest=hashlib.sha256(blob).hexdigest()
st.sidebar.caption("Checkpoint SHA256: "+digest[:16])
c=json.loads((ROOT/'training_config.json').read_text())
with st.sidebar.expander("Original notebook training settings",expanded=True):
    st.json(c)
    legacy=st.checkbox("Use these settings for a checkpoint without acquisition metadata")
    st.caption("Confirm these match the actual training run. Original laptop preset: 5 shots, 4000 steps, decimation 10, 20 m grid, 10 Hz source.")
@st.cache_resource
def load_checkpoint(digest,settings):
    return load_trained(path,legacy_config=json.loads(settings) if settings else None)
try:
    net,ck=load_checkpoint(digest,json.dumps(c,sort_keys=True) if legacy else '')
except Exception as e:
    st.error(str(e))
    st.info("For an original notebook checkpoint, confirm the settings in the sidebar and tick the checkbox. For different training settings, edit training_config.json first.")
    st.stop()
a=ck['acquisition']; c=ck['config']; shape=tuple(a['input_shape'])
st.success(f"Checkpoint loaded. Required raw seismic shape: {shape}")
st.caption(f"Output: {tuple(a['model_shape'])}; grid {a['dx_m']} m; saved time interval {a['dt_s']} s; source {a['source_hz']} Hz.")
def plot(v,title):
    fig,ax=plt.subplots(figsize=(8,4));im=ax.imshow(v,aspect='auto',extent=[0,(c['nx']-1)*c['dx_m']/1000,(c['nz']-1)*c['dz_m']/1000,0],vmin=1500,vmax=6000,cmap='viridis')
    ax.set(xlabel='Distance (km)',ylabel='Depth (km)',title=title);fig.colorbar(im,ax=ax,label='Velocity (m/s)');st.pyplot(fig);plt.close(fig)
mode=st.radio("Acquisition",['Upload matching test data','Generate and simulate'],horizontal=True)
d=v=None
if mode=='Upload matching test data':
    st.write("Upload raw seismic .npy or .npz with keys seismic and optional velocity. Do not upload a batch or a velocity map as seismic.")
    st.caption("Use EXPORT_TEST_SAMPLE.py in your training notebook to export correctly scaled data. Shape alone does not establish matching acquisition geometry.")
    f=st.file_uploader("Seismic file",type=['npy','npz'])
    if f:
        try:
            arr=np.load(f,allow_pickle=False)
            if isinstance(arr,np.lib.npyio.NpzFile):
                if 'seismic' not in arr.files:raise ValueError("NPZ needs a seismic key; use the supplied notebook exporter.")
                d=arr['seismic'];v=arr['velocity'] if 'velocity' in arr.files else None
            else:d=arr
            st.write("Uploaded shape:",d.shape)
            if tuple(d.shape)!=shape:raise ValueError(f"Received {d.shape}; required {shape}. Export a test sample from the same training run. Do not reshape or pad unrelated data.")
            if not np.isfinite(d).all():raise ValueError('Seismic contains NaN or infinity.')
            if not st.checkbox('I confirm this is raw seismic with the checkpoint acquisition settings.'):
                d=None
        except Exception as e:st.error(str(e));d=None
else:
    st.warning("Full wave simulation on CPU can take several minutes. Matching held-out notebook test data is faster.")
    seed=st.number_input('Earth model seed',0,100000,7);layers=st.slider('Layers',2,4,3)
    if st.button('Generate and simulate'):
        try:
            with st.spinner('Simulating the checkpoint acquisition...'):
                v,_,_=make_earth(np.random.default_rng(seed),c,layers)
                d=simulate_earth(v,c)
            st.session_state['generated']=(digest,int(seed),layers,d,v)
        except Exception as e:st.error(str(e))
    saved=st.session_state.get('generated')
    if saved and saved[:3]==(digest,int(seed),layers):d,v=saved[3:]
if v is not None and v.shape==tuple(a['model_shape']):plot(v,'Reference velocity')
if d is not None:
    st.caption(f"Seismic ready: {d.shape}")
    if st.button('Predict velocity',type='primary'):
        try:
            with st.spinner('Predicting...'):pred=predict_map(net,ck,d)
            plot(pred,'Predicted velocity')
            if v is not None and v.shape==pred.shape:st.metric('RMSE (m/s)',f"{np.sqrt(np.mean((pred-v)**2)):.1f}")
            b=io.BytesIO();np.save(b,pred);st.download_button('Download predicted velocity',b.getvalue(),'predicted_velocity.npy')
        except Exception as e:st.error(str(e))
st.caption('Synthetic research demonstration. Performance on field seismic is unverified.')
