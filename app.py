from pathlib import Path
import csv, io, json, time, hashlib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from lab import auth, store
from lab.authors import render as render_authors
from lab.v2_adapter import (configuration, make_earth, acquisition, time_depth, read_well,
    simulate_earth, load_trained, predict_map, map_scores, rank_candidates, REFERENCES)

ROOT=Path(__file__).resolve().parent
TRAINING_CONFIG=json.loads((ROOT/"training_config.json").read_text())
MODEL_PATH=ROOT/"models/fcnvmb_sim.pt"
st.set_page_config(page_title="Seismic Velocity Modelling",page_icon="🌍",layout="wide")
st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1500px}
h1,h2,h3{letter-spacing:-.025em}
.hero{padding:2.5rem;background:linear-gradient(120deg,#203c4a,#496365);border-radius:20px;color:white;margin:1rem 0 2rem}
.hero h1{color:white;font-size:3rem}.hero p{color:#e4eeee}
[data-testid="stMetricValue"]{color:#c36858}
</style>""",unsafe_allow_html=True)
PAGES=["Home","Model Explorer","Seismic Acquisition","Velocity Prediction","Results and Error Analysis",
       "Time–Depth and Well Comparison","Robustness Lab","Candidate Least Squares",
       "FWI Comparison","Theory and Methodology","My Experiments","Profile and Settings","Authors"]
if "_pending_nav" in st.session_state:st.session_state.nav=st.session_state.pop("_pending_nav")
def goto(page):st.session_state["_pending_nav"]=page;st.rerun()
user=auth.identity()
with st.sidebar:
    st.markdown("### SEISMIC\nVelocity Modelling")
    st.caption("Build: SEG-Y 2GB + salt · test-2/main")
    page=st.radio("Workspace",PAGES,key="nav",label_visibility="collapsed")
    st.divider()
    if user:
        st.caption("Signed in");st.write(user.email)
        if st.button("Log out"):
            try:auth.client().auth.sign_out()
            finally:auth.clear();st.rerun()
    else:st.caption("Guest workspace. Sign in on Home for private cloud saves.")
    st.caption("4 km · up to four geological units · synthetic research")
    with st.expander("Checkpoint settings"):
        st.json(TRAINING_CONFIG)
        use_legacy=st.checkbox("Use these verified training settings for an original notebook checkpoint",key="legacy_metadata")
        st.caption("Confirm the settings match your training run. Enriched checkpoints do not need this option.")
        if MODEL_PATH.exists():st.caption("Weights SHA256: "+hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:16])

@st.cache_resource
def cached_checkpoint(signature,legacy_settings):return load_trained(MODEL_PATH,"cpu",legacy_config=json.loads(legacy_settings) if legacy_settings else None)
def trained():
    if not MODEL_PATH.exists():raise ValueError("Train the revised notebook and copy results/fcnvmb_sim.pt into models/.")
    net,ck=cached_checkpoint(hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),json.dumps(TRAINING_CONFIG,sort_keys=True) if use_legacy else "")
    if ck.get("schema")!="fcnvmb_v2":raise ValueError("Use the checkpoint from velocity_modelling_v2.ipynb.")
    return net,ck
def active_config():
    if MODEL_PATH.exists():
        try:return trained()[1]["config"]
        except Exception as e:st.sidebar.warning("Checkpoint not ready: "+str(e)+" For original weights, verify settings and tick the checkbox above.")
    return configuration(**TRAINING_CONFIG)
C=active_config();A=acquisition(C)
if C.get("quick_test"):
    st.sidebar.warning("Quick-test mode: the tiny training set checks execution, not prediction accuracy.")
current_contract=json.dumps(A,sort_keys=True)
if st.session_state.get("_active_contract")!=current_contract:
    for key in ["sample","prediction","velocity","layer_labels","geology"]:st.session_state.pop(key,None)
    st.session_state._active_contract=current_contract
def chart(fig):
    fig.update_layout(height=410,margin=dict(l=25,r=20,t=50,b=35),paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Arial"))
    st.plotly_chart(fig,use_container_width=True)
def heat(v,title,kind="velocity",meta=None):
    a=A if meta is None else meta
    gather=kind=="gather";error=kind=="error"
    y=np.arange(v.shape[0])*(a["dt_s"] if gather else a["dz_m"]/1000)
    x=np.asarray(a["receiver_x_m"])/1000 if gather else np.arange(v.shape[1])*a["dx_m"]/1000
    lim=max(float(np.quantile(np.abs(v),.995)),1e-12)
    opts=dict(zmin=-lim,zmax=lim) if gather or error else dict(zmin=C["velocity_min"],zmax=C["velocity_max"])
    fig=go.Figure(go.Heatmap(z=v,x=x,y=y,colorscale="RdBu_r" if gather or error else "Viridis",**opts))
    fig.update_layout(title=title,xaxis_title="Horizontal distance (km)",yaxis_title="Recording time (s)" if gather else "Depth (km)")
    fig.update_yaxes(autorange="reversed");chart(fig)
def profiles(v,p,x):
    fig=go.Figure()
    for label,a in [("True / synthetic",v),("Predicted",p)]:
        if a is not None:fig.add_scatter(x=a[:,x],y=np.arange(a.shape[0])*C["dz_m"]/1000,name=label,mode="lines")
    fig.update_layout(xaxis_title="P-wave velocity (m/s)",yaxis_title="Depth (km)");fig.update_yaxes(autorange="reversed");chart(fig)
def validate_meta(meta):
    # Require matching physical coordinates and source, not only matching tensor shape.
    keys=["schema","input_shape","model_shape","dx_m","dz_m","dt_s","source_hz","source_x_m",
          "receiver_x_m","source_z_m","receiver_z_m","preprocessing","physics"]
    missing=[k for k in keys if k not in meta]
    if missing:raise ValueError("Acquisition metadata missing: "+", ".join(missing))
    for k in keys:
        if k=="preprocessing":
            if meta[k]!="raw":raise ValueError("Only calibrated raw amplitudes are accepted by this workflow.")
        elif isinstance(A[k],(list,int,float)):
            try:match=np.shape(meta[k])==np.shape(A[k]) and np.allclose(meta[k],A[k],rtol=1e-6,atol=1e-7)
            except (TypeError,ValueError):match=meta[k]==A[k]
            if not match:raise ValueError(f"Checkpoint acquisition mismatch: {k}.")
        elif meta[k]!=A[k]:raise ValueError(f"Checkpoint acquisition mismatch: {k}.")
def load_arrays(payload,name):
    from lab.data import load_research_sample
    d,v,m=load_research_sample(payload,name,A)
    validate_meta(m);return d,v,m
def set_sample(d,v,m,name):
    for key in ["prediction","inference_seconds","robust"]:st.session_state.pop(key,None)
    st.session_state.sample=dict(data=d,truth=v,meta=m,name=name)
def input_picker():
    with st.expander("Load seismic data",expanded="sample" not in st.session_state):
        st.caption("Prepared examples must match the active checkpoint. Put exported matching NPZ files in data/ to offer them here.")
        examples=sorted((ROOT/"data").glob("*.npz"))
        compatible=[];report=[]
        for example in examples:
            try:
                load_arrays(example.read_bytes(),example.name)
                compatible.append(example)
                report.append(dict(File=example.name,Status="Compatible",Details="Acquisition and shape match"))
            except Exception as e:
                report.append(dict(File=example.name,Status="Incompatible",Details=str(e)))
        if report:
            with st.expander("Samples found in data/",expanded=not compatible):
                st.dataframe(report,use_container_width=True,hide_index=True)
        if compatible:
            chosen=st.selectbox("Prepared matching example",compatible,format_func=lambda p:p.stem)
            if st.button("Load prepared matching example"):
                try:
                    d,v,m=load_arrays(chosen.read_bytes(),chosen.name)
                    set_sample(d,v,m,chosen.stem);st.success("Prepared example loaded.")
                except Exception as e:st.error(str(e))
            if user and st.button('Bookmark prepared example'):
                try:
                    pr=store.profile(user);items=pr.get('bookmarks',[]);item=dict(kind='sample',filename=chosen.name)
                    if item not in items:items.append(item)
                    store.update_profile(user,bookmarks=items[-100:]);st.success('Bookmarked.')
                except Exception as e:st.error(str(e))
        else:
            st.info(f"No compatible prepared sample found. This checkpoint requires {tuple(A['input_shape'])}, {A['dx_m']:g} m grid, {A['dt_s']:g} s saved sampling and {A['source_hz']:g} Hz source.")
            st.write("Open EXPORT_TEST_SAMPLE.py from this repository. Paste its entire contents into a new cell in your original training notebook, with the original DATA_DIR/configuration available. Run the cell, then upload Downloads/matching_test_sample.npz below or add it to this repository's data/ folder.")
            st.caption("Alternatively: Model Explorer → generate a model (optional salt) → Seismic Acquisition → Simulate the current geological model. Full CPU simulation may take several minutes.")
        f=st.file_uploader("Revised NPZ sample (raw seismic, optional velocity, acquisition metadata)",type=["npz"],key="seismic_input")
        st.caption(f"Expected seismic shape {tuple(A['input_shape'])}; velocity shape {tuple(A['model_shape'])}. Export matching data using the supplied EXPORT_TEST_SAMPLE.py. Incompatible samples cannot be fixed by reshaping.")
        if f and st.button("Validate and load NPZ"):
            try:d,v,m=load_arrays(f.getvalue(),f.name);set_sample(d,v,m,f.name);st.success("Loaded.")
            except Exception as e:st.error(str(e))
        raw_file=st.file_uploader("Raw NPY seismic (metadata confirmation required)",type=["npy"],key="raw_npy")
        raw_confirm=st.checkbox("I verified this NPY uses the active checkpoint acquisition and raw amplitudes.")
        if raw_file and raw_confirm and st.button("Load raw NPY"):
            try:
                from lab.data import numeric
                if raw_file.size>64*1024**2:raise ValueError("NPY exceeds 64 MiB.")
                d=numeric(np.load(io.BytesIO(raw_file.getvalue()),allow_pickle=False),tuple(A['input_shape']),"seismic")
                set_sample(d,None,dict(A),raw_file.name);st.success("Raw seismic loaded.")
            except Exception as e:st.error(str(e))
        with st.expander("SEG-Y input"):
            from lab.segy import inspect_segy,load_segy,SHOT_FIELDS
            st.caption('SEG-Y limit: 2 GB. Files are processed for this session and are never saved to Supabase. Large browser uploads still consume server memory; use a sufficiently provisioned local/server deployment for 2 GB surveys.')
            prepared_segy=sorted((ROOT/'data').glob('*.sgy'))+sorted((ROOT/'data').glob('*.segy'))
            selected_segy=st.selectbox('Prepared SEG-Y example',[None]+prepared_segy,format_func=lambda x:'None / upload my file' if x is None else x.stem)
            seg=st.file_uploader("Pre-stack SEG-Y (maximum 2 GB)",type=["sgy","segy"])
            source=seg if seg is not None else selected_segy
            field=st.selectbox("Shot grouping",list(SHOT_FIELDS));endian=st.selectbox("Byte order",["big","little"])
            if source is not None:
                try:
                    payload=source
                    info=inspect_segy(payload,SHOT_FIELDS[field],endian)
                    st.dataframe(info["shots"],use_container_width=True)
                    ids=[r["shot_id"] for r in info["shots"]]
                    selected=st.multiselect(f"Choose {A['shots']} shot IDs",ids,max_selections=A["shots"])
                    st.warning("Field data remain experimental. Matching headers does not validate the synthetic-trained model.")
                    confirm=st.checkbox(f"I verified the {A['source_hz']:g} Hz source wavelet, raw amplitude convention, source/receiver depths and compatible acoustic assumptions.")
                    if st.button("Validate SEG-Y",disabled=len(selected)!=A["shots"] or not confirm):
                        d,v,m=load_segy(payload,selected,SHOT_FIELDS[field],endian,expected=A)
                        m.update({k:A[k] for k in ["schema","input_shape","model_shape","dz_m","source_hz","source_x_m",
                            "receiver_x_m","source_z_m","receiver_z_m","physics"]})
                        set_sample(d,v,m,source.name);st.success("Headers validated; physical assumptions acknowledged by user.")
                except Exception as e:st.error(str(e))
    s=st.session_state.get("sample")
    if s:
        try:
            validate_meta(s["meta"])
            if list(s["data"].shape)!=A["input_shape"]:raise ValueError("Shape changed.")
        except Exception:
            st.warning("The active sample belongs to another acquisition. Load a matching sample.");return None
        st.caption(f"Active sample: {s['name']} · {A['shots']} shots · {A['time_samples']} samples · {A['receivers']} receivers")
    return s
def metrics(p,v):
    values=map_scores(p,v)
    for col,(k,val) in zip(st.columns(4),values.items()):col.metric(k,f"{val:,.3f}")
    st.caption("MSE: (m/s)²; RMSE: m/s; PSNR: dB. PSNR/SSIM use a fixed 4500 m/s range; SSIM is not percentage accuracy.")
    return values
def pack_sample(s,p=None):
    b=io.BytesIO();arrays=dict(seismic=s["data"],metadata=json.dumps(s["meta"]))
    if s["truth"] is not None:arrays["velocity"]=s["truth"]
    if p is not None:arrays["prediction"]=p
    np.savez_compressed(b,**arrays);return b.getvalue()
def save(s,p=None):
    if s['meta'].get('transient_only') or s['meta'].get('input_format')=='SEG-Y':
        st.info('SEG-Y sessions are temporary: Supabase saving is disabled. Download results to keep a local copy.')
        return
    if not user:st.info("Sign in on Home to save privately.");return
    with st.form("save_experiment"):
        name=st.text_input("Experiment name",s["name"],max_chars=100)
        notes=st.text_area("Research notes",max_chars=4000)
        keep=st.checkbox("Save arrays (Supabase limit: 12 MiB)",True)
        ok=st.form_submit_button("Save experiment")
    if ok:
        try:
            blob=pack_sample(s,p) if keep else None
            if blob and len(blob)>12*1024**2:raise ValueError("Arrays exceed 12 MiB. Uncheck arrays to save metadata only.")
            if not name.strip():raise ValueError("Enter a name.")
            vals=map_scores(p,s["truth"]) if p is not None and s["truth"] is not None else {}
            store.save(user,name.strip(),notes,s["meta"],vals,blob);st.success("Saved privately.")
        except Exception as e:st.error(f"Save failed: {e}")

if page=="Home":
    st.markdown('<div class="hero"><h1>Four kilometres.<br>Up to four units.</h1><p>Build constrained synthetic models, inspect seismic evidence and evaluate every prediction.</p></div>',unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.metric("Model depth",f"{(C['nz']-1)*C['dz_m']/1000:g} km")
    b.metric("Default units",C["n_layers"]);c.metric("Checkpoint dataset",f"{C['n_train']:,} train + {C['n_test']:,} test" if C.get("counts_known") else "Counts not recorded")
    st.write("Four units are a project-scale simplification. The real Earth can have more layers; no universal four-layer limit is claimed.")
    st.caption(f"Input {tuple(A['input_shape'])} → output {tuple(A['model_shape'])}; {A['dx_m']:g} m grid, {A['dt_s']*1000:g} ms saved sampling, {A['source_hz']:g} Hz source.")
    st.caption("Use your trained v2 checkpoint. Upload matching held-out data with EXPORT_TEST_SAMPLE.py.")
    if st.button("Explore the geological model",type="primary"):goto("Model Explorer")
    if not user:auth.panel()

elif page=="Model Explorer":
    st.title("Model Explorer")
    with st.form("earth"):
        a,b=st.columns(2);seed=a.number_input("Random seed",0,1000000,17)
        n=b.select_slider("Number of main geological units",options=[2,3,4],value=4)
        salt=st.checkbox("Include salt body",value=False,help="Adds a 4500 m/s salt body to the generated model. Click Generate velocity model to apply.")
        generate=st.form_submit_button("Generate velocity model",type="primary")
    if "velocity" not in st.session_state or generate:
        v,labels,info=make_earth(np.random.default_rng(seed),C,n,salt=salt)
        st.session_state.update(velocity=v,layer_labels=labels,geology=info,model_seed=int(seed),model_layers=int(n),model_salt=bool(salt))
    v=st.session_state.velocity
    if list(v.shape)!=A["model_shape"]:
        st.warning("Regenerate the model after changing the checkpoint.");st.stop()
    a,b=st.columns([2,1])
    with a:heat(v,"Synthetic P-wave velocity")
    with b:
        st.metric("Geological units",st.session_state.geology["n_layers"])
        st.metric("Minimum layer thickness",f"{st.session_state.geology['minimum_thickness_m']:.0f} m")
        st.caption("Velocity gradients inside a unit are not extra geological layers.")
        if st.session_state.geology.get("salt"):
            st.warning("Salt body enabled at 4500 m/s. It is an additional body beyond the 2–4 background units. Your notebook default training prior excluded salt; prediction accuracy on salt is unverified.")
    x=st.slider("Vertical profile position (m)",0,int((C["nx"]-1)*C["dx_m"]),int((C["nx"]-1)*C["dx_m"]/2),int(C["dx_m"]))
    profiles(v,None,int(x/C["dx_m"]))
    with st.expander("How was this model designed?"):
        st.write("Use the v2 notebook lithology/gradient generator, with 2–4 units, at least 300 m initial column thickness and at least 200 m interface spacing before optional fault displacement. Interfaces are folded and smoothed; weak lateral heterogeneity is included. The optional salt body uses the notebook generator at 4500 m/s and adds a body beyond the background layers.")
        st.json(st.session_state.geology)
        st.caption("This matches the v2 notebook prior, whose training examples may contain two, three or four units. The lithology labels and velocities are synthetic assumptions, not local field measurements.")
    b=io.BytesIO();np.save(b,v)
    st.download_button("Download velocity map",b.getvalue(),"velocity_model.npy")
    if user and st.button("Bookmark seed and layer count"):
        try:
            pr=store.profile(user);items=pr.get("bookmarks",[])
            item=dict(kind="four_unit_seed",seed=st.session_state.model_seed,layers=st.session_state.model_layers,salt=st.session_state.get("model_salt",False))
            if item not in items:items.append(item)
            store.update_profile(user,bookmarks=items[-100:]);st.success("Bookmarked.")
        except Exception as e:st.error(str(e))
    if st.button("Continue to acquisition"):goto("Seismic Acquisition")

elif page=="Seismic Acquisition":
    st.title("Seismic Acquisition");s=input_picker()
    with st.expander("Simulate the current geological model"):
        st.write(f"{C['shots']} sequential shots; {C['nt']} time steps; {C['source_hz']:g} Hz Ricker; {C['nt']*C['dt_sim_s']:g} s duration.")
        st.warning("Simulation on this server uses CPU and can take a long time. Generate the dataset on your GPU notebook and upload an NPZ for routine app use.")
        if st.button("Simulate model on CPU",disabled="velocity" not in st.session_state):
            bar=st.progress(0.)
            try:
                d=simulate_earth(st.session_state.velocity,C,"cpu",bar.progress)
                m={**A,**st.session_state.geology,"seed":st.session_state.model_seed}
                set_sample(d,st.session_state.velocity,m,"Generated model");st.rerun()
            except Exception as e:st.error(str(e))
    if s:
        geometry=go.Figure()
        geometry.add_scatter(x=np.asarray(A['receiver_x_m'])/1000,y=np.full(A['receivers'],A['receiver_z_m']),mode='markers',name='Receivers',marker=dict(size=3))
        geometry.add_scatter(x=np.asarray(A['source_x_m'])/1000,y=np.full(A['shots'],A['source_z_m']),mode='markers',name='Sources',marker=dict(size=12,symbol='star'))
        geometry.update_layout(title='Source and receiver geometry',xaxis_title='Distance (km)',yaxis_title='Depth (m)');chart(geometry)
        shot=st.slider("Shot",1,A["shots"],max(1,A["shots"]//2))-1
        heat(s["data"][shot],f"Shot {shot+1}","gather")
        receiver=st.slider("Receiver index",0,A["receivers"]-1,A["receivers"]//2)
        f=go.Figure(go.Scatter(x=np.arange(A["time_samples"])*A["dt_s"],y=s["data"][shot,:,receiver]))
        f.update_layout(xaxis_title="Recording time (s)",yaxis_title="Amplitude");chart(f)
        time=np.arange(C['nt'])*C['dt_sim_s']; q=(np.pi*C['source_hz']*(time-1/C['source_hz']))**2
        wave=(1-2*q)*np.exp(-q)
        f=go.Figure(go.Scatter(x=time,y=wave));f.update_layout(title=f"{C['source_hz']:g} Hz Ricker source",xaxis_title='Time (s)',yaxis_title='Amplitude');chart(f)
        freq=np.fft.rfftfreq(s['data'].shape[1],A['dt_s']);amp=np.abs(np.fft.rfft(s['data'],axis=1)).mean(axis=(0,2))
        f=go.Figure(go.Scatter(x=freq,y=amp));f.update_layout(title='Mean amplitude spectrum',xaxis_title='Frequency (Hz)',yaxis_title='Amplitude');chart(f)
        if st.button('Continue to velocity prediction'):goto('Velocity Prediction')
        st.json(A,expanded=False)
        st.download_button("Download sample",pack_sample(s),"seismic_sample.npz")

elif page=="Velocity Prediction":
    st.title("Velocity Prediction");s=input_picker()
    if not MODEL_PATH.exists():st.info("Copy the newly trained checkpoint to models/fcnvmb_sim.pt.")
    if s:
        if s["meta"].get("experimental"):st.warning("Field-data prediction is experimental and has no demonstrated field accuracy.")
        if st.button("Predict velocity",type="primary",disabled=not MODEL_PATH.exists()):
            try:
                validate_meta(s["meta"]);net,ck=trained();start=time.perf_counter()
                p=predict_map(net,ck,s["data"],"cpu")
                st.session_state.prediction=p;st.session_state.inference_seconds=time.perf_counter()-start
                st.success(f"Prediction and preprocessing: {st.session_state.inference_seconds:.3f} s")
            except Exception as e:st.error(str(e))
        if "prediction" in st.session_state:
            heat(st.session_state.prediction,"Predicted P-wave velocity")
            if st.button("Inspect errors and depth profiles"):goto("Results and Error Analysis")

elif page=="Results and Error Analysis":
    st.title("Results and Error Analysis")
    s=st.session_state.get("sample");p=st.session_state.get("prediction")
    if s is None or p is None:st.info("Load a matching sample and predict first.")
    else:
        v=s["truth"]
        if v is not None:
            for col,a,title,kind in zip(st.columns(3),[v,p,p-v],["True","Predicted","Signed error"],["velocity","velocity","error"]):
                with col:heat(a,title,kind)
            values=metrics(p,v)
            step=max(1,v.size//6000)
            f=go.Figure(go.Scattergl(x=v.ravel()[::step],y=p.ravel()[::step],mode='markers',marker=dict(size=3,opacity=.3),name='Pixels'))
            f.add_scatter(x=[1500,6000],y=[1500,6000],mode='lines',name='1:1')
            f.update_layout(xaxis_title='True velocity (m/s)',yaxis_title='Predicted velocity (m/s)');chart(f)
            lo,hi=st.slider("Depth interval (m)",0,int((C["nz"]-1)*C["dz_m"]),(0,int((C["nz"]-1)*C["dz_m"])),int(C["dz_m"]))
            if hi-lo>=6*C["dz_m"]:st.json(map_scores(p[int(lo/C["dz_m"]):int(hi/C["dz_m"])+1],v[int(lo/C["dz_m"]):int(hi/C["dz_m"])+1]))
            else:st.caption("Choose at least seven depth samples for SSIM.")
            x=st.slider("Profile position (m)",0,int((C["nx"]-1)*C["dx_m"]),int((C["nx"]-1)*C["dx_m"]/2),int(C["dx_m"]))
            profiles(v,p,int(x/C["dx_m"]))
        else:heat(p,"Predicted velocity");st.info("No true velocity: model-domain error cannot be computed.")
        st.download_button("Download prediction and input",pack_sample(s,p),"velocity_result.npz")
        vals=map_scores(p,s['truth']) if s['truth'] is not None else {}
        st.download_button('Download metrics JSON',json.dumps(vals,indent=2),'metrics.json')
        out=io.StringIO();np.savetxt(out,p,delimiter=',',fmt='%.3f')
        st.download_button('Download prediction CSV',out.getvalue(),'prediction.csv')
        save(s,p)
    st.divider();st.subheader("All-model test report")
    report=st.file_uploader("Upload results/test/per_model.csv from the notebook",type=["csv"])
    if report:
        try:
            rows=list(csv.DictReader(io.StringIO(report.getvalue().decode("utf-8-sig"))))
            vals=np.asarray([float(r["RMSE"]) for r in rows])
            if not len(rows) or not np.isfinite(vals).all():raise ValueError("Invalid report.")
            st.write(f"{len(rows)} models · mean RMSE {vals.mean():.2f} m/s · median {np.median(vals):.2f} · 95th percentile {np.quantile(vals,.95):.2f}")
            chart(go.Figure(go.Histogram(x=vals,nbinsx=40)).update_layout(xaxis_title="RMSE (m/s)",yaxis_title="Model count"))
            st.dataframe(sorted(rows,key=lambda r:float(r["MSE"])),use_container_width=True)
            st.caption("Display every model, including failures. Lowest test error is descriptive; it must not select the trained checkpoint.")
        except Exception as e:st.error(str(e))

elif page=="Time–Depth and Well Comparison":
    st.title("Time–Depth and Well Comparison")
    st.latex(r"t_{2w}(z,x)=2\int_0^z \frac{d\zeta}{v_P(\zeta,x)}")
    st.caption("Vertical two-way time in a column approximation. Dipping reflectors and finite-offset travel paths require ray tracing or wave modelling.")
    s=st.session_state.get("sample");p=st.session_state.get("prediction")
    v=s["truth"] if s is not None else st.session_state.get("velocity")
    if v is None and p is None:st.info("Generate a model or load a sample first.")
    else:
        x=st.slider("Column position (m)",0,int((C["nx"]-1)*C["dx_m"]),int((C["nx"]-1)*C["dx_m"]/2),int(C["dx_m"]))
        ix=int(x/C["dx_m"]);z=np.arange(C["nz"])*C["dz_m"];f=go.Figure();series={}
        for label,a in [("True / synthetic",v),("Predicted",p)]:
            if a is None:continue
            if np.any(a[:,ix]<=0):
                st.warning("Prediction contains nonpositive velocities; time–depth integration is undefined.");continue
            series[label]=time_depth(a[:,ix],C["dz_m"]);f.add_scatter(x=series[label],y=z/1000,name=label)
        f.update_layout(xaxis_title="Vertical two-way time (s)",yaxis_title="Depth (km)");f.update_yaxes(autorange="reversed");chart(f)
        if len(series)==2:st.metric("Vertical time–depth RMSE",f"{np.sqrt(np.mean((series['True / synthetic']-series['Predicted'])**2)):.4f} s")
        rows=["depth_m,"+",".join(series)]+[str(depth)+","+",".join(str(t[i]) for t in series.values()) for i,depth in enumerate(z)]
        st.download_button("Download time–depth CSV","\n".join(rows),"time_depth.csv")
        well=st.file_uploader("Measured well CSV: depth_m,vp_ms (one well per file)",type=["csv"])
        if well:
            try:
                wz,wv=read_well(io.BytesIO(well.getvalue()));f=go.Figure()
                f.add_scatter(x=wv,y=wz/1000,name="Uploaded well")
                if v is not None:f.add_scatter(x=v[:,ix],y=z/1000,name="Synthetic / true model")
                if p is not None:f.add_scatter(x=p[:,ix],y=z/1000,name="Prediction")
                f.update_layout(xaxis_title="P-wave interval velocity (m/s)",yaxis_title="Depth (km)");f.update_yaxes(autorange="reversed");chart(f)
                st.caption("An unrelated well provides a range/trend comparison, not prediction ground truth. Verify origin, datum, units and processing.")
                colocated=st.checkbox("This measured well is co-located with this seismic prediction and uses the same depth datum.")
                if colocated and p is not None:
                    mask=(wz>=z[0])&(wz<=z[-1])
                    if mask.sum()<2:raise ValueError("Insufficient overlapping well samples.")
                    pp=np.interp(wz[mask],z,p[:,ix]);st.metric("Well-sample velocity RMSE",f"{np.sqrt(np.mean((pp-wv[mask])**2)):.2f} m/s")
            except Exception as e:st.error(str(e))

elif page=="Robustness Lab":
    st.title("Robustness Lab");s=input_picker()
    if s:
        noise=st.slider("Gaussian noise (%)",0,25,5)/100
        amplitude=st.slider("Amplitude multiplier",.25,3.,1.,.05)
        mode=st.selectbox("High-pass method",["Hz","Notebook bins"])
        cutoff=st.slider("Cutoff (Hz)" if mode=="Hz" else "Fraction of lowest FFT bins",0.,float(.4 if mode=="Notebook bins" else .5/A['dt_s']),0.,.01 if mode=="Notebook bins" else .5)
        seed=st.number_input("Noise seed",0,100000,0)
        if st.button("Apply from original input"):
            d=s['data'].copy()*amplitude
            if noise:d+=np.random.default_rng(seed).normal(0,np.quantile(np.abs(d),.995)*noise,d.shape).astype(np.float32)
            if cutoff:
                ff=np.fft.rfft(d,axis=1)
                if mode=='Hz':ff[:,np.fft.rfftfreq(d.shape[1],A['dt_s'])<cutoff,:]=0
                else:ff[:,:int(ff.shape[1]*cutoff),:]=0
                d=np.fft.irfft(ff,n=d.shape[1],axis=1).astype(np.float32)
            st.session_state.robust=dict(data=d,settings=dict(noise=noise,amplitude=amplitude,mode=mode,cutoff=cutoff,seed=int(seed)))
        r=st.session_state.get('robust')
        if r:
            st.json(r['settings']);shot=min(2,A['shots']-1)
            for col,d,title in zip(st.columns(2),[s['data'],r['data']],['Original shot','Altered shot']):
                with col:heat(d[shot],title,'gather')
            fig=go.Figure()
            for d,label in [(s['data'],'Original'),(r['data'],'Altered')]:
                fig.add_scatter(x=np.fft.rfftfreq(d.shape[1],A['dt_s']),y=np.abs(np.fft.rfft(d,axis=1)).mean(axis=(0,2)),name=label)
            fig.update_layout(xaxis_title='Frequency (Hz)',yaxis_title='Mean amplitude');chart(fig)
            if st.button('Compare predictions'):
                try:
                    net,ck=trained();r['baseline']=predict_map(net,ck,s['data']);r['pred']=predict_map(net,ck,r['data'])
                except Exception as e:st.error(str(e))
            if 'pred' in r:
                for col,v,title in zip(st.columns(2),[r['baseline'],r['pred']],['Original-input prediction','Altered-input prediction']):
                    with col:heat(v,title)
                if s['truth'] is not None:st.dataframe([dict(condition=k,**map_scores(r[key],s['truth'])) for k,key in [('Original','baseline'),('Altered','pred')]])
                save({**s,'data':r['data'],'name':s['name']+' robustness','meta':{**s['meta'],'robustness':r['settings']}},r['pred'])
        st.caption('Every perturbation starts from original input; saved settings describe the displayed experiment.')

elif page=="FWI Comparison":
    st.title('Deep learning and FWI')
    h=json.loads((ROOT/'assets/notebook_metrics.json').read_text());f=h['fwi']
    st.info('Historical original notebook CPU run. These values are not measurements of your current checkpoint or session.')
    st.dataframe([dict(Quantity='MSE (m/s)²',FCN=f['network_mse'],FWI=f['fwi_mse']),dict(Quantity='Time per model (s)',FCN=f['inference_s'],FWI=f['fwi_s'])])
    fig=go.Figure(go.Bar(x=['FCN inference','FWI'],y=[f['inference_s'],f['fwi_s']]))
    fig.update_yaxes(type='log',title='Time (s)');chart(fig)
    st.write('The network was faster; FWI was more accurate on the one compared model. FWI started from a Gaussian-smoothed true model. Network training and dataset generation are excluded from inference timing.')
    st.write(f"Historical network training: {f['training_min']} min; compared models: {f['models']}.")
    st.subheader('Historical robustness results');st.dataframe(h['conditions'])
    st.caption('Historical metrics used the original model range; do not compare PSNR directly with this v2 app’s 4500 m/s range. No live FWI optimisation or fabricated FWI maps are presented.')

elif page=="Candidate Least Squares":
    st.title("Candidate Least Squares");s=input_picker()
    st.write("Compare candidate velocity models by re-simulating their waveforms and computing the residual against the observed gathers.")
    st.latex(r"J(v)=\frac{1}{N}\sum_i\left[F(v)_i-d_i\right]^2")
    st.caption("The lowest residual ranks the supplied candidates. It does not establish a unique Earth model. Source wavelet, geometry and amplitude conventions must match.")
    candidates=st.file_uploader("Candidate velocity NPY maps",type=["npy"],accept_multiple_files=True)
    if s and candidates:
        agree=st.checkbox("Run the expensive CPU forward simulations for these candidates.")
        if st.button("Rank candidates",disabled=not agree):
            try:
                models=[]
                for f in candidates:
                    if f.size>4*1024**2:raise ValueError("Candidate file exceeds 4 MiB.")
                    v=np.load(io.BytesIO(f.getvalue()),allow_pickle=False)
                    if list(v.shape)!=A["model_shape"]:raise ValueError("Candidate grid mismatch.")
                    models.append((f.name,v))
                rows=rank_candidates(s["data"],models,C,"cpu");st.dataframe(rows,use_container_width=True)
                st.download_button("Download residual ranking",json.dumps(rows,indent=2),"candidate_residuals.json")
            except Exception as e:st.error(str(e))

elif page=="Theory and Methodology":
    st.title("Theory and Methodology")
    st.markdown("""
1. **Geology:** 4 km depth, two to four background geological units, with notebook thickness constraints. Optional salt adds a 4500 m/s body and is outside the default training prior.
2. **Velocity prior:** illustrative increasing sedimentary interval velocities with small within-unit depth gradients. Local well calibration is optional in the notebook.
3. **Physics:** 2-D constant-density acoustics, eighth-order space/second-order time, side/bottom sponge. No calibrated free surface, attenuation, elasticity or anisotropy.
4. **Acquisition:** checkpoint metadata defines source and receiver positions, depths, source frequency, record length and sampling. Read the active acquisition settings below; the supplied laptop configuration uses 5 shots, 10 Hz and a 4 s record.
5. **Learning:** a modified U-Net maps shot gathers to a velocity grid. The v2 notebook crops the final feature map when large enough and interpolates otherwise. This learned projection is not physical time–depth conversion.
6. **Split:** the notebook laptop preset uses 24 training maps and 4 test maps. Actual checkpoint dataset counts may not be recorded. The separate full time–depth benchmark has 17,000/3,000 point samples; these are not FCN velocity-map counts.
7. **Evaluation:** save every test prediction and its error, depth-dependent error and vertical time–depth comparison. Report aggregate and failure statistics.
8. **Reality check:** real Earth can have more than four layers. Marmousi2 and OpenFWI are synthetic benchmarks. Field claims require measured seismic and independent well/velocity constraints.
""")
    st.subheader('Network architecture')
    st.write(f"Modified U-Net: {C['shots']} shot channels; four pooling levels; base width {C['base']}; skip connections; transposed convolution decoder; 1×1 output convolution. Input is padded to a multiple of 16; output is cropped or resized to {C['nz']} × {C['nx']}.")
    st.caption('19 Conv2d layers and 4 transposed convolutions. Learned output projection is not physical time–depth conversion.')
    curves=st.file_uploader('Display your saved training-curve figure (optional)',type=['png','jpg','jpeg'])
    if curves:st.image(curves,caption='User-supplied saved training curves; not a live training run.')
    st.json(A,expanded=False)
    for title,url in REFERENCES:st.link_button(title,url)
    st.caption("The numerical priors are project design choices; they are not tables copied from these references.")

elif page=="My Experiments":
    st.title("My Experiments")
    if not user:auth.panel()
    else:
        try:
            records=store.listing()
            if not records:st.info("No saved experiments.")
            else:
                st.dataframe([{k:r[k] for k in ["name","created_at","notes"]} for r in records],use_container_width=True)
                selected=st.selectbox("Experiment",records,format_func=lambda r:r["name"]+" "+r["created_at"][:16])
                st.json(selected["metrics"]);st.json(selected["settings"],expanded=False)
                if selected.get("object_path") and st.button("Reopen arrays"):
                    blob=store.download(selected);d,v,m=load_arrays(blob,"saved.npz");set_sample(d,v,m,selected["name"])
                    with np.load(io.BytesIO(blob),allow_pickle=False) as z:
                        if 'prediction' in z:
                            from lab.data import numeric
                            st.session_state.prediction=numeric(z['prediction'],tuple(A['model_shape']),'prediction')
                    goto('Results and Error Analysis' if 'prediction' in st.session_state else 'Velocity Prediction')
                compare=st.multiselect("Compare two saved experiments",records,format_func=lambda r:r['name']+' '+r['id'][:6],max_selections=2)
                if len(compare)==2:
                    st.dataframe([dict(name=r['name'],**r['metrics']) for r in compare])
                    st.caption('Compare metrics only for matching truth, acquisition and evaluation definitions.')
                with st.form("notes"):
                    notes=st.text_area("Notes",selected["notes"],max_chars=4000);ok=st.form_submit_button("Update notes")
                if ok:auth.client().table("experiments").update({"notes":notes}).eq("id",selected["id"]).execute();st.success("Updated.")
                confirm=st.checkbox("Delete selected experiment and its arrays")
                if st.button("Delete",disabled=not confirm):store.delete(selected);st.rerun()
        except Exception as e:st.error(f"Could not load saved experiments: {e}")

elif page=="Profile and Settings":
    st.title("Profile and Settings")
    if not user:auth.panel()
    else:
        try:
            pr=store.profile(user)
            with st.form("profile"):
                name=st.text_input("Display name",pr["display_name"],max_chars=80)
                tips=st.checkbox("Show study tips",pr.get("preferences",{}).get("show_tips",True))
                ok=st.form_submit_button("Save profile")
            if ok:store.update_profile(user,display_name=name,preferences={"show_tips":tips});st.success("Saved.")
            for i,item in enumerate(pr.get("bookmarks",[])):
                if item.get('kind')=='sample':
                    if st.button('Open example '+item['filename'],key=f'sample_bookmark_{i}'):
                        f=ROOT/'data'/Path(item['filename']).name
                        if f.is_file():
                            d,v,m=load_arrays(f.read_bytes(),f.name);set_sample(d,v,m,f.stem);goto('Seismic Acquisition')
                        else:st.info('This bookmarked example is not deployed.')
                    continue
                if item.get("kind")!="four_unit_seed":continue
                if st.button(f"Seed {item['seed']} · {item['layers']} units",key=f"bookmark_{i}"):
                    v,labels,info=make_earth(np.random.default_rng(item["seed"]),C,item["layers"],salt=item.get("salt",False))
                    st.session_state.update(velocity=v,layer_labels=labels,geology=info,model_seed=item["seed"],model_layers=item["layers"],model_salt=item.get("salt",False));goto("Model Explorer")
            st.caption("Supabase Auth manages passwords. Passwords are never saved to application profile tables.")
        except Exception as e:st.error(str(e))
elif page=="Authors":render_authors(ROOT)
st.divider();st.caption("Seismic Velocity Modelling · four-unit synthetic study · field performance unverified")


