# Seismic Velocity Lab — clean main-branch repository

A fresh standalone Streamlit app based on the notebook v2 network and acoustic simulator. No old quick-test samples, login, database, or side-branch dependencies. Trained weights are NOT bundled.

## 1. Prepare the files on Windows
1. Extract the ZIP into a new folder, for example Documents\seismic-velocity-lab.
2. Open that folder. app.py, requirements.txt, training_config.json, lab and models must be directly inside it.
3. Copy your trained checkpoint into models and name it fcnvmb_sim.pt. Either the original notebook checkpoint or your enriched exported checkpoint works.
4. training_config.json describes the original laptop preset: 201 x 301 grid, 20 m spacing, 5 shots, 4000 simulation steps, decimation 10, 10 Hz source and base 16. Change these ONLY if your actual training settings differed. For an original checkpoint, tick the sidebar confirmation to supply this missing metadata. The app still validates weights strictly.

## 2. Create a NEW GitHub repository
1. GitHub: + > New repository.
2. Name: seismic-velocity-lab (or another unused name).
3. Create the repository with the default main branch. No side branch is needed.
4. Add file > Upload files: upload the CONTENTS of the extracted folder. Include the lab and models folders; do not upload just the ZIP or wrap everything in another parent folder.
5. Commit to main. Confirm app.py is at the repository root and models/fcnvmb_sim.pt exists.
6. GitHub browser uploads have file-size limits; if your checkpoint exceeds the displayed limit, use git/Git LFS rather than renaming or splitting it. Your previously observed checkpoint was about 8 MB.

## 3. Run locally (recommended first)
Open Anaconda Prompt and change into the extracted folder:

    cd C:\Users\st981\Documents\seismic-velocity-lab
    python -m pip install -r requirements.txt
    python -m streamlit run app.py

Or double-click run_windows.bat AFTER installing requirements. If missing metadata is reported, verify the sidebar settings and tick the original-checkpoint checkbox.

## 4. Export MATCHING seismic from your training notebook
Open EXPORT_TEST_SAMPLE.py and copy its entire text into a new code cell in velocity_modelling_v2.ipynb. Run after imports/configuration with the ORIGINAL DATA_DIR. No training or FWI rerun is needed. It creates Downloads/matching_test_sample.npz from the first held-out test sample, undoing the notebook amplitude scale because the app applies it once during inference.
In the app choose Upload matching test data, select that NPZ, confirm raw acquisition settings, then Predict velocity.
If your original training data folder is unavailable, Generate and simulate creates data using the loaded checkpoint acquisition. CPU simulation can take several minutes.

## 5. Deploy if desired
In your Streamlit Community Cloud account create an app from the NEW repository: branch main, entry point app.py. Ensure Python supports the requirements (3.11 recommended). No database credentials are required. Reboot after replacing model files. A SHA256 prefix in the sidebar identifies loaded weights.

## What is checked
Checkpoint fields, strict neural-network weight shapes, acquisition consistency, amplitude scale, and finite input arrays of the exact required shape. The original notebook weights omit geometry: confirmed training_config.json supplies it; weights alone cannot recover that geometry. Arbitrary field seismic is not made compatible by reshaping.

## Validation and limits
Source files compile; export scaling and acquisition shape checks were inspected against the supplied notebook. Full neural inference and Streamlit interaction were not executed in the packaging environment because PyTorch is unavailable. Run the local check above before deployment. Existing v2 generator/simulator scientific limitations remain: 2D constant-density isotropic acoustic synthetics, no attenuation, generic geological prior, field performance unverified.

Network/simulator source adapted from your velocity-model-using-cnn v2-test branch (lab/v2_runtime.py and lab/v2_adapter.py). This is a clean reduced app, not a copy of every page in the previous portal.
