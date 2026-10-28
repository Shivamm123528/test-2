# Seismic Velocity Modelling — full navigation

All 12 sections from the previous app are restored with their original page logic: Home; Model Explorer; Seismic Acquisition; Velocity Prediction; Results and Error Analysis; Time–Depth and Well Comparison; Robustness Lab; Candidate Least Squares; Theory and Methodology; My Experiments; Profile and Settings; Authors.

## Fresh GitHub repository (main only)
1. Extract this ZIP into a NEW folder. Do not mix it with the earlier simplified package.
2. Copy your trained checkpoint into models/fcnvmb_sim.pt. Weights are not included.
3. Create a new GitHub repository with branch main.
4. Upload the extracted CONTENTS, not the ZIP or an enclosing parent folder. app.py and requirements.txt go at root; keep lab, models, supabase and .streamlit folders.
5. Deploy from this new repository, main branch, app.py entry point; Python 3.11 recommended.
6. In the app sidebar expand Checkpoint settings. If using the original notebook checkpoint, verify training_config.json matches your training run and tick the checkbox. The supplied laptop configuration is 201x301, 20 m grid, 5 shots, 4000 steps, decimation 10, 10 Hz, base 16, target offset 1500/range 4500 m/s. Enriched checkpoints already include acquisition metadata.

## Matching seismic: important
Open EXPORT_TEST_SAMPLE.py and copy its WHOLE contents into a new cell in your ORIGINAL velocity_modelling_v2.ipynb, with original imports/configuration/DATA_DIR available. Run that cell, no retraining or FWI rerun. It creates Downloads/matching_test_sample.npz with raw seismic, true velocity and acquisition metadata. Stored notebook seismic is amplitude-scaled; this exporter undoes that scaling so the app applies it exactly once.
In Seismic Acquisition upload this NPZ and click Validate and load NPZ. Then go to Velocity Prediction and click Predict velocity. Run a new model simulation only when you want new synthetic data; full CPU simulation is expensive. No incompatible quick-test examples are bundled.

## Local run on Windows
Open Anaconda Prompt in the extracted folder:

    python -m pip install -r requirements.txt
    python -m streamlit run app.py

After installing dependencies you may also double-click run_windows.bat.

## Accounts and private saved work (optional)
All 12 navigation options appear without database secrets. Geological modelling, acquisition, prediction, errors, time-depth comparison, robustness, residual ranking, theory, downloads and author cards work in guest mode. My Experiments and Profile and Settings retain the original sign-in workflow and require Supabase for persistent private saves.
To enable them in a NEW Supabase project:
1. Run supabase/migrations/001_initial.sql once in the Supabase SQL editor.
2. Configure email authentication/confirmation and recovery settings in Supabase.
3. Put the project URL and public anon key in Streamlit app settings > Secrets using SECRETS_EXAMPLE.toml as the format. Never use a service-role key or commit private credentials.
Existing configured Supabase projects can keep their existing schema/secrets; do not rerun the fresh-project migration on them.

## Authors
Original author cards and details are included. Add original portraits optionally at assets/authors/shivam.jpeg and assets/authors/ankit.png. Without photos the cards show initials.

## Validation and scope
Python files compile, all 12 page labels and branches are present, internal imports resolve at the source level, and the exported raw amplitude/metadata convention matches the supplied notebook. PyTorch and Streamlit were unavailable in the packaging environment, so full inference/UI/account round trips have NOT been executed here. Test locally before deployment.
This restores the previous portal page logic while keeping metadata compatibility and the matching-data exporter. Trained weights, old screenshots, portraits and mismatched quick examples are not bundled. Synthetic constant-density acoustic assumptions do not establish field accuracy.
