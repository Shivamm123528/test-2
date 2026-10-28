# Selectable samples for test-2/main

1. Extract this ZIP.
2. In GitHub open Shivamm123528/test-2 > data (main branch).
3. Add file > Upload files: upload BOTH NPZ files from the extracted data folder. Commit to main.
4. Replace the repository-root app.py with the included app.py to get sample compatibility reporting and the salt toggle help.
5. Reboot the Streamlit app. Check it deploys test-2/main/app.py.
6. Seismic Acquisition > Prepared matching example: select v2_layered_seed17 or v2_salt_seed29 and click Load prepared matching example. Older app builds use the label Prepared v2 example / Load prepared v2 example instead.
7. Go to Velocity Prediction and click Predict velocity.

Both samples contain raw finite seismic (5,400,301), true velocity (201,301) and complete acquisition metadata: 20 m spacing, 10 ms saved sampling, 10 Hz Ricker, 4000 simulation steps at 1 ms. These were freshly forward simulated, not resized from old files. Checkpoint scaling is applied once by the app.

v2_layered_seed17: layered Earth without salt.
v2_salt_seed29: layered Earth with a 4500 m/s salt body.

You can keep old sample_17/sample_29 files, but they remain incompatible with this checkpoint. New app.py reports why and offers only compatible examples.

Validation: NumPy/SciPy propagation cross-checked with the notebook PyTorch Modeler on a 200-step two-shot test; relative L2 difference ~8e-7. Full-grid data are finite, have correct dimensions and metadata, and were checked with the checkpoint from test-2/main. These are synthetic demonstration samples, not held-out training test data. Passing inference does not establish prediction accuracy. The default training prior excluded salt; the salt sample tests extrapolation.
