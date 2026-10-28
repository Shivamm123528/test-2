# Fix test-2/main
1. Replace app.py at the ROOT of Shivamm123528/test-2, main branch, with the included app.py. Commit changes.
2. Streamlit Manage app > Settings: verify repository Shivamm123528/test-2, branch main, entry point app.py. Reboot. Sidebar must show Build: sample-check + salt · test-2/main.
3. Model Explorer has Include salt body. Tick it then click Generate velocity model to apply.
4. sample_17.npz and sample_29.npz are old acquisition files: 5x250x301, 10 m grid, 4 ms samples, 25 Hz source. The v2 model requires 5x400x301, 20 m grid, 10 ms samples, 10 Hz source. These cannot be made compatible by reshaping.
5. Open EXPORT_TEST_SAMPLE.py. Copy the entire code into a new cell in the ORIGINAL training notebook with the original configuration and DATA_DIR. Run it; no retraining. Upload Downloads/matching_test_sample.npz through the app or into data/ on main. This includes correct raw amplitude and metadata.
6. If original test data are unavailable: Model Explorer > Generate velocity model, optional salt; then Seismic Acquisition > Simulate the current geological model. CPU simulation may take several minutes.
All existing pages are retained. Salt is 4500 m/s and outside the default notebook training prior. Full PyTorch/Streamlit runtime was not executed here. Loader checks accepted matching NPZ and rejected old shapes, wrong frequency and missing metadata.
