# SEG-Y examples + 2 GB transient processing

Replace/add these files in test-2/main (preserve every other file):
- app.py: repository root, replace.
- lab/segy.py: inside lab, replace.
- lab/store.py: inside lab, replace. Adds a second safeguard against saving SEG-Y sessions.
- .streamlit/config.toml: inside .streamlit, create/replace. Contains maxUploadSize = 2000 (Streamlit MiB). Parser separately enforces <= 2,000,000,000 bytes (2 GB).
- data/v2_layered_seed17.sgy and data/v2_salt_seed29.sgy: inside data, add.

Ensure requirements.txt contains segyio>=1.9.13,<2 (already present in the supplied full app). No new Supabase configuration or SQL is needed. Reboot the Streamlit app after uploading. It must deploy test-2/main/app.py. Sidebar build label becomes SEG-Y 2GB + salt.

## Use the small examples
Seismic Acquisition > Load seismic data > SEG-Y input > Prepared SEG-Y example.
Choose one .sgy example. Shot grouping: Field record (byte 9). Byte order: big.
Select shot IDs 1,2,3,4,5, confirm the displayed acquisition assumptions and click Validate SEG-Y.
Go to Velocity Prediction > Predict velocity.
Each file is about 2.77 MB, with 1505 traces: 5 shots x 301 receivers; 400 samples per trace; 10 ms interval; 20 m receiver spacing; 10 Hz synthetic Ricker. IEEE float32 SEG-Y, metre coordinates. These examples contain the same raw seismic as the corresponding NPZ examples. SEG-Y itself supplies no true velocity map, so error metrics are unavailable for the SEG-Y upload alone.

## Large surveys
Upload a SEG-Y of up to 2 GB through the file uploader instead of putting large surveys in GitHub. The reader scans shot headers and reads waveform arrays only for selected shots. File-like sources are copied in 8 MiB chunks into a temporary directory and cleaned after each parsing operation; there is no getvalue() full-size bytes copy. Source and receiver geometry, time sample count, sample interval and shot count must still match the trained checkpoint. A larger size limit does not make arbitrary field surveys compatible, and there is no silent reshaping or resampling.

## Supabase and memory
SEG-Y sessions are tagged transient_only. The UI disables experiment saving for them, including robustness results, and lab/store.py rejects Supabase saves of tagged SEG-Y sessions as a second guard. No SEG-Y file, derived arrays, experiment record or SEG-Y session metadata is sent to Supabase. Download results to your computer if needed. Existing NPZ experiment saves continue as before.
Streamlit's standard browser uploader still holds uploaded bytes in server RAM. Chunked copying avoids additional full-file copies but is not a streaming browser-to-disk transport. A 2 GB upload can exceed a hosted app's RAM even when the configured limit permits it. Use a sufficiently provisioned local/server deployment for large files. The 2 GB workflow was checked through the size guard and chunked I/O tests, not by processing a full 2 GB upload here.

## Validation
Small SEG-Y files were written and reopened with segyio; all waveform values matched the original NPZ exactly. Header grouping, invalid-shot rejection, over-limit rejection, chunk-only file-like reads, transient-save block and prediction using your test-2/main checkpoint were tested. Full Streamlit browser upload/UI was not run here. Salt remains outside the default model training prior.
