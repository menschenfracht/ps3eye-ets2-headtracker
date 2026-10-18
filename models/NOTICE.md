# Model files

Both models are licensed under the Apache License 2.0 (full text: https://www.apache.org/licenses/LICENSE-2.0).

| File | Origin | Notes |
|---|---|---|
| `face_detection_short_range.onnx` | BlazeFace short-range face detector, from Google's MediaPipe (Apache-2.0), TFLite model taken from the `mediapipe` 0.10.14 wheel and converted to ONNX with `tflite2onnx` | Input `input` [1,3,128,128] NCHW; outputs `regressors` [1,896,16] and `classificators` [1,896,1]. Output checked against the TFLite original (max diff ~5e-5). |
| `face_landmark_192.onnx` | FaceMesh 192x192 landmark model, from PINTO0309's model zoo (`032_FaceMesh`, Apache-2.0), which redistributes Google's MediaPipe weights | Input `input` [1,3,192,192] NCHW; outputs `landmarks` [1,1,1,1404] (468 points x 3) and `score` [1,1,1,1]. Checked against the TFLite original (max diff ~1e-4). |

Neither file is modified beyond format conversion. These models are not the reference
pipeline's (SCRFD / OpenFace PDM) models, which have research-only terms.
