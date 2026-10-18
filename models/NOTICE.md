# Model files

These models are redistributed here for the head tracker. Licences:

- **YuNet** — MIT License. A copy is in `LICENSE-yunet` (from OpenCV Zoo).
- **BlazeFace** and **FaceMesh** — Apache License 2.0 (full text: https://www.apache.org/licenses/LICENSE-2.0).

| File | Origin | Licence | Notes |
|---|---|---|---|
| `face_detection_yunet.onnx` | YuNet (Shiqi Yu / OpenCV Zoo, `face_detection_yunet_2023mar.onnx`) | MIT | **Used by default.** Fixed input `input` [1,3,640,640], BGR, raw 0..255. Outputs: `cls_{8,16,32}`, `obj_{8,16,32}`, `bbox_{8,16,32}`, `kps_{8,16,32}`. Decode follows OpenCV's `FaceDetectorYN`; validated against `cv2.FaceDetectorYN` on our frames (identical boxes/scores). |
| `face_detection_full_range.onnx` | BlazeFace full-range, from Google's MediaPipe | Apache-2.0 | Kept for reference. TFLite (`blaze_face_full_range.tflite`) from `storage.googleapis.com/mediapipe-models/`. Input `input` [1,3,192,192] NCHW, `[-1,1]`; 48x48 grid anchors (scale 0.44921875). Checked vs TFLite (max diff ~1e-4). |
| `face_detection_short_range.onnx` | BlazeFace short-range, from Google's MediaPipe | Apache-2.0 | Kept for reference. TFLite from the `mediapipe` 0.10.14 wheel. Input `input` [1,3,128,128] NCHW; 896 anchors. Checked vs TFLite (max diff ~5e-5). |
| `face_landmark_192.onnx` | FaceMesh 192x192, via PINTO0309's model zoo (`032_FaceMesh`) | Apache-2.0 | Optional (FaceMesh harness). Input `input` [1,3,192,192] NCHW; outputs `landmarks` [1,1,1,1404] and `score` [1,1,1,1]. Checked vs TFLite (max diff ~1e-4). |

None of these files is modified beyond format conversion. They are not the reference
pipeline's (SCRFD / OpenFace PDM) models, which have research-only terms.
