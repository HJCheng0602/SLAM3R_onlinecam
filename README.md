
# Enhanced SLAM3R: Real-Time Reconstruction via Online Camera Stream

## 🌟 Project Overview

This is an enhanced and improved version of the well-known **SLAM3R** framework, designed to significantly expand its input versatility. The core advancement lies in enabling the use of **online camera feeds**, empowering users to perform **remote** and **real-time** 3D reconstruction using readily available mobile device cameras, thus overcoming the limitations of static local video files.

The overall functionality remains consistent with the original [SLAM3R GitHub repository](https://github.com/PKU-VCL-3DV/SLAM3R).

---

## ✨ Key Added Feature

The single, crucial feature introduced in this improved version is:

-   **🌐 Online Camera Support of gradio demo**: The system now supports using an **IP Camera** (Internet Protocol Camera) stream as the input source for 3D reconstruction.

### How to Use the New Feature

---
#### Demo
To leverage your mobile device camera for reconstruction from **any location**, follow these steps within the Gradio interface:

1.  **Locate the Input Type Menu:**
    Navigate to the **Type** dropdown menu within the Gradio interface.

2.  **Select "web camera":**
    You will find an additional option: **"web camera"**. Select this option.

3.  **Provide Connection Details:**
    Once selected, you must input the necessary credentials to connect to your mobile device's camera stream:
    | Configuration Item | Description |
    | :--- | :--- |
    | **IP Camera Sharing URL** | Enter the **streaming URL** (e.g., `rtsp://...`) provided by your IP camera application. |
    | **Password** | Enter the required **password** for accessing the shared IP camera stream. |

4.  **Initiate Reconstruction:**
    After entering the details, you can begin the 3D reconstruction process using the live camera feed.
---
#### Script
look up the hint in the `scripts/demo_wild.sh`:

```bash
######################################################################################
# Your input can be a directory of images, a video file, and a webcamera url.
# For directory of images, set the TEST_DATASET to path like: data/test/myimages
# For video, set the TEST_DATASET to path like: data/test/myvideo.mp4
# For webcamera, set the TEST_DATASET to url like: http://rab:12345678@192.168.137.83:8081
######################################################################################
```
then you can try

```bash
bash scripts/demo_wild.sh
```

Now you can start reconstruction now !

---

## ⚠️ Performance Note on IP Camera Usage

It is crucial to understand that the practical usability of the IP camera stream may be significantly affected by:

-   **Latency (Delay)**: High network latency can introduce inconsistencies and instability, potentially impacting the quality and real-time performance of the reconstruction.

---

## 📚 IP Camera Information

For detailed instructions and information on how to set up, share, and obtain the necessary URL and password for your mobile device as an IP camera, please refer to this resource:

-   **🔗 IP Camera Bridge Reference**: [ipcamera](https://github.com/shenyaocn/IP-Camera-Bridge)