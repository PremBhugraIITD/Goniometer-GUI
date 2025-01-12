import { useRef, useEffect, useState } from "react";
import axios from "axios";
import "./VideoSection.css";

const VideoSection = ({
  activeResult,
  density,
  onProcessingChange,
  onCSVError,
}) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordedChunks = useRef([]);

  useEffect(() => {
    // Request access to the video stream (camera)
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices
        .getUserMedia({ video: true })
        .then((stream) => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
          }
        })
        .catch((err) => {
          console.error("Error accessing the camera: ", err);
        });
    }
  }, []);

  const startScrcpy = async () => {
    try {
      const response = await axios.post("http://localhost:3000/run-scrcpy");
      console.log(response.data);
    } catch (error) {
      console.error("Error starting scrcpy:", error);
    }
  };

  const startRecording = () => {
    const stream = videoRef.current.srcObject;
    if (stream) {
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "video/webm",
      });
      mediaRecorderRef.current = mediaRecorder;
      recordedChunks.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunks.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const blob = new Blob(recordedChunks.current, { type: "video/webm" });
        const formData = new FormData();
        formData.append("video", blob, "screenRecording.webm");

        try {
          setIsProcessing(true);
          onProcessingChange(true);
          if (activeResult === "hysteresis") {
            console.log("Running Hysteresis analysis");
            const uploadResponse = await axios.post(
              "http://localhost:3000/hysteresis-analysis",
              formData,
              {
                headers: { "Content-Type": "multipart/form-data" },
              }
            );
            console.log("Hysteresis analysis exited");
          } else if (activeResult === "pendant-drop-video") {
            formData.append("density", density);
            console.log("Running Pendant Drop (Video) analysis");
            const uploadResponse = await axios.post(
              "http://localhost:3000/pendant-drop-video",
              formData,
              {
                headers: { "Content-Type": "multipart/form-data" },
              }
            );
            console.log("Pendant drop (video) analysis exited");
          }
          onCSVError(false);
          //   console.log(new Date().toLocaleString());
        } catch (error) {
          onCSVError(true);
          console.error("Error running python script video:", error);
        } finally {
          setIsProcessing(false);
          onProcessingChange(false);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const takeScreenshot = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (video && canvas) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const context = canvas.getContext("2d");
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(async (blob) => {
        if (blob) {
          try {
            setIsProcessing(true);
            const formData = new FormData();
            formData.append("image", blob, "screenshot.png");

            if (activeResult === "sessile-drop") {
              console.log("Running Sessile Drop");
              await axios.post("http://localhost:3000/sessile-drop", formData, {
                headers: { "Content-Type": "multipart/form-data" },
              });
              console.log("Sessile Drop exited");
            } else if (activeResult === "pendant-drop-image") {
              formData.append("density", density);
              console.log("Running Pendant Drop");
              await axios.post(
                "http://localhost:3000/pendant-drop-image",
                formData,
                {
                  headers: { "Content-Type": "multipart/form-data" },
                }
              );
              console.log("Pendant Drop exited");
            } else if (activeResult === "calibration") {
              console.log("Running Calibration");
              await axios.post("http://localhost:3000/calibration", formData, {
                headers: { "Content-Type": "multipart/form-data" },
              });
              console.log("Calibration exited");
            }
          } catch (error) {
            console.error("Error running python script:", error);
          } finally {
            setIsProcessing(false);
          }
        }
      }, "image/png");
    }
  };

  return (
    <div className="video-container">
      <video ref={videoRef} autoPlay className="video-feed"></video>
      <canvas ref={canvasRef} style={{ display: "none" }}></canvas>

      <div className="button-group">
        {activeResult === "sessile-drop" && (
          <button
            onClick={takeScreenshot}
            className="screenshot-button"
            disabled={isProcessing}
          >
            {isProcessing ? "Processing..." : "Capture Screenshot"}
          </button>
        )}
        {activeResult === "pendant-drop-image" && (
          <button
            onClick={takeScreenshot}
            className="screenshot-button"
            disabled={isProcessing}
          >
            {isProcessing ? "Processing..." : "Capture Screenshot"}
          </button>
        )}
        {activeResult === "pendant-drop-video" && (
          <>
            {!isRecording ? (
              <button
                onClick={startRecording}
                className="capture-video-button"
                disabled={isProcessing}
              >
                Start Recording
              </button>
            ) : (
              <button
                onClick={stopRecording}
                className="capture-video-button"
                disabled={isProcessing}
              >
                Stop Recording
              </button>
            )}
          </>
        )}
        {activeResult === "calibration" && (
          <button
            onClick={takeScreenshot}
            className="screenshot-button"
            disabled={isProcessing}
          >
            {isProcessing ? "Processing..." : "Capture Screenshot"}
          </button>
        )}
        {activeResult === "hysteresis" && (
          <>
            {!isRecording ? (
              <button
                onClick={startRecording}
                className="capture-video-button"
                disabled={isProcessing}
              >
                Start Recording
              </button>
            ) : (
              <button
                onClick={stopRecording}
                className="capture-video-button"
                disabled={isProcessing}
              >
                Stop Recording
              </button>
            )}
          </>
        )}
        <button
          onClick={startScrcpy}
          className="screenshot-button"
          disabled={!activeResult}
        >
          Advanced Tools
        </button>
      </div>
    </div>
  );
};

export default VideoSection;
