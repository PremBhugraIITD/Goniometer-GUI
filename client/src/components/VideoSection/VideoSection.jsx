import { useRef, useEffect, useState } from "react";
import axios from "axios";
import "./VideoSection.css";

const VideoSection = ({ activeResult }) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

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

  const takeScreenshot = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (video && canvas) {
      // Set canvas dimensions to match the video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Draw the video frame onto the canvas
      const context = canvas.getContext("2d");
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      // Convert canvas content to a Blob
      canvas.toBlob(async (blob) => {
        if (blob) {
          try {
            setIsProcessing(true);

            // Create FormData and append the blob
            const formData = new FormData();
            formData.append("image", blob, "screenshot.png");
            console.log("Running Sessile Drop");
            const uploadResponse = await axios.post(
              "http://localhost:3000/sessile-drop",
              formData,
              {
                headers: { "Content-Type": "multipart/form-data" },
              }
            );
            console.log("Sessile Drop exited");
            console.log(uploadResponse.data.message);
          } catch (error) {
            console.error("Error:", error);
          } finally {
            setIsProcessing(false);
          }
        }
      }, "image/png");
    }
  };

  return (
    <div className="video-container">
      <div className="video-area">
        {/* Live camera feed */}
        <video
          ref={videoRef}
          autoPlay
          className="video-feed"
        ></video>

        {/* Hidden canvas used for capturing screenshots */}
        <canvas ref={canvasRef} style={{ display: "none" }}></canvas>

        {/* Buttons for user actions */}
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
          <button
            onClick={startScrcpy}
            className="start-scrcpy-button"
          >
            Advanced Tools
          </button>
        </div>
      </div>
    </div>
  );
};

export default VideoSection;
