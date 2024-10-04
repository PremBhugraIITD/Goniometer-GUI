import { useRef, useEffect } from "react";
import "./VideoSection.css";

const VideoSection = () => {
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

  // Function to capture a screenshot
  const takeScreenshot = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (video && canvas) {
      // Set canvas dimensions to match the video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Draw the video frame onto the canvas
      const context = canvas.getContext("2d");
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Create a downloadable image from the canvas
      const imageUrl = canvas.toDataURL("image/png");
      const link = document.createElement("a");
      link.href = imageUrl;
      link.download = "screenshot.png"; // File name for the saved image
      link.click();
    }
  };

  return (
    <div className="video-container">
      <div className="video-area">
        {/* Live camera feed */}
        <video
          ref={videoRef}
          autoPlay
          // style={{ transform: "scaleX(-1)" }}
          className="video-feed"
        ></video>

        {/* Hidden canvas used for capturing screenshots */}
        <canvas ref={canvasRef} style={{ display: "none" }}></canvas>

        {/* Button to take screenshot */}
        <button onClick={takeScreenshot} className="screenshot-button">
          Take Screenshot
        </button>
      </div>
    </div>
  );
};

export default VideoSection;
