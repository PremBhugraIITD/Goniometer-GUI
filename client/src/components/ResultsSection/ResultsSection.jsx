import React, { useEffect, useState } from "react";
import download_icon from "../../assets/download_icon.svg";
import "./ResultsSection.css";

const ResultsSection = ({ activeResult, isProcessing, csvError }) => {
  const [sessileDropOutput, setSessileDropOutput] = useState("");
  const [pendantDropOutput, setPendantDropOutput] = useState("");
  const [hysteresisOutput, setHysteresisOutput] = useState("");
  const [calibrationOutput, setCalibrationOutput] = useState("");

  const downloadCSV = () => {
    window.location.href = "http://localhost:3000/download-results";
  };

  useEffect(() => {
    if (activeResult === "sessile-drop") {
      const eventSource = new EventSource(
        "http://localhost:3000/sessile-drop-stream"
      );

      eventSource.onmessage = (event) => {
        setSessileDropOutput(event.data.replace(/^"|"$/g, ""));
      };

      eventSource.onerror = () => {
        console.error("Error connecting to SSE endpoint");
        eventSource.close();
      };

      return () => {
        eventSource.close();
      };
    } else if (activeResult === "pendant-drop") {
      const eventSource = new EventSource(
        "http://localhost:3000/pendant-drop-stream"
      );
      eventSource.onmessage = (event) => {
        setPendantDropOutput(event.data.replace(/^"|"$/g, ""));
      };
      eventSource.onerror = () => {
        console.error("Error connecting to SSE endpoint");
        eventSource.close();
      };
      return () => {
        eventSource.close();
      };
    } else if (activeResult === "hysteresis") {
      const eventSource = new EventSource(
        "http://localhost:3000/hysteresis-stream"
      );
      eventSource.onmessage = (event) => {
        setHysteresisOutput(event.data.replace(/^"|"$/g, ""));
      };
      eventSource.onerror = () => {
        console.error("Error connecting to SSE endpoint");
        eventSource.close();
      };
      return () => {
        eventSource.close();
      };
    } else if (activeResult === "calibration") {
      const eventSource = new EventSource(
        "http://localhost:3000/calibration-stream"
      );

      eventSource.onmessage = (event) => {
        setCalibrationOutput(event.data.replace(/^"|"$/g, ""));
      };

      eventSource.onerror = () => {
        console.error("Error connecting to SSE endpoint");
        eventSource.close();
      };

      return () => {
        eventSource.close();
      };
    }
  }, [activeResult]);

  return (
    <div className="results-container">
      <h2>Results</h2>
      <div className="results-scroll">
        {activeResult === "sessile-drop" ? (
          <div className="results-area" id="sessile-drop-analysis">
            {console.log(sessileDropOutput.split(/\\r\\n/))}
            {sessileDropOutput.split(/\\r\\n/).map((line, index) => {
              return <p key={index}>{line}</p>;
            })}
          </div>
        ) : activeResult === "hysteresis" ? (
          <div className="results-area" id="hysteresis-analysis">
            {isProcessing || csvError ? (
              <div className="results-area">
                {console.log(hysteresisOutput.split(/\\r\\n/))}
                {hysteresisOutput.split(/\\r\\n/).map((line, index) => {
                  return <p key={index}>{line}</p>;
                })}
              </div>
            ) : (
              <div className="results-area">
                <button onClick={downloadCSV} className="download-button">
                  Download CSV
                </button>
              </div>
            )}
          </div>
        ) : activeResult === "pendant-drop" ? (
          <div className="results-area" id="pendant-drop-analysis">
            {console.log(pendantDropOutput.split(/\\r\\n/))}
            {pendantDropOutput.split(/\\r\\n/).map((line, index) => {
              return <p key={index}>{line}</p>;
            })}
          </div>
        ) : activeResult === "calibration" ? (
          <div className="results-area" id="calibration">
            {console.log(calibrationOutput.split(/\\r\\n/))}
            {calibrationOutput.split(/\\r\\n/).map((line, index) => {
              return <p key={index}>{line}</p>;
            })}
          </div>
        ) : (
          <p>Please select a controller option to see results.</p>
        )}
      </div>
    </div>
  );
};

export default ResultsSection;
