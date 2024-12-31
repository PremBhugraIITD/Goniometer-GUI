import React, { useEffect, useState } from "react";
import download_icon from "../../assets/download_icon.svg";
import "./ResultsSection.css";

const ResultsSection = ({ activeResult }) => {
  const [sessileDropOutput, setSessileDropOutput] = useState("");
  const [pendantDropOutput, setPendantDropOutput] = useState("");
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
          <div className="results-area" id="results-type-one">
            <h3>Results Ready:</h3>
            <div className="download">
              <img src={download_icon} alt="Download" />
              <p>Export CSV</p>
            </div>
          </div>
        ) : activeResult === "pendant-drop" ? (
          <div className="results-area" id="pendant-drop-analysis">
            {console.log(pendantDropOutput.split(/\\r\\n/))}
            {pendantDropOutput.split(/\\r\\n/).map((line, index) => {
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
