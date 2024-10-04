import React from "react";
import download_icon from "../../assets/download_icon.svg";
import "./ResultsSection.css";

const ResultsSection = ({ activeResult }) => {
  return (
    <div className="results-container">
      <h2>Results Section</h2>

      {activeResult === "hysteresis" || activeResult === "sessile-drop" ? (
        <div className="results-area" id="results-type-one">
          <h3>Results Ready:</h3>
          <div className="download">
            <img src={download_icon} alt="Download" />
            <p>Export CSV</p>
          </div>
        </div>
      ) : activeResult === "pendant-drop" ? (
        <div className="results-area" id="results-type-two">
          <h3>Results Ready:</h3>
          <div className="output">
            <p>Output 1:</p>
            <p>Output 2:</p>
          </div>
        </div>
      ) : (
        <p>Please select a controller option to see results.</p>
      )}
    </div>
  );
};

export default ResultsSection;
