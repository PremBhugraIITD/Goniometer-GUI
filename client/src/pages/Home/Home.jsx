import React, { useState } from "react";
import fs from "fs";
import VideoSection from "../../components/VideoSection/VideoSection";
import ControllerSection from "../../components/ControllerSection/ControllerSection";
import InputSection from "../../components/InputSection/InputSection";
import ResultsSection from "../../components/ResultsSection/ResultsSection";
import "./Home.css";

const Home = () => {
  const [activeResult, setActiveResult] = useState(null);
  const [density, setDensity] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [csvError, setCSVError] = useState(true);

  const handleDensitySubmit = (newDensity) => {
    setDensity(newDensity);
  };

  const handleSelection = (selection) => {
    setActiveResult(selection);
  };

  const handleProcessingChange = (processing) => {
    setIsProcessing(processing);
  };

  const handleCSVError = (error) => {
    setCSVError(error);
  };

  return (
    <div className="home-container">
      <VideoSection
        activeResult={activeResult}
        density={density}
        onProcessingChange={handleProcessingChange}
        onCSVError={handleCSVError}
      />
      <ControllerSection
        onSelect={handleSelection}
        onDensitySubmit={handleDensitySubmit}
      />
      <ResultsSection
        activeResult={activeResult}
        isProcessing={isProcessing}
        csvError={csvError}
      />
      <InputSection />
    </div>
  );
};

export default Home;
