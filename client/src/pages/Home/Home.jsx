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
  const [needleDiameter, setNeedleDiameter] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [csvError, setCSVError] = useState(true);

  const handleDensityAndDiameterSubmit = (newDensity, newNeedleDiameter) => {
    setDensity(newDensity);
    setNeedleDiameter(newNeedleDiameter);
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
        needleDiameter={needleDiameter}
        onProcessingChange={handleProcessingChange}
        onCSVError={handleCSVError}
      />
      <ControllerSection
        onSelect={handleSelection}
        onDensityAndDiameterSubmit={handleDensityAndDiameterSubmit}
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
