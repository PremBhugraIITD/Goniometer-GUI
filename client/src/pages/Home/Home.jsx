import React, { useState } from "react";
import VideoSection from "../../components/VideoSection/VideoSection";
import ControllerSection from "../../components/ControllerSection/ControllerSection";
import InputSection from "../../components/InputSection/InputSection";
import ResultsSection from "../../components/ResultsSection/ResultsSection";
import "./Home.css";

const Home = () => {
  const [activeResult, setActiveResult] = useState(null);
  const [density, setDensity] = useState("");
  const [needleDiameter, setNeedleDiameter] = useState("");

  const handleDensityAndDiameterSubmit = (newDensity, newNeedleDiameter) => {
    setDensity(newDensity);
    setNeedleDiameter(newNeedleDiameter);
  };

  const handleSelection = (selection) => {
    setActiveResult(selection);
  };
  return (
    <div className="home-container">
      <VideoSection
        activeResult={activeResult}
        density={density}
        needleDiameter={needleDiameter}
      />
      <ControllerSection
        onSelect={handleSelection}
        onDensityAndDiameterSubmit={handleDensityAndDiameterSubmit}
      />
      <ResultsSection activeResult={activeResult} />
      <InputSection />
    </div>
  );
};

export default Home;
