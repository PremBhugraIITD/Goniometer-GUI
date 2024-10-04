import React,{useState} from "react";
import VideoSection from "../../components/VideoSection/VideoSection";
import ControllerSection from "../../components/ControllerSection/ControllerSection";
import InputSection from "../../components/InputSection/InputSection";
import ResultsSection from "../../components/ResultsSection/ResultsSection";
import "./Home.css";

const Home = () => {
    const [activeResult, setActiveResult] = useState(null);

  const handleSelection = (selection) => {
    setActiveResult(selection);
  };
  return (
    <div className="home-container">
      <VideoSection />
      <ControllerSection onSelect={handleSelection}/>
      <ResultsSection activeResult={activeResult}/>
      <InputSection />
    </div>
  );
};

export default Home;
