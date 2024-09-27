import React from "react";
import VideoSection from "../../components/VideoSection/VideoSection";
import ControllerSection from "../../components/ControllerSection/ControllerSection";
import "./Home.css";

const Home = () => {
    return <div className="home-container">
        <VideoSection />
        <ControllerSection />
    </div>;
};

export default Home;