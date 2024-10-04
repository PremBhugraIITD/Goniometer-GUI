import React from "react";
import { Link } from "react-router-dom";
import "./ControllerSection.css";

const ControllerSection = ({onSelect}) => {
  return (
    <div className="controller-container">
      <h2>Controller Choice</h2>
      <Link onClick={() => onSelect("hysteresis")}>Hysteresis</Link>
      <Link onClick={() => onSelect("pendant-drop")}>Pendant Drop</Link>
      <Link onClick={() => onSelect("sessile-drop")}>Sessile Drop</Link>
    </div>
  );
};

export default ControllerSection;
