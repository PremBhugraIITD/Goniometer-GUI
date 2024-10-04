import React from "react";
import { Link } from "react-router-dom";
import "./ControllerSection.css";

const ControllerSection = ({ onSelect }) => {
  return (
    <div className="controller-container">
      <h2>Controller Choice</h2>
      <div className="buttons">
        <div className="btn">
          <Link onClick={() => onSelect("hysteresis")}>Hysteresis</Link>
          <input placeholder="Input" required />
        </div>
        <div className="btn">
          <Link onClick={() => onSelect("pendant-drop")}>Pendant Drop</Link>
          <input placeholder="Input" required />
        </div>
        <div className="btn">
          <Link onClick={() => onSelect("sessile-drop")}>Sessile Drop</Link>
          <input placeholder="Input" required />
        </div>
      </div>
    </div>
  );
};

export default ControllerSection;
