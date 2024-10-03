import React from "react";
import { Link } from "react-router-dom";
import "./ControllerSection.css";

const ControllerSection = () => {
  return (
    <div className="controller-container">
      <div className="button-area">
        <h2>Controller Choice</h2>
        <Link>Button 1</Link>
        <Link>Button 2</Link>
        <Link>Button 3</Link>
      </div>
      <div className="input-area">
        <h2>Input Section</h2>
        <form>
          <label id="input1" for="input1">Input 1</label>
          <input placeholder="Input 1" required />

          <label id="input2" for="input2">Input 2</label>
          <input placeholder="Input 2" required />

          <button type="submit">
            Sign in
          </button>
        </form>
      </div>
    </div>
  );
};

export default ControllerSection;
