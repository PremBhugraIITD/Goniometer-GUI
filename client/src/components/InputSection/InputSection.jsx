import React from "react";
import "./InputSection.css";

const InputSection = () => {
  return (
    <div className="input-container">
      <h2>Features and Functionality</h2>
      <div className="input-scroll">
      <ul>
        <li>
          <h3>Contact Angle Measurement:</h3> Accurately measures the contact
          angle to study surface wettability.
        </li>
        <li>
          <h3>Surface Tension Calculation:</h3> This feature measures the
          surface tension of fluids with precision.
        </li>
        <li>
          <h3>Hysteresis Calculation:</h3> This function assesses the difference
          between advancing and receding contact angles.
        </li>
        <li>
          <h3>User-Friendly Interface:</h3> Designed with an intuitive layout
          for ease of use.
        </li>
        <li>
          <h3>Real-Time Data Display:</h3> Provides immediate visual feedback on
          measurements.
        </li>
      </ul>
      </div>
    </div>
  );
};

export default InputSection;
