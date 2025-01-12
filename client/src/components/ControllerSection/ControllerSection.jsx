import React, { useState } from "react";
import "./ControllerSection.css";

const ControllerSection = ({ onSelect, onDensityAndDiameterSubmit }) => {
  const [selectedOption, setSelectedOption] = useState("");
  const [density, setDensity] = useState("");
  const [needleDiameter, setNeedleDiameter] = useState("");

  const handleSubmit = () => {
    if (onDensityAndDiameterSubmit) {
      onDensityAndDiameterSubmit(density, needleDiameter);
    }
  };

  const handleDropdownChange = (event) => {
    const selectedValue = event.target.value;
    setSelectedOption(selectedValue);
    onSelect(selectedValue);
  };

  return (
    <div className="controller-container">
      <h2>Controller Choice</h2>
      <div className="dropdown">
        <select
          value={selectedOption}
          onChange={handleDropdownChange}
          className="dropdown-select"
        >
          <option value="" disabled>
            Select an orientation
          </option>
          <option value="hysteresis">Hysteresis</option>
          <option value="pendant-drop">Pendant Drop</option>
          <option value="sessile-drop">Sessile Drop</option>
          <option value="calibration">Calibration</option>
        </select>
      </div>

      {selectedOption === "pendant-drop" && (
        <div className="input-fields">
          <input
            type="text"
            placeholder="Density"
            value={density}
            onChange={(e) => setDensity(e.target.value)}
            required
            className="input-field"
          />
          <input
            type="text"
            placeholder="Needle Diameter"
            value={needleDiameter}
            onChange={(e) => setNeedleDiameter(e.target.value)}
            required
            className="input-field"
          />
          <button onClick={handleSubmit} className="submit-button">
            Submit
          </button>
        </div>
      )}
    </div>
  );
};

export default ControllerSection;
