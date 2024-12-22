import React, { useState } from "react";
import "./ControllerSection.css";

const ControllerSection = ({ onSelect }) => {
  const [selectedOption, setSelectedOption] = useState("");

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
        </select>
      </div>

      {selectedOption === "pendant-drop" && (
        <div className="input-fields">
          <input
            type="text"
            placeholder="Density"
            required
            className="input-field"
          />
          <input
            type="text"
            placeholder="Needle Diameter"
            required
            className="input-field"
          />
        </div>
      )}
    </div>
  );
};

export default ControllerSection;
