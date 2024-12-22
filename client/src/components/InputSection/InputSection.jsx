import React from "react";
import "./InputSection.css";

const InputSection = () => {
  return (
    <div className="input-container">
      <h2>Goniometer Input</h2>
      <form>
        <label id="input1" htmlFor="input1">
          Input 1
        </label>
        <input placeholder="Input 1" required />
        <label id="input2" htmlFor="input2">
          Input 2
        </label>
        <input placeholder="Input 2" required />
        <button type="submit">Submit</button>
      </form>
    </div>
  );
};

export default InputSection;
