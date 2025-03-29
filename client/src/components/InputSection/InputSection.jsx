import React, { useState } from "react";
import "./InputSection.css";
import { use } from "react";

// const InputSection = () => {
//   return (
//     <div className="input-container">
//       <h2>Features and Functionality</h2>
//       <div className="input-scroll">
//       <ul>
//         <li>
//           <h3>Contact Angle Measurement:</h3> Accurately measures the contact
//           angle to study surface wettability.
//         </li>
//         <li>
//           <h3>Surface Tension Calculation:</h3> This feature measures the
//           surface tension of fluids with precision.
//         </li>
//         <li>
//           <h3>Hysteresis Calculation:</h3> This function assesses the difference
//           between advancing and receding contact angles.
//         </li>
//         <li>
//           <h3>User-Friendly Interface:</h3> Designed with an intuitive layout
//           for ease of use.
//         </li>
//         <li>
//           <h3>Real-Time Data Display:</h3> Provides immediate visual feedback on
//           measurements.
//         </li>
//       </ul>
//       </div>
//     </div>
//   );
// };

const InputSection = () => {
  const [g_code, setg_code] = useState({
    x: 0,
    y: 0,
    z: 0,
  });

  const request = (x, y, z) => {
    fetch(
      `https://blynk.cloud/external/api/update?token=o8pzTsuTKKa8oQ6Hgu3srDppbpGx3edc&V0=G0%20X${x}%20Y${y}%20Z${z}%20F100`
    )
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        console.log("Response: " + data);
      })
      .catch((err) => {
        console.log("Error: " + err);
      });
  };

  return (
    <div className="input-container">
      <h2>Hardware Control</h2>
      <div className="xyz-control">
        <div className="x control">
          <button
            onClick={() => {
              setg_code((prevValue) => {
                const updatedValue = {
                  ...prevValue,
                  x: prevValue.x + 1,
                };
                request(updatedValue.x, updatedValue.y, updatedValue.z);
                return updatedValue;
              });
            }}
          >
            ▲
          </button>
          <p>X: {g_code.x}</p>
          <button
            onClick={() => {
              setg_code((prevValue) => {
                const updatedValue = {
                  ...prevValue,
                  x: prevValue.x - 1,
                };
                request(updatedValue.x, updatedValue.y, updatedValue.z);
                return updatedValue;
              });
            }}
          >
            ▼
          </button>
        </div>
        <div className="y control">
          <button
            onClick={() => {
              setg_code((prevValue) => {
                const updatedValue = {
                  ...prevValue,
                  y: prevValue.y + 1,
                };
                request(updatedValue.x, updatedValue.y, updatedValue.z);
                return updatedValue;
              });
            }}
          >
            ▲
          </button>
          <p>Y: {g_code.y}</p>
          <button
            onClick={() => {
              setg_code((prevValue) => {
                const updatedValue = {
                  ...prevValue,
                  y: prevValue.y - 1,
                };
                request(updatedValue.x, updatedValue.y, updatedValue.z);
                return updatedValue;
              });
            }}
          >
            ▼
          </button>
        </div>
        <div className="z control">
          <button
            onClick={() => {
              setg_code((prevValue) => {
                const updatedValue = {
                  ...prevValue,
                  z: prevValue.z + 1,
                };
                request(updatedValue.x, updatedValue.y, updatedValue.z);
                return updatedValue;
              });
            }}
          >
            ▲
          </button>
          <p>Z: {g_code.z}</p>
          <button
            onClick={() => {
              setg_code((prevValue) => {
                const updatedValue = {
                  ...prevValue,
                  z: prevValue.z - 1,
                };
                request(updatedValue.x, updatedValue.y, updatedValue.z);
                return updatedValue;
              });
            }}
          >
            ▼
          </button>
        </div>
      </div>
    </div>
  );
};

export default InputSection;
