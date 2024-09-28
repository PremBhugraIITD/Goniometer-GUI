import React from "react";
import {Link} from "react-router-dom";
import "./ControllerSection.css";

const ControllerSection = () => {
    return <div className="controller-container">
        <div className="button-area">
            <h2>Controller Choice</h2>
            <Link>Button 1</Link>
            <Link>Button 2</Link>
            <Link>Button 3</Link>
        </div>
        <div className="input-area"></div>
    </div>;
};

export default ControllerSection;