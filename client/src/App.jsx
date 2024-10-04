import React from "react";
import Navbar from "./components/Navbar/Navbar.jsx";
import Home from "./pages/Home/Home.jsx";
import Footer from "./components/Footer/Footer.jsx";
import "./App.css";

const App = () => {
  return (
    <div className="app-container">
      {/* <div className="nav-content">
        <Navbar />
      </div> */}
      <div className="page-content">
        <Home />
      </div>
      {/* <Footer /> */}
    </div>
  );
};

export default App;
