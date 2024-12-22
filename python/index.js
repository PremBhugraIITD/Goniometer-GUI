import express from "express";
import multer from "multer";
import { spawn } from "child_process";
import fs from "fs";
import cors from "cors";

const app = express();
const PORT = 3000;

app.use(
  cors({
    origin: "http://localhost:5173", // Replace with your React app's origin
    methods: ["GET", "POST"],
  })
);

// Configure multer for file uploads
const upload = multer({
  dest: "uploads/", // Temporary upload location
});

app.post("/sessile-drop", upload.single("image"), (req, res) => {
  console.log("Sessile Drop entered");
  const tempPath = req.file.path;
  const targetPath =
    "c:/Users/Prem/OneDrive - IIT Delhi/Desktop/screenshot.png";

  // Move the file to the desired location
  fs.rename(tempPath, targetPath, (err) => {
    if (err) {
      console.error("Error saving image:", err);
      return res.status(500).json({ message: "Failed to save image." });
    }
    console.log("Image uploaded successfully");
  });
  const pythonProcess = spawn("C://Python312//python.exe", [
    "Sessile_ossila_algo.py",
  ]);
  pythonProcess.on("error", (error) => {
    console.error("Error starting Python script:", error);
    return res.status(500).json({ output: "Error running Python script." });
  });

  pythonProcess.on("close", (code) => {
    if (code !== 0) {
      console.error(`Python script exited with code ${code}`);
      return res
        .status(500)
        .json({ output: `Python script failed with code ${code}.` });
    }

    console.log("Python script completed successfully");
  });
  res.json({ message: "Execution Completed" });
});

app.get("/sessile-drop-stream", (req, res) => {
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
  
    const interval = setInterval(() => {
      fs.readFile("Static_Contact_Angle.txt", "utf8", (err, data) => {
        if (err) {
          console.error("Error reading file:", err);
          res.write("data: Error reading file\n\n");
          clearInterval(interval);
          return;
        }
        res.write(`data: ${JSON.stringify(data)}\n\n`);
      });
    }, 1000);
  
    req.on("close", () => {
      clearInterval(interval);
    });
  });

app.post("/pendant-drop", (req, res) => {
  console.log("Pendant Drop entered");

  res.json({ message: "Execution Completed" });
});

app.post("/hysteresis", (req, res) => {
  console.log("Hysteresis entered");

  res.json({ message: "Execution Completed" });
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
