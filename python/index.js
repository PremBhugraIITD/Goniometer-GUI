import express from "express";
import multer from "multer";
import { spawn } from "child_process";
import fs from "fs";

const app = express();
const PORT = 3000;

// Configure multer for file uploads
const upload = multer({
  dest: "uploads/", // Temporary upload location
});

app.post("/sessile-drop", upload.single("image"), (req, res) => {
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
  ]); // For Windows
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
  console.log("Sessile Drop entered");
  res.json({ message: "Execution Completed" });
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
