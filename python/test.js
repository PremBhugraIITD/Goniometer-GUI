import express from "express";
import { exec } from "child_process";
const app = express();



app.listen(3001, () => {
  console.log("Server is running on port 3000");
});
