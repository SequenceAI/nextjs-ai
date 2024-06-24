// ./app/page.js

"use client";

import React, { useState } from "react";
import ObjectDetection from "@/components/object-detection";

const Home = () => {
  const [overlayStates, setOverlayStates] = useState([]); // State to manage overlay states

  const handleButtonClick = (purpose) => {
    if (overlayStates.includes(purpose)) {
      // If overlay state exists, remove it
      setOverlayStates(overlayStates.filter((state) => state !== purpose));
    } else {
      // If overlay state does not exist, add it
      setOverlayStates([...overlayStates, purpose]);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-8">
      <h1 className="gradient-title font-extrabold text-3xl md:text-6xl lg:text-8xl tracking-tighter md:px-6 text-center">
        Thief Detection Alarm
      </h1>
      <div className="flex space-x-4 mb-4">
        <button
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-700"
          onClick={() => handleButtonClick("model1")}
        >
          Start Model 1
        </button>
        <button
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-700"
          onClick={() => handleButtonClick("model2")}
        >
          Start Model 2
        </button>
        <button
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-700"
          onClick={() => handleButtonClick("model3")}
        >
          Start Model 3
        </button>
        <button
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-700"
          onClick={() => handleButtonClick("model4")}
        >
          Start Model 4
        </button>
      </div>
      <ObjectDetection overlayStates={overlayStates} />
    </main>
  );
};

export default Home;
