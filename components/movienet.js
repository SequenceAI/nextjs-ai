"use client";

import React, { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";

const MovinetDetection = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [model, setModel] = useState(null);
  const [labels, setLabels] = useState([]);
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [frames, setFrames] = useState([]);
  const [initialStates, setInitialStates] = useState({});
  const [currentStates, setCurrentStates] = useState({});

  useEffect(() => {
    const loadModel = async () => {
      setIsLoading(true);
      try {
        // Load labels
        const response = await fetch(
          "https://raw.githubusercontent.com/tensorflow/models/f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/official/projects/movinet/files/kinetics_600_labels.txt"
        );
        const text = await response.text();
        const loadedLabels = text.split("\n").map((label) => label.trim());
        setLabels(loadedLabels);

        // Load MoViNet model
        const loadedModel = await tf.loadGraphModel("/model.json");

        // Initialize state tensors
        const initialStates = {};
        loadedModel.inputs.forEach((input) => {
          if (input.name !== "image") {
            const shape = input.shape.map((dim) => (dim === -1 ? 1 : dim));
            const dtype = input.dtype === "int32" ? "int32" : "float32";
            initialStates[input.name] = tf.zeros(shape, dtype);
          }
        });

        setInitialStates(initialStates);
        setCurrentStates(initialStates);
        setModel(loadedModel);
      } catch (error) {
        console.error("Failed to load model:", error);
      } finally {
        setIsLoading(false);
      }
    };

    loadModel();
  }, []);

  const preprocessFrame = (frame) => {
    return tf.tidy(() => {
      let tensor = tf.browser.fromPixels(frame);
      tensor = tf.image.resizeBilinear(tensor, [224, 224]);
      tensor = tensor.expandDims(0).toFloat().div(tf.scalar(255));
      return tensor;
    });
  };

  const drawPredictions = (frame, predictions) => {
    const ctx = canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    predictions.forEach((prediction, i) => {
      const text = `${prediction.className}: ${prediction.probability.toFixed(2)}`;
      ctx.fillStyle = "red";
      ctx.font = "18px Arial";
      ctx.fillText(text, 10, 50 + i * 30);
    });
  };

  const processFrame = async () => {
    if (
      webcamRef.current &&
      webcamRef.current.video.readyState === 4 &&
      model
    ) {
      const video = webcamRef.current.video;
      const frame = preprocessFrame(video);

      setFrames((prevFrames) => {
        const newFrames = [...prevFrames, frame];
        if (newFrames.length > 64) {
          newFrames.shift();
        }
        return newFrames;
      });

      if (frames.length === 64) {
        const input = tf.stack(frames);
        const inputs = { image: input };

        Object.keys(currentStates).forEach((key) => {
          const currentState = currentStates[key];
          const reshapedState = tf.reshape(currentState, [-1]);
          inputs[key] = reshapedState;
        });

        const outputs = await model.executeAsync(inputs);
        const predictions = outputs[0];
        const newStates = outputs.slice(1);

        const topK = tf.topk(predictions, 5).indices.arraySync()[0];
        const topKPredictions = topK.map((idx) => ({
          className: labels[idx],
          probability: predictions.arraySync()[0][idx],
        }));

        drawPredictions(video, topKPredictions);
        tf.dispose(input);

        // Update state tensors
        const updatedStates = {};
        newStates.forEach((state, i) => {
          const key = Object.keys(currentStates)[i];
          updatedStates[key] = state;
        });
        setCurrentStates(updatedStates);
      }
    }
  };

  useEffect(() => {
    const interval = setInterval(processFrame, 100);
    return () => clearInterval(interval);
  }, [model, frames, currentStates]);

  return (
    <div className="mt-8">
      {isLoading ? (
        <div className="gradient-text">Loading AI Model...</div>
      ) : (
        <div className="relative flex justify-center items-center gradient p-1.5 rounded-md">
          <Webcam
            ref={webcamRef}
            className="rounded-md w-full lg:h-[720px]"
            muted
          />
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 z-99999 w-full lg:h-[720px]"
          />
        </div>
      )}
    </div>
  );
};

export default MovinetDetection;
