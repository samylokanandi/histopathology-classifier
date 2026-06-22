import React from 'react'
import UploadPredict from './components/UploadPredict'

export default function App() {
  return (
    <div className="page">
      <div className="header">
        <div className="eyebrow">Grad-CAM · ResNet18 · IDC Classification</div>
        <h1 className="title">Breast Histopathology Classifier</h1>
        <p className="subtitle">
          Upload a histopathology image patch and see both the model's prediction
          and a visual explanation of which regions drove that call.
        </p>
        <p className="notice">
          Note: the backend spins down when idle, so the first request after a
          while may take 60–90 seconds to wake up. Later requests are fast.
        </p>
      </div>

      <UploadPredict />

      <div className="footer-note">
        built by Samy Lokanandi · react · fastapi · pytorch ·{' '}
        <a href="https://github.com/samylokanandi/histopathology-classifier" target="_blank" rel="noreferrer">
          source on github
        </a>
      </div>
    </div>
  )
}