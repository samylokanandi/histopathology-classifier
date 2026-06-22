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
      </div>

      <UploadPredict />

      <div className="footer-note">
        built with react · fastapi · pytorch — source on github
      </div>
    </div>
  )
}
