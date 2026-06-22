import React, { useRef, useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function UploadPredict() {
  const [originalUrl, setOriginalUrl] = useState(null)
  const [heatmapUrl, setHeatmapUrl] = useState(null)
  const [view, setView] = useState('original') // 'original' | 'heatmap'
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  const handleFile = async (file) => {
    if (!file) return
    setError(null)
    setResult(null)
    setHeatmapUrl(null)
    setOriginalUrl(URL.createObjectURL(file))
    setView('original')
    setLoading(true)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Request failed (${res.status})`)
      }
      const data = await res.json()
      setResult(data)
      setHeatmapUrl(`data:image/png;base64,${data.heatmap_base64}`)
      setView('heatmap')
    } catch (err) {
      setError(err.message || 'Something went wrong reaching the model server.')
    } finally {
      setLoading(false)
    }
  }

  const onDrop = (e) => {
    e.preventDefault()
    handleFile(e.dataTransfer.files?.[0])
  }

  const isFlagged = result?.prediction === 'class_1'
  const displayLabel = result
    ? (isFlagged ? 'IDC pattern detected' : 'No IDC pattern detected')
    : null

  return (
    <div className="panel">
      <div className="panel-grid">
        <div className="scope">
          <div className="scope-label">Input patch</div>
          <div
            className={`dropzone ${originalUrl ? 'has-image' : ''}`}
            onClick={() => fileInputRef.current?.click()}
            onDrop={onDrop}
            onDragOver={(e) => e.preventDefault()}
          >
            {view === 'heatmap' && heatmapUrl ? (
              <img src={heatmapUrl} alt="Grad-CAM heatmap overlay" />
            ) : originalUrl ? (
              <img src={originalUrl} alt="Uploaded histopathology patch" />
            ) : (
              <div className="dropzone-prompt">
                <strong>Drop an image, or click to browse</strong>
                A histopathology patch (any size — it'll be resized automatically)
              </div>
            )}
            {loading && <div className="scan-line" />}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="file-input"
            onChange={(e) => handleFile(e.target.files?.[0])}
          />

          <div className="view-toggle">
            <button
              className={view === 'original' ? 'active' : ''}
              disabled={!originalUrl}
              onClick={() => setView('original')}
            >
              Original
            </button>
            <button
              className={view === 'heatmap' ? 'active' : ''}
              disabled={!heatmapUrl}
              onClick={() => setView('heatmap')}
            >
              Grad-CAM
            </button>
          </div>

          <button className="upload-btn" onClick={() => fileInputRef.current?.click()} disabled={loading}>
            {loading ? 'Analyzing…' : originalUrl ? 'Try another image' : 'Choose image'}
          </button>
        </div>

        <div className="readout">
          {error && <div className="error-banner">{error}</div>}

          {!result && !error && !loading && (
            <div className="readout-empty">
              Upload a histopathology patch to see the model's prediction and a
              Grad-CAM overlay showing which regions of the image drove that decision.
            </div>
          )}

          {loading && <div className="readout-empty">Running inference and computing Grad-CAM…</div>}

          {result && (
            <>
              <div className={`finding ${isFlagged ? 'flagged' : 'clear'}`}>
                <div className="finding-label">{isFlagged ? 'Flagged' : 'Clear'}</div>
                <div className="finding-class">{displayLabel}</div>
                <div className="finding-confidence">confidence: {(result.confidence * 100).toFixed(1)}%</div>
                <div className="bar-track">
                  <div className="bar-fill" style={{ width: `${result.confidence * 100}%` }} />
                </div>
              </div>

              <div className="explainer">
                <strong>What the heatmap shows:</strong> warmer regions (red/yellow)
                are the areas of the patch that most influenced this prediction,
                computed via Grad-CAM on the final convolutional block. This is a
                research/portfolio demo, not a diagnostic tool.
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
