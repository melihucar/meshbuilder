import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import MeshViewer from './components/MeshViewer'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { cn } from '@/lib/utils'

type Status = 'idle' | 'uploading' | 'processing' | 'success' | 'error'
type Mode = 'single' | 'multi'
type Quality = 'draft' | 'standard' | 'high'
type Format = 'glb' | 'obj' | 'ply'

interface GenerationResult {
  meshUrl: string
  meshId: string
}

const API_URL = import.meta.env.DEV ? 'http://localhost:8000' : ''

function App() {
  const [mode, setMode] = useState<Mode>('single')
  const [images, setImages] = useState<File[]>([])
  const [imagePreviews, setImagePreviews] = useState<string[]>([])
  const [status, setStatus] = useState<Status>('idle')
  const [statusMessage, setStatusMessage] = useState('')
  const [result, setResult] = useState<GenerationResult | null>(null)
  const [selectedFormat, setSelectedFormat] = useState<Format>('glb')
  const [selectedQuality, setSelectedQuality] = useState<Quality>('standard')
  const [autoRotate, setAutoRotate] = useState(true)
  const [wireframe, setWireframe] = useState(false)

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (mode === 'single') {
      const file = acceptedFiles[0]
      if (file) {
        setImages([file])
        setImagePreviews([URL.createObjectURL(file)])
      }
    } else {
      const newImages = [...images, ...acceptedFiles].slice(0, 12)
      setImages(newImages)
      setImagePreviews(newImages.map(f => URL.createObjectURL(f)))
    }
    setStatus('idle')
    setResult(null)
  }, [mode, images])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/webp': ['.webp']
    },
    maxFiles: mode === 'single' ? 1 : 12
  })

  const handleGenerate = async () => {
    if (images.length === 0) return

    setStatus('uploading')
    setStatusMessage('Uploading images...')

    try {
      const formData = new FormData()

      if (mode === 'single') {
        formData.append('image', images[0])
      } else {
        images.forEach(img => {
          formData.append('images', img)
        })
      }

      setStatus('processing')
      setStatusMessage(
        mode === 'single'
          ? 'Generating 3D mesh... This may take a minute.'
          : `Processing ${images.length} images... This may take a few minutes.`
      )

      const endpoint = mode === 'single' ? '/generate' : '/generate-multiview'
      const params = new URLSearchParams({ format: selectedFormat })
      if (mode === 'single') {
        params.append('quality', selectedQuality)
      }
      const response = await fetch(`${API_URL}${endpoint}?${params}`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Generation failed')
      }

      const data = await response.json()

      setResult({
        meshUrl: `${API_URL}${data.mesh_url}`,
        meshId: data.mesh_id
      })
      setStatus('success')
      setStatusMessage('Mesh generated successfully!')

    } catch (err) {
      setStatus('error')
      setStatusMessage(err instanceof Error ? err.message : 'An error occurred')
    }
  }

  const handleClear = () => {
    setImages([])
    setImagePreviews([])
    setStatus('idle')
    setResult(null)
  }

  const handleRemoveImage = (index: number) => {
    const newImages = images.filter((_, i) => i !== index)
    const newPreviews = imagePreviews.filter((_, i) => i !== index)
    setImages(newImages)
    setImagePreviews(newPreviews)
  }

  const handleModeChange = (newMode: string) => {
    setMode(newMode as Mode)
    handleClear()
  }

  const handleDownload = () => {
    if (result?.meshUrl) {
      window.open(result.meshUrl, '_blank')
    }
  }

  const hasImages = images.length > 0

  return (
    <div className="flex min-h-screen bg-background">
      {/* Sidebar */}
      <aside className="w-[400px] bg-card border-r border-border p-6 flex flex-col gap-6 overflow-y-auto">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <span className="text-2xl">🎨</span>
          <h1 className="text-xl font-semibold text-foreground">Mesh Builder</h1>
        </div>

        {/* Mode Toggle */}
        <Tabs value={mode} onValueChange={handleModeChange} className="w-full">
          <TabsList className="grid w-full grid-cols-2 bg-muted">
            <TabsTrigger value="single">Single Image</TabsTrigger>
            <TabsTrigger value="multi">Multi-View</TabsTrigger>
          </TabsList>
        </Tabs>

        {!hasImages ? (
          <>
            {/* Dropzone */}
            <Card
              {...getRootProps()}
              className={cn(
                "border-2 border-dashed cursor-pointer transition-all bg-muted/50 hover:bg-muted hover:border-primary",
                isDragActive && "border-primary bg-primary/10"
              )}
            >
              <CardContent className="flex flex-col items-center justify-center py-10 px-6">
                <input {...getInputProps()} />
                <div className="text-4xl mb-4">{mode === 'single' ? '📷' : '📸'}</div>
                <h3 className="font-medium text-foreground mb-2">
                  {mode === 'single' ? 'Drop your image here' : 'Drop multiple images here'}
                </h3>
                <p className="text-sm text-muted-foreground">or click to browse (PNG, JPG, WEBP)</p>
                {mode === 'multi' && (
                  <Badge variant="secondary" className="mt-3">2-12 images from different angles</Badge>
                )}
              </CardContent>
            </Card>

            {/* Instructions */}
            <Card className="bg-muted/50">
              <CardContent className="py-4">
                <h3 className="text-sm font-medium mb-3 text-foreground">How it works</h3>
                {mode === 'single' ? (
                  <ol className="text-sm text-muted-foreground space-y-2 list-decimal list-inside">
                    <li>Upload an image of an object</li>
                    <li>AI generates a 3D mesh</li>
                    <li>View and download your 3D model</li>
                  </ol>
                ) : (
                  <ol className="text-sm text-muted-foreground space-y-2 list-decimal list-inside">
                    <li>Upload 4-6 images from different angles</li>
                    <li>Front, back, left, right views work best</li>
                    <li>AI fuses views into accurate 3D mesh</li>
                  </ol>
                )}
              </CardContent>
            </Card>
          </>
        ) : (
          <div className="flex flex-col gap-4">
            {/* Image Preview */}
            {mode === 'single' ? (
              <img
                src={imagePreviews[0]}
                alt="Preview"
                className="w-full rounded-lg border border-border"
              />
            ) : (
              <div className="grid grid-cols-3 gap-2">
                {imagePreviews.map((preview, index) => (
                  <div key={index} className="relative aspect-square rounded-lg overflow-hidden border border-border group">
                    <img src={preview} alt={`View ${index + 1}`} className="w-full h-full object-cover" />
                    <button
                      className="absolute top-1 right-1 w-5 h-5 rounded-full bg-black/70 text-white text-xs flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                      onClick={() => handleRemoveImage(index)}
                    >
                      ×
                    </button>
                    <span className="absolute bottom-1 left-1 bg-black/70 text-white text-[10px] px-1.5 py-0.5 rounded">
                      {index + 1}
                    </span>
                  </div>
                ))}
                {images.length < 12 && (
                  <div
                    {...getRootProps()}
                    className="aspect-square border-2 border-dashed border-border rounded-lg flex items-center justify-center cursor-pointer hover:border-primary hover:text-primary transition-colors text-muted-foreground text-sm"
                  >
                    <input {...getInputProps()} />
                    + Add
                  </div>
                )}
              </div>
            )}

            {mode === 'multi' && (
              <p className="text-sm text-muted-foreground text-center">
                {images.length} image{images.length !== 1 ? 's' : ''} selected
                {images.length < 2 && <span className="text-destructive"> (need at least 2)</span>}
              </p>
            )}

            {/* Quality Selection (Single mode only) */}
            {mode === 'single' && (
              <div className="space-y-2">
                <label className="text-xs text-muted-foreground uppercase tracking-wide">Quality</label>
                <div className="flex gap-2">
                  {(['draft', 'standard', 'high'] as const).map(quality => (
                    <Button
                      key={quality}
                      variant={selectedQuality === quality ? 'default' : 'outline'}
                      size="sm"
                      className="flex-1"
                      onClick={() => setSelectedQuality(quality)}
                    >
                      {quality.charAt(0).toUpperCase() + quality.slice(1)}
                    </Button>
                  ))}
                </div>
              </div>
            )}

            {/* Format Selection */}
            <div className="space-y-2">
              <label className="text-xs text-muted-foreground uppercase tracking-wide">Format</label>
              <div className="flex gap-2">
                {(['glb', 'obj', 'ply'] as const).map(format => (
                  <Button
                    key={format}
                    variant={selectedFormat === format ? 'default' : 'outline'}
                    size="sm"
                    className="flex-1"
                    onClick={() => setSelectedFormat(format)}
                  >
                    {format.toUpperCase()}
                  </Button>
                ))}
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-3">
              <Button
                className="flex-1"
                onClick={handleGenerate}
                disabled={
                  status === 'uploading' ||
                  status === 'processing' ||
                  (mode === 'multi' && images.length < 2)
                }
              >
                {status === 'processing' ? (
                  <>
                    <span className="w-4 h-4 border-2 border-transparent border-t-current rounded-full animate-spin" />
                    Generating...
                  </>
                ) : (
                  'Generate 3D Mesh'
                )}
              </Button>
              <Button variant="outline" onClick={handleClear}>
                Clear
              </Button>
            </div>
          </div>
        )}

        {/* Status */}
        {status !== 'idle' && (
          <Card className={cn(
            "border-none",
            status === 'processing' && "bg-primary/10 text-primary",
            status === 'success' && "bg-green-500/10 text-green-500",
            status === 'error' && "bg-destructive/10 text-destructive"
          )}>
            <CardContent className="py-3 flex items-center gap-3">
              {status === 'processing' && (
                <span className="w-4 h-4 border-2 border-transparent border-t-current rounded-full animate-spin" />
              )}
              {status === 'success' && <span>✓</span>}
              {status === 'error' && <span>✗</span>}
              <span className="text-sm">{statusMessage}</span>
            </CardContent>
          </Card>
        )}

        {status === 'processing' && (
          <Progress value={33} className="h-1" />
        )}

        {/* Download */}
        {result && (
          <div className="mt-auto pt-6 border-t border-border">
            <h3 className="text-sm text-muted-foreground mb-3">Download</h3>
            <Button className="w-full" onClick={handleDownload}>
              Download {selectedFormat.toUpperCase()}
            </Button>
          </div>
        )}
      </aside>

      {/* Main Viewer */}
      <main className="flex-1 relative bg-background">
        {result?.meshUrl ? (
          <>
            <MeshViewer url={result.meshUrl} autoRotate={autoRotate} wireframe={wireframe} />

            {/* Viewer Controls */}
            <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex gap-2 bg-card/90 backdrop-blur-sm p-2 rounded-lg border border-border">
              <Button
                variant={autoRotate ? 'default' : 'outline'}
                size="sm"
                onClick={() => setAutoRotate(!autoRotate)}
              >
                {autoRotate ? '⏸️ Pause' : '▶️ Rotate'}
              </Button>
              <Button
                variant={wireframe ? 'default' : 'outline'}
                size="sm"
                onClick={() => setWireframe(!wireframe)}
              >
                🔲 Wireframe
              </Button>
            </div>
          </>
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground">
            <div className="text-6xl mb-4 opacity-50">🧊</div>
            <p>Your 3D model will appear here</p>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
