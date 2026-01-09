'use client';

import { useState, useEffect } from 'react';
import { ImageIcon, AlertCircle, Loader2 } from 'lucide-react';

interface ForecastingImageViewerProps {
  src: string | null;
  caption?: string;
  alt?: string;
}

export function ForecastingImageViewer({ src, caption, alt }: ForecastingImageViewerProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [imageSrc, setImageSrc] = useState<string | null>(null);

  useEffect(() => {
    // Reset states when src changes
    setLoading(true);
    setError(false);

    if (!src || src.trim() === '') {
      setImageSrc(null);
      setLoading(false);
      return;
    }

    // Build the image URL - all images are in /forecasting/
    // Extract just the filename if it has a path
    const filename = src.split('/').pop() || src;
    const finalSrc = `/forecasting/${filename}`;

    setImageSrc(finalSrc);
  }, [src]);

  // No source provided
  if (!src || src.trim() === '') {
    return (
      <div className="flex items-center justify-center h-64 bg-slate-900/50 rounded-xl border border-slate-800">
        <div className="text-center text-gray-500">
          <ImageIcon className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p className="text-sm">Selecciona un modelo y horizonte para ver el grafico</p>
        </div>
      </div>
    );
  }

  // Error loading image
  if (error) {
    return (
      <div className="flex items-center justify-center h-64 bg-slate-900/50 rounded-xl border border-slate-800">
        <div className="text-center text-gray-500">
          <AlertCircle className="w-12 h-12 mx-auto mb-3 text-red-400 opacity-70" />
          <p className="text-sm">No se pudo cargar la imagen</p>
          <p className="text-xs mt-2 opacity-60 font-mono">{src.split('/').pop()}</p>
        </div>
      </div>
    );
  }

  // Image source not ready yet
  if (!imageSrc) {
    return (
      <div className="flex items-center justify-center h-64 bg-slate-900/50 rounded-xl border border-slate-800">
        <div className="flex items-center gap-3 text-gray-400">
          <Loader2 className="w-6 h-6 animate-spin text-purple-500" />
          <span className="text-sm">Preparando imagen...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="relative bg-slate-900/50 rounded-xl border border-slate-800 overflow-hidden">
      {/* Loading overlay */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-900/80 z-10">
          <div className="flex items-center gap-3 text-gray-400">
            <Loader2 className="w-6 h-6 animate-spin text-purple-500" />
            <span className="text-sm">Cargando imagen...</span>
          </div>
        </div>
      )}

      {/* Image container */}
      <div className="relative w-full" style={{ minHeight: '400px' }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={imageSrc}
          alt={alt || caption || 'Forecast Chart'}
          className="w-full h-auto object-contain"
          style={{ maxHeight: '600px' }}
          onLoad={() => setLoading(false)}
          onError={() => {
            setLoading(false);
            setError(true);
          }}
        />
      </div>

      {/* Caption */}
      {caption && (
        <div className="px-4 py-3 bg-slate-800/50 border-t border-slate-700">
          <p className="text-sm text-gray-300 text-center font-medium">{caption}</p>
        </div>
      )}
    </div>
  );
}
