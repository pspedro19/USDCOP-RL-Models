import React, { useState, useEffect } from 'react';

const ImageViewer = ({ src, alt, caption, baseUrl }) => {
    const [imageUrl, setImageUrl] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(false);
    const [errorDetails, setErrorDetails] = useState('');

    useEffect(() => {
        if (!src) {
            setLoading(false);
            return;
        }

        // Build the final URL - normalize path and extract just the filename for flexible resolution
        let finalSrc = "";
        if (src && baseUrl) {
            let normalizedPath = src.replace(/\\/g, '/');
            // Remove common prefixes that don't match actual storage
            normalizedPath = normalizedPath.replace(/^results\//, '');
            normalizedPath = normalizedPath.replace(/^ml_pipeline\//, '');
            normalizedPath = normalizedPath.replace(/^weekly_update\//, '');
            // For forecast images with date folders (e.g., forecasts/2025-12-29/forward_forecast_ridge.png)
            // preserve the full path so the API can resolve the correct week's image
            // For backtest images, use just the filename
            if (normalizedPath.includes('forecasts/') && normalizedPath.match(/\d{4}-\d{2}-\d{2}/)) {
                // Keep full path for date-specific forecasts
                finalSrc = baseUrl + normalizedPath;
            } else {
                // Extract just the filename for backtests and other images
                const filename = normalizedPath.split('/').pop();
                finalSrc = baseUrl + filename;
            }
        } else if (src) {
            finalSrc = "/" + src.replace(/\\/g, '/');
        }

        // Fetch with authentication
        const fetchImage = async () => {
            try {
                setLoading(true);
                setError(false);

                // Get auth token from localStorage
                const token = localStorage.getItem('auth_token');

                const headers = {};
                if (token) {
                    headers['Authorization'] = `Bearer ${token}`;
                }

                const response = await fetch(finalSrc, { headers });

                if (!response.ok) {
                    console.error(`Image fetch failed: ${response.status} ${response.statusText} for ${finalSrc}`);
                    throw new Error(`HTTP ${response.status}`);
                }

                // Convert response to blob and create object URL
                const blob = await response.blob();
                const objectUrl = URL.createObjectURL(blob);
                setImageUrl(objectUrl);
                setLoading(false);
            } catch (err) {
                console.error('Error loading image:', err, finalSrc);
                setError(true);
                setErrorDetails(finalSrc);
                setLoading(false);
            }
        };

        fetchImage();

        // Cleanup: revoke object URL when component unmounts or src changes
        return () => {
            if (imageUrl) {
                URL.revokeObjectURL(imageUrl);
            }
        };
    }, [src, baseUrl]);

    if (!src) return null;

    // Show error state with details instead of returning null
    if (error) {
        return (
            <div className="glass-panel image-card">
                <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    minHeight: '200px',
                    padding: '20px',
                    color: 'var(--text-secondary)',
                    textAlign: 'center'
                }}>
                    <div style={{ fontSize: '2em', marginBottom: '10px' }}>📊</div>
                    <div>No se pudo cargar la imagen</div>
                    <div style={{ fontSize: '0.8em', marginTop: '8px', opacity: 0.7 }}>
                        {src.split('/').pop()}
                    </div>
                </div>
            </div>
        );
    }

    if (loading) {
        return (
            <div className="glass-panel image-card">
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    minHeight: '200px',
                    color: 'var(--text-secondary)'
                }}>
                    Cargando imagen...
                </div>
            </div>
        );
    }

    return (
        <div className="glass-panel image-card">
            <img
                src={imageUrl}
                alt={alt || "Dashboard Image"}
                onError={() => setError(true)}
                loading="lazy"
            />
            {caption && <div className="image-caption">{caption}</div>}
        </div>
    );
};

export default ImageViewer;
