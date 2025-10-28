import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, RotateCw, ZoomIn, ZoomOut, FileImage, AlertCircle } from 'lucide-react';
import DragDropZone from './DragDropZone';
import { createImagePreview, revokeImagePreview, generateThumbnail } from 'services/imageProcessing';
import { formatFileSize, formatDate } from 'utils/formatters';
import LoadingSpinner from 'components/common/LoadingSpinner';

interface ImageUploadProps {
  onImageProcess: (file: File) => void;
  onError: (error: string) => void;
  isProcessing?: boolean;
  className?: string;
}

interface ImagePreview {
  file: File;
  preview: string;
  thumbnail: string;
  dimensions?: { width: number; height: number };
}

const ImageUpload: React.FC<ImageUploadProps> = ({
  onImageProcess,
  onError,
  isProcessing = false,
  className = ''
}) => {
  const [selectedImage, setSelectedImage] = useState<ImagePreview | null>(null);
  const [isGeneratingThumbnail, setIsGeneratingThumbnail] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotation] = useState(0);

  const handleFileSelect = useCallback(async (file: File) => {
    try {
      setIsGeneratingThumbnail(true);
      
      // Create preview URLs
      const preview = createImagePreview(file);
      const thumbnail = await generateThumbnail(file);
      
      // Get image dimensions
      const img = new Image();
      img.onload = () => {
        setSelectedImage({
          file,
          preview,
          thumbnail,
          dimensions: {
            width: img.naturalWidth,
            height: img.naturalHeight
          }
        });
        setIsGeneratingThumbnail(false);
      };
      img.src = preview;
      
    } catch (error) {
      console.error('Error processing image:', error);
      onError('Failed to process image preview');
      setIsGeneratingThumbnail(false);
    }
  }, [onError]);

  const handleRemoveImage = useCallback(() => {
    if (selectedImage) {
      revokeImagePreview(selectedImage.preview);
      setSelectedImage(null);
      setZoom(1);
      setRotation(0);
    }
  }, [selectedImage]);

  const handleProcessImage = useCallback(() => {
    if (selectedImage) {
      onImageProcess(selectedImage.file);
    }
  }, [selectedImage, onImageProcess]);

  const handleZoomIn = useCallback(() => {
    setZoom(prev => Math.min(prev + 0.25, 3));
  }, []);

  const handleZoomOut = useCallback(() => {
    setZoom(prev => Math.max(prev - 0.25, 0.5));
  }, []);

  const handleRotate = useCallback(() => {
    setRotation(prev => (prev + 90) % 360);
  }, []);

  return (
    <div className={`space-y-6 ${className}`}>
      {!selectedImage ? (
        <DragDropZone
          onFileSelect={handleFileSelect}
          onError={onError}
          disabled={isProcessing}
        />
      ) : (
        <AnimatePresence>
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.3 }}
            className="bg-white rounded-lg border border-gray-200 overflow-hidden"
          >
            {/* Image Preview Header */}
            <div className="bg-gray-50 px-6 py-4 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <FileImage className="w-5 h-5 text-gray-500" />
                  <div>
                    <h3 className="font-medium text-gray-900">{selectedImage.file.name}</h3>
                    <p className="text-sm text-gray-500">
                      {formatFileSize(selectedImage.file.size)} • {formatDate(selectedImage.file.lastModified.toString())}
                    </p>
                  </div>
                </div>
                <button
                  onClick={handleRemoveImage}
                  className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                  disabled={isProcessing}
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Image Preview */}
            <div className="relative bg-gray-100 p-6">
              <div className="flex items-center justify-center min-h-[400px]">
                {isGeneratingThumbnail ? (
                  <LoadingSpinner size="lg" text="Processing image..." />
                ) : (
                  <motion.div
                    animate={{ 
                      scale: zoom,
                      rotate: rotation
                    }}
                    transition={{ duration: 0.3 }}
                    className="relative max-w-full max-h-full"
                  >
                    <img
                      src={selectedImage.preview}
                      alt="Selected image preview"
                      className="max-w-full max-h-[500px] object-contain rounded-lg shadow-lg"
                    />
                  </motion.div>
                )}
              </div>

              {/* Image Controls */}
              <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2">
                <div className="flex items-center space-x-2 bg-white rounded-lg shadow-lg border border-gray-200 p-2">
                  <button
                    onClick={handleZoomOut}
                    className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded transition-colors"
                    disabled={isProcessing}
                  >
                    <ZoomOut className="w-4 h-4" />
                  </button>
                  
                  <span className="px-3 py-1 text-sm font-medium text-gray-700 bg-gray-100 rounded">
                    {Math.round(zoom * 100)}%
                  </span>
                  
                  <button
                    onClick={handleZoomIn}
                    className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded transition-colors"
                    disabled={isProcessing}
                  >
                    <ZoomIn className="w-4 h-4" />
                  </button>
                  
                  <div className="w-px h-6 bg-gray-300 mx-1" />
                  
                  <button
                    onClick={handleRotate}
                    className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded transition-colors"
                    disabled={isProcessing}
                  >
                    <RotateCw className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>

            {/* Image Info */}
            {selectedImage.dimensions && (
              <div className="bg-gray-50 px-6 py-4 border-t border-gray-200">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">Dimensions:</span>
                    <p className="font-medium">
                      {selectedImage.dimensions.width} × {selectedImage.dimensions.height}
                    </p>
                  </div>
                  <div>
                    <span className="text-gray-500">File Size:</span>
                    <p className="font-medium">{formatFileSize(selectedImage.file.size)}</p>
                  </div>
                  <div>
                    <span className="text-gray-500">Format:</span>
                    <p className="font-medium">{selectedImage.file.type}</p>
                  </div>
                  <div>
                    <span className="text-gray-500">Modified:</span>
                    <p className="font-medium">
                      {formatDate(selectedImage.file.lastModified.toString())}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="bg-white px-6 py-4 border-t border-gray-200">
              <div className="flex items-center justify-between">
                <button
                  onClick={handleRemoveImage}
                  className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                  disabled={isProcessing}
                >
                  Choose Different Image
                </button>
                
                <button
                  onClick={handleProcessImage}
                  disabled={isProcessing}
                  className="btn-primary flex items-center space-x-2"
                >
                  {isProcessing ? (
                    <>
                      <LoadingSpinner size="sm" color="white" />
                      <span>Processing...</span>
                    </>
                  ) : (
                    <span>Analyze Image</span>
                  )}
                </button>
              </div>
            </div>
          </motion.div>
        </AnimatePresence>
      )}
    </div>
  );
};

export default ImageUpload;
