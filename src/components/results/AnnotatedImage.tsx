import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ZoomIn, ZoomOut, RotateCw, Download, Eye, EyeOff, X } from 'lucide-react';
import { Disease } from 'types/detection';
import { DISEASE_TYPES } from 'utils/constants';

interface AnnotatedImageProps {
  originalImage: string;
  annotatedImage: string;
  diseases: Disease[];
  className?: string;
}

const AnnotatedImage: React.FC<AnnotatedImageProps> = ({
  originalImage,
  annotatedImage,
  diseases,
  className = ''
}) => {
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotation] = useState(0);
  const [showAnnotations, setShowAnnotations] = useState(true);
  const [selectedDisease, setSelectedDisease] = useState<Disease | null>(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev + 0.25, 3));
  };

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev - 0.25, 0.5));
  };

  const handleRotate = () => {
    setRotation(prev => (prev + 90) % 360);
  };

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = showAnnotations ? annotatedImage : originalImage;
    link.download = `liver-analysis-${new Date().toISOString().split('T')[0]}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getDiseaseType = (diseaseName: string) => {
    return DISEASE_TYPES.find(type => type.name === diseaseName.toLowerCase()) || DISEASE_TYPES[0];
  };

  const toggleDiseaseVisibility = (disease: Disease) => {
    setSelectedDisease(selectedDisease?.id === disease.id ? null : disease);
  };

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Image Container */}
      <div className="relative bg-gray-100 rounded-lg overflow-hidden">
        <div
          ref={containerRef}
          className="relative flex items-center justify-center min-h-[400px] p-4"
        >
          {!imageLoaded && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
            </div>
          )}
          
          <motion.div
            animate={{ 
              scale: zoom,
              rotate: rotation
            }}
            transition={{ duration: 0.3 }}
            className="relative max-w-full max-h-full"
          >
            <img
              src={showAnnotations ? annotatedImage : originalImage}
              alt="Liver histopathology analysis"
              className="max-w-full max-h-[500px] object-contain rounded-lg shadow-lg"
              onLoad={() => setImageLoaded(true)}
            />
          </motion.div>
        </div>

        {/* Image Controls */}
        <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2">
          <div className="flex items-center space-x-2 bg-white rounded-lg shadow-lg border border-gray-200 p-2">
            <button
              onClick={handleZoomOut}
              className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded transition-colors"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            
            <span className="px-3 py-1 text-sm font-medium text-gray-700 bg-gray-100 rounded">
              {Math.round(zoom * 100)}%
            </span>
            
            <button
              onClick={handleZoomIn}
              className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded transition-colors"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            
            <div className="w-px h-6 bg-gray-300 mx-1" />
            
            <button
              onClick={handleRotate}
              className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded transition-colors"
            >
              <RotateCw className="w-4 h-4" />
            </button>
            
            <div className="w-px h-6 bg-gray-300 mx-1" />
            
            <button
              onClick={() => setShowAnnotations(!showAnnotations)}
              className={`p-2 rounded transition-colors ${
                showAnnotations 
                  ? 'text-primary-600 bg-primary-50' 
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
              }`}
            >
              {showAnnotations ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
            </button>
            
            <button
              onClick={handleDownload}
              className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded transition-colors"
            >
              <Download className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Disease Legend */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <h3 className="text-lg font-semibold text-gray-900 mb-3">Detected Conditions</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          {diseases.map((disease) => {
            const diseaseType = getDiseaseType(disease.name);
            const isSelected = selectedDisease?.id === disease.id;
            
            return (
              <motion.button
                key={disease.id}
                onClick={() => toggleDiseaseVisibility(disease)}
                className={`p-3 rounded-lg border-2 transition-all duration-200 ${
                  isSelected
                    ? 'border-primary-500 bg-primary-50'
                    : 'border-gray-200 bg-white hover:border-gray-300 hover:bg-gray-50'
                }`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="flex items-center space-x-3">
                  <div
                    className="w-4 h-4 rounded-full"
                    style={{ backgroundColor: diseaseType.color }}
                  />
                  <div className="text-left">
                    <p className="font-medium text-sm text-gray-900">
                      {diseaseType.displayName}
                    </p>
                    <p className="text-xs text-gray-500">
                      {Math.round(disease.confidence * 100)}% confidence
                    </p>
                  </div>
                </div>
              </motion.button>
            );
          })}
        </div>

        {/* Selected Disease Details */}
        <AnimatePresence>
          {selectedDisease && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200"
            >
              <div className="flex items-start justify-between">
                <div>
                  <h4 className="font-medium text-gray-900">
                    {getDiseaseType(selectedDisease.name).displayName}
                  </h4>
                  <p className="text-sm text-gray-600 mt-1">
                    {getDiseaseType(selectedDisease.name).description}
                  </p>
                  <div className="mt-2 space-y-1">
                    <p className="text-sm">
                      <span className="font-medium">Confidence:</span> {Math.round(selectedDisease.confidence * 100)}%
                    </p>
                    <p className="text-sm">
                      <span className="font-medium">Severity:</span> 
                      <span className={`ml-1 px-2 py-1 rounded text-xs font-medium ${
                        selectedDisease.severity === 'high' ? 'bg-red-100 text-red-800' :
                        selectedDisease.severity === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-green-100 text-green-800'
                      }`}>
                        {selectedDisease.severity}
                      </span>
                    </p>
                    <p className="text-sm">
                      <span className="font-medium">Location:</span> 
                      X: {Math.round(selectedDisease.boundingBox.x)}, Y: {Math.round(selectedDisease.boundingBox.y)}
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedDisease(null)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default AnnotatedImage;
