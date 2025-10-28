import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { AlertCircle, CheckCircle, Clock, Zap } from 'lucide-react';
import ImageUpload from 'components/upload/ImageUpload';
import LoadingSpinner from 'components/common/LoadingSpinner';
import { useDetection } from 'hooks/useDetection';
import { DetectionRequest } from 'types/detection';
import { PROCESSING_STAGES } from 'utils/constants';

const Upload: React.FC = () => {
  const navigate = useNavigate();
  const { result, status, isProcessing, error, processImage, clearResult } = useDetection();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
  };

  const handleError = (errorMessage: string) => {
    // Error handling is done in the ImageUpload component
    console.error('Upload error:', errorMessage);
  };

  const handleProcessImage = async (file: File) => {
    try {
      const request: DetectionRequest = {
        image: file,
        // Add patient information if needed
      };

      await processImage(request);
    } catch (err) {
      console.error('Error processing image:', err);
    }
  };

  // Navigate to results when processing is complete
  React.useEffect(() => {
    if (result) {
      navigate('/results', { state: { result } });
    }
  }, [result, navigate]);

  const getCurrentStage = () => {
    if (!status) return null;
    return PROCESSING_STAGES.find(stage => stage.stage === status.status);
  };

  const currentStage = getCurrentStage();

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            Upload Histopathology Image
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Upload a liver histopathology image for AI-powered disease detection and analysis.
            Our system supports JPEG, PNG, and TIFF formats.
          </p>
        </motion.div>

        {/* Processing Status */}
        {isProcessing && status && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8"
          >
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">Processing Status</h3>
                <div className="flex items-center space-x-2">
                  <Zap className="w-5 h-5 text-primary-600" />
                  <span className="text-sm font-medium text-primary-600">AI Analysis</span>
                </div>
              </div>

              {/* Progress Bar */}
              <div className="mb-4">
                <div className="flex justify-between text-sm text-gray-600 mb-2">
                  <span>{status.message}</span>
                  <span>{Math.round(status.progress)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <motion.div
                    className="bg-primary-600 h-2 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${status.progress}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
              </div>

              {/* Stage Indicators */}
              <div className="flex items-center justify-between">
                {PROCESSING_STAGES.map((stage, index) => (
                  <div key={stage.stage} className="flex flex-col items-center">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                      status.progress >= stage.progress
                        ? 'bg-primary-600 text-white'
                        : 'bg-gray-200 text-gray-500'
                    }`}>
                      {status.progress >= stage.progress ? (
                        <CheckCircle className="w-4 h-4" />
                      ) : (
                        <span>{index + 1}</span>
                      )}
                    </div>
                    <span className={`text-xs mt-1 text-center ${
                      currentStage?.stage === stage.stage ? 'text-primary-600 font-medium' : 'text-gray-500'
                    }`}>
                      {stage.label}
                    </span>
                  </div>
                ))}
              </div>

              {/* Estimated Time */}
              {status.estimatedTime && (
                <div className="mt-4 flex items-center justify-center text-sm text-gray-500">
                  <Clock className="w-4 h-4 mr-2" />
                  <span>Estimated time remaining: {status.estimatedTime} seconds</span>
                </div>
              )}
            </div>
          </motion.div>
        )}

        {/* Error Display */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8"
          >
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex items-center">
                <AlertCircle className="w-5 h-5 text-red-600 mr-3" />
                <div>
                  <h3 className="text-sm font-medium text-red-800">Processing Error</h3>
                  <p className="text-sm text-red-700 mt-1">{error}</p>
                </div>
              </div>
              <div className="mt-4">
                <button
                  onClick={() => {
                    clearResult();
                    setSelectedFile(null);
                  }}
                  className="text-sm text-red-600 hover:text-red-800 underline"
                >
                  Try again
                </button>
              </div>
            </div>
          </motion.div>
        )}

        {/* Upload Interface */}
        {!isProcessing && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <ImageUpload
              onImageProcess={handleProcessImage}
              onError={handleError}
              isProcessing={isProcessing}
            />
          </motion.div>
        )}

        {/* Processing Overlay */}
        {isProcessing && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
          >
            <div className="bg-white rounded-lg p-8 max-w-md mx-4 text-center">
              <LoadingSpinner size="lg" color="primary" />
              <h3 className="text-lg font-semibold text-gray-900 mt-4 mb-2">
                Analyzing Your Image
              </h3>
              <p className="text-gray-600 mb-4">
                Our AI is processing your histopathology image. This may take a few moments.
              </p>
              <div className="text-sm text-gray-500">
                {status?.message || 'Processing...'}
              </div>
            </div>
          </motion.div>
        )}

        {/* Information Cards */}
        {!isProcessing && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6"
          >
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex items-center mb-3">
                <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center mr-3">
                  <Zap className="w-5 h-5 text-blue-600" />
                </div>
                <h3 className="font-semibold text-gray-900">AI-Powered Analysis</h3>
              </div>
              <p className="text-sm text-gray-600">
                Advanced deep learning models trained on thousands of histopathology images 
                provide accurate disease detection with high confidence scores.
              </p>
            </div>

            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex items-center mb-3">
                <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center mr-3">
                  <Clock className="w-5 h-5 text-green-600" />
                </div>
                <h3 className="font-semibold text-gray-900">Fast Processing</h3>
              </div>
              <p className="text-sm text-gray-600">
                Get results in under 30 seconds. Our optimized AI models provide 
                rapid analysis without compromising accuracy.
              </p>
            </div>

            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex items-center mb-3">
                <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center mr-3">
                  <CheckCircle className="w-5 h-5 text-purple-600" />
                </div>
                <h3 className="font-semibold text-gray-900">Medical Grade</h3>
              </div>
              <p className="text-sm text-gray-600">
                HIPAA compliant platform designed for healthcare professionals. 
                Secure, reliable, and trusted by medical institutions worldwide.
              </p>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default Upload;
