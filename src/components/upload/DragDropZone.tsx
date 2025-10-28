import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';
import { Upload, Image, AlertCircle, CheckCircle } from 'lucide-react';
import { validateImage, ImageValidationResult } from 'services/imageProcessing';
import { ACCEPTED_IMAGE_FORMATS, MAX_FILE_SIZE } from 'utils/constants';
import { formatFileSize } from 'utils/formatters';

interface DragDropZoneProps {
  onFileSelect: (file: File) => void;
  onError: (error: string) => void;
  disabled?: boolean;
  className?: string;
}

const DragDropZone: React.FC<DragDropZoneProps> = ({
  onFileSelect,
  onError,
  disabled = false,
  className = ''
}) => {
  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    // Handle rejected files
    if (rejectedFiles.length > 0) {
      const rejection = rejectedFiles[0];
      if (rejection.errors[0]?.code === 'file-too-large') {
        onError(`File size too large. Maximum size is ${formatFileSize(MAX_FILE_SIZE)}.`);
      } else if (rejection.errors[0]?.code === 'file-invalid-type') {
        onError(`Invalid file type. Please upload a JPEG, PNG, or TIFF image.`);
      } else {
        onError('File upload failed. Please try again.');
      }
      return;
    }

    // Validate accepted file
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      const validation: ImageValidationResult = validateImage(file);
      
      if (validation.isValid) {
        onFileSelect(file);
      } else {
        onError(validation.error || 'Invalid file');
      }
    }
  }, [onFileSelect, onError]);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'image/*': ACCEPTED_IMAGE_FORMATS.map(format => format.replace('image/', '.'))
    },
    maxSize: MAX_FILE_SIZE,
    multiple: false,
    disabled
  });

  const getDropzoneClasses = () => {
    let classes = 'relative border-2 border-dashed rounded-lg p-8 transition-all duration-200 cursor-pointer ';
    
    if (disabled) {
      classes += 'border-gray-200 bg-gray-50 cursor-not-allowed ';
    } else if (isDragReject) {
      classes += 'border-red-300 bg-red-50 ';
    } else if (isDragActive) {
      classes += 'border-primary-400 bg-primary-50 ';
    } else {
      classes += 'border-gray-300 bg-white hover:border-primary-400 hover:bg-primary-50 ';
    }
    
    return classes + className;
  };

  const getIcon = () => {
    if (disabled) {
      return <Upload className="w-12 h-12 text-gray-400" />;
    } else if (isDragReject) {
      return <AlertCircle className="w-12 h-12 text-red-500" />;
    } else if (isDragActive) {
      return <CheckCircle className="w-12 h-12 text-primary-500" />;
    } else {
      return <Image className="w-12 h-12 text-gray-400" />;
    }
  };

  const getText = () => {
    if (disabled) {
      return 'Upload disabled';
    } else if (isDragReject) {
      return 'Invalid file type';
    } else if (isDragActive) {
      return 'Drop your image here';
    } else {
      return 'Drag & drop your image here';
    }
  };

  const getSubtext = () => {
    if (disabled) {
      return 'Processing in progress...';
    } else if (isDragReject) {
      return 'Only JPEG, PNG, and TIFF files are supported';
    } else {
      return 'or click to browse files';
    }
  };

  return (
    // Apply dropzone props to a plain div to avoid framer-motion/DOM event type conflicts
    <div {...getRootProps({ className: getDropzoneClasses() })}>
      <input {...getInputProps()} />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex flex-col items-center justify-center text-center">
          <motion.div
            animate={{ 
              scale: isDragActive ? 1.1 : 1,
              rotate: isDragActive ? 5 : 0
            }}
            transition={{ duration: 0.2 }}
          >
            {getIcon()}
          </motion.div>

          <motion.h3
            className="mt-4 text-lg font-medium text-gray-900"
            animate={{ color: isDragReject ? '#EF4444' : isDragActive ? '#3B82F6' : '#111827' }}
          >
            {getText()}
          </motion.h3>

          <p className="mt-2 text-sm text-gray-500">
            {getSubtext()}
          </p>

          <div className="mt-4 text-xs text-gray-400">
            <p>Supported formats: JPEG, PNG, TIFF</p>
            <p>Maximum file size: {formatFileSize(MAX_FILE_SIZE)}</p>
          </div>
        </div>
      </motion.div>

      {/* Drag overlay */}
      {isDragActive && !isDragReject && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute inset-0 bg-primary-600 bg-opacity-10 rounded-lg flex items-center justify-center"
        >
          <div className="text-center">
            <CheckCircle className="w-16 h-16 text-primary-600 mx-auto mb-2" />
            <p className="text-primary-600 font-medium">Release to upload</p>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default DragDropZone;
