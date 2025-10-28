import { ACCEPTED_IMAGE_FORMATS, MAX_FILE_SIZE } from 'utils/constants';

export interface ImageMetadata {
  name: string;
  size: number;
  type: string;
  lastModified: number;
  dimensions?: {
    width: number;
    height: number;
  };
}

export interface ImageValidationResult {
  isValid: boolean;
  error?: string;
  metadata?: ImageMetadata;
}

// Validate uploaded image file
export const validateImage = (file: File): ImageValidationResult => {
  // Check file type
  if (!ACCEPTED_IMAGE_FORMATS.includes(file.type)) {
    return {
      isValid: false,
      error: `Unsupported file format. Please upload a JPEG, PNG, or TIFF image.`
    };
  }

  // Check file size
  if (file.size > MAX_FILE_SIZE) {
    return {
      isValid: false,
      error: `File size too large. Maximum size is ${MAX_FILE_SIZE / (1024 * 1024)}MB.`
    };
  }

  // Check file name
  if (!file.name || file.name.trim() === '') {
    return {
      isValid: false,
      error: 'Invalid file name.'
    };
  }

  return {
    isValid: true,
    metadata: {
      name: file.name,
      size: file.size,
      type: file.type,
      lastModified: file.lastModified
    }
  };
};

// Get image dimensions
export const getImageDimensions = (file: File): Promise<{ width: number; height: number }> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);

    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve({
        width: img.naturalWidth,
        height: img.naturalHeight
      });
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('Failed to load image'));
    };

    img.src = url;
  });
};

// Create image preview URL
export const createImagePreview = (file: File): string => {
  return URL.createObjectURL(file);
};

// Revoke image preview URL to free memory
export const revokeImagePreview = (url: string): void => {
  URL.revokeObjectURL(url);
};

// Compress image if needed
export const compressImage = (
  file: File,
  maxWidth: number = 1920,
  maxHeight: number = 1080,
  quality: number = 0.8
): Promise<File> => {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = () => {
      // Calculate new dimensions
      let { width, height } = img;
      
      if (width > height) {
        if (width > maxWidth) {
          height = (height * maxWidth) / width;
          width = maxWidth;
        }
      } else {
        if (height > maxHeight) {
          width = (width * maxHeight) / height;
          height = maxHeight;
        }
      }

      canvas.width = width;
      canvas.height = height;

      // Draw and compress
      ctx?.drawImage(img, 0, 0, width, height);
      
      canvas.toBlob(
        (blob) => {
          if (blob) {
            const compressedFile = new File([blob], file.name, {
              type: file.type,
              lastModified: Date.now()
            });
            resolve(compressedFile);
          } else {
            reject(new Error('Failed to compress image'));
          }
        },
        file.type,
        quality
      );
    };

    img.onerror = () => reject(new Error('Failed to load image'));
    img.src = URL.createObjectURL(file);
  });
};

// Convert image to base64
export const imageToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result);
      } else {
        reject(new Error('Failed to convert image to base64'));
      }
    };
    
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });
};

// Check if image needs compression
export const shouldCompressImage = (file: File): boolean => {
  return file.size > 10 * 1024 * 1024; // Compress if larger than 10MB
};

// Generate thumbnail
export const generateThumbnail = (
  file: File,
  maxSize: number = 200
): Promise<string> => {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = () => {
      let { width, height } = img;
      
      // Calculate thumbnail dimensions
      if (width > height) {
        height = (height * maxSize) / width;
        width = maxSize;
      } else {
        width = (width * maxSize) / height;
        height = maxSize;
      }

      canvas.width = width;
      canvas.height = height;

      ctx?.drawImage(img, 0, 0, width, height);
      
      const thumbnailUrl = canvas.toDataURL('image/jpeg', 0.8);
      resolve(thumbnailUrl);
    };

    img.onerror = () => reject(new Error('Failed to generate thumbnail'));
    img.src = URL.createObjectURL(file);
  });
};
