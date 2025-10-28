export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Disease {
  id: string;
  name: string;
  confidence: number;
  boundingBox: BoundingBox;
  color: string;
  severity: 'low' | 'medium' | 'high';
  description?: string;
}

export interface DetectionMetrics {
  precision: number;
  recall: number;
  f1Score: number;
  mAP: number;
  processingTime: number;
}

export interface DetectionResult {
  resultId: string;
  status: 'processing' | 'completed' | 'error';
  diseases: Disease[];
  annotatedImage: string;
  originalImage: string;
  metrics: DetectionMetrics;
  timestamp: string;
  patientId?: string;
}

export interface DetectionRequest {
  image: File;
  patientId?: string;
  patientName?: string;
  patientAge?: number;
  patientGender?: 'male' | 'female' | 'other';
  additionalNotes?: string;
}

export interface ProcessingStatus {
  status: 'uploading' | 'processing' | 'analyzing' | 'generating_report' | 'completed' | 'error';
  progress: number;
  message: string;
  estimatedTime?: number;
}

export interface DiseaseType {
  id: string;
  name: string;
  displayName: string;
  color: string;
  description: string;
  normalRange?: string;
  symptoms?: string[];
}
