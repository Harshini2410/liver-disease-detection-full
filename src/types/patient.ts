export interface Patient {
  id: string;
  name: string;
  age: number;
  gender: 'male' | 'female' | 'other';
  dateOfBirth?: string;
  medicalRecordNumber?: string;
  contactInfo?: {
    email?: string;
    phone?: string;
  };
  createdAt: string;
  updatedAt: string;
}

export interface PatientHistory {
  patientId: string;
  analyses: AnalysisRecord[];
  totalAnalyses: number;
  lastAnalysis?: string;
}

export interface AnalysisRecord {
  id: string;
  patientId: string;
  timestamp: string;
  diseases: string[];
  confidence: number;
  imageUrl: string;
  reportUrl?: string;
}

export interface PatientFormData {
  name: string;
  age: number;
  gender: 'male' | 'female' | 'other';
  medicalRecordNumber?: string;
  email?: string;
  phone?: string;
}
