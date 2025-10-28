import axios, { AxiosResponse } from 'axios';
import { DetectionRequest, DetectionResult, ProcessingStatus } from 'types/detection';
import { Patient, PatientHistory, AnalysisRecord } from 'types/patient';
import { API_ENDPOINTS } from 'utils/constants';

// Use REACT_APP_API_URL when provided (for production), otherwise
// use an empty string so the CRA dev server proxy (package.json -> proxy)
// forwards `/api/*` calls to the backend at localhost:5000 during development.
const API_BASE_URL = process.env.REACT_APP_API_URL || '';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000 // 30 seconds timeout for medical image processing
});

// Request interceptor for adding auth tokens if needed
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export interface ApiResponse<T> {
  data: T;
  message: string;
  success: boolean;
}

// Detection API
export const detectionApi = {
  // Upload image for analysis
  async detectDisease(request: DetectionRequest): Promise<string> {
    const formData = new FormData();
    formData.append('image', request.image);
    
    if (request.patientId) formData.append('patientId', request.patientId);
    if (request.patientName) formData.append('patientName', request.patientName);
    if (request.patientAge) formData.append('patientAge', request.patientAge.toString());
    if (request.patientGender) formData.append('patientGender', request.patientGender);
    if (request.additionalNotes) formData.append('additionalNotes', request.additionalNotes);

    // Let the browser set the multipart/form-data Content-Type (it includes the boundary).
    const response = await api.post<ApiResponse<{ resultId: string }>>(
      API_ENDPOINTS.DETECT,
      formData,
      {
        onUploadProgress: (progressEvent) => {
          const progress = progressEvent.total
            ? Math.round((progressEvent.loaded * 100) / progressEvent.total)
            : 0;
          console.log('Upload progress:', progress);
        },
      }
    );

    return response.data.data.resultId;
  },

  // Get detection results
  async getResult(resultId: string): Promise<DetectionResult> {
    const response = await api.get<ApiResponse<DetectionResult>>(
      `${API_ENDPOINTS.RESULTS}/${resultId}`
    );
    return response.data.data;
  },

  // Get processing status
  async getProcessingStatus(resultId: string): Promise<ProcessingStatus> {
    const response = await api.get<ApiResponse<ProcessingStatus>>(
      `${API_ENDPOINTS.RESULTS}/${resultId}/status`
    );
    return response.data.data;
  },

  // Download report
  async downloadReport(resultId: string, format: 'pdf' | 'json' = 'pdf'): Promise<Blob> {
    const response = await api.get(
      `${API_ENDPOINTS.RESULTS}/${resultId}/report`,
      {
        params: { format },
        responseType: 'blob',
      }
    );
    return response.data;
  }
};

// Patient API
export const patientApi = {
  // Get all patients
  async getPatients(): Promise<Patient[]> {
    const response = await api.get<ApiResponse<Patient[]>>(API_ENDPOINTS.PATIENTS);
    return response.data.data;
  },

  // Get patient by ID
  async getPatient(patientId: string): Promise<Patient> {
    const response = await api.get<ApiResponse<Patient>>(
      `${API_ENDPOINTS.PATIENTS}/${patientId}`
    );
    return response.data.data;
  },

  // Create new patient
  async createPatient(patient: Omit<Patient, 'id' | 'createdAt' | 'updatedAt'>): Promise<Patient> {
    const response = await api.post<ApiResponse<Patient>>(
      API_ENDPOINTS.PATIENTS,
      patient
    );
    return response.data.data;
  },

  // Update patient
  async updatePatient(patientId: string, updates: Partial<Patient>): Promise<Patient> {
    const response = await api.put<ApiResponse<Patient>>(
      `${API_ENDPOINTS.PATIENTS}/${patientId}`,
      updates
    );
    return response.data.data;
  },

  // Delete patient
  async deletePatient(patientId: string): Promise<void> {
    await api.delete(`${API_ENDPOINTS.PATIENTS}/${patientId}`);
  }
};

// History API
export const historyApi = {
  // Get patient history
  async getPatientHistory(patientId: string): Promise<PatientHistory> {
    const response = await api.get<ApiResponse<PatientHistory>>(
      `${API_ENDPOINTS.HISTORY}/patient/${patientId}`
    );
    return response.data.data;
  },

  // Get all analyses
  async getAllAnalyses(): Promise<AnalysisRecord[]> {
    const response = await api.get<ApiResponse<AnalysisRecord[]>>(
      API_ENDPOINTS.HISTORY
    );
    return response.data.data;
  },

  // Get analysis by ID
  async getAnalysis(analysisId: string): Promise<AnalysisRecord> {
    const response = await api.get<ApiResponse<AnalysisRecord>>(
      `${API_ENDPOINTS.HISTORY}/${analysisId}`
    );
    return response.data.data;
  }
};

export default api;
