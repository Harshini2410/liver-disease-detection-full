import { DiseaseType } from '@/types/detection';

export const ACCEPTED_IMAGE_FORMATS = [
  'image/jpeg',
  'image/jpg', 
  'image/png',
  'image/tiff',
  'image/tif'
];

export const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB

export const DISEASE_TYPES: DiseaseType[] = [
  {
    id: 'fibrosis',
    name: 'fibrosis',
    displayName: 'Fibrosis',
    color: '#3B82F6',
    description: 'Scarring of liver tissue due to chronic damage',
    normalRange: 'F0 (No fibrosis)',
    symptoms: ['Fatigue', 'Abdominal pain', 'Swelling']
  },
  {
    id: 'steatosis',
    name: 'steatosis',
    displayName: 'Steatosis',
    color: '#10B981',
    description: 'Fatty liver disease with fat accumulation',
    normalRange: '< 5% fat content',
    symptoms: ['Weight gain', 'High cholesterol', 'Diabetes']
  },
  {
    id: 'inflammation',
    name: 'inflammation',
    displayName: 'Inflammation',
    color: '#EF4444',
    description: 'Inflammatory response in liver tissue',
    normalRange: 'Minimal inflammation',
    symptoms: ['Pain', 'Fever', 'Elevated liver enzymes']
  },
  {
    id: 'ballooning',
    name: 'ballooning',
    displayName: 'Hepatocyte Ballooning',
    color: '#F59E0B',
    description: 'Swelling and damage of liver cells',
    normalRange: 'No ballooning',
    symptoms: ['Cell damage', 'Impaired function']
  }
];

export const CONFIDENCE_LEVELS = {
  HIGH: { threshold: 0.8, color: '#10B981', label: 'High' },
  MEDIUM: { threshold: 0.6, color: '#F59E0B', label: 'Medium' },
  LOW: { threshold: 0.0, color: '#EF4444', label: 'Low' }
};

export const API_ENDPOINTS = {
  DETECT: '/api/detect',
  RESULTS: '/api/results',
  PATIENTS: '/api/patients',
  HISTORY: '/api/history'
};

export const PROCESSING_STAGES = [
  { stage: 'uploading', label: 'Uploading Image', progress: 10 },
  { stage: 'processing', label: 'Processing Image', progress: 30 },
  { stage: 'analyzing', label: 'AI Analysis', progress: 70 },
  { stage: 'generating_report', label: 'Generating Report', progress: 90 },
  { stage: 'completed', label: 'Analysis Complete', progress: 100 }
];
