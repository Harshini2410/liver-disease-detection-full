import { useState, useCallback, useEffect } from 'react';
import { Patient, PatientHistory, AnalysisRecord, PatientFormData } from '@/types/patient';
import { patientApi, historyApi } from '@/services/api';

interface UsePatientDataReturn {
  patients: Patient[];
  selectedPatient: Patient | null;
  patientHistory: PatientHistory | null;
  isLoading: boolean;
  error: string | null;
  loadPatients: () => Promise<void>;
  loadPatient: (patientId: string) => Promise<void>;
  createPatient: (patientData: PatientFormData) => Promise<Patient>;
  updatePatient: (patientId: string, updates: Partial<Patient>) => Promise<void>;
  deletePatient: (patientId: string) => Promise<void>;
  loadPatientHistory: (patientId: string) => Promise<void>;
  clearError: () => void;
}

export const usePatientData = (): UsePatientDataReturn => {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [patientHistory, setPatientHistory] = useState<PatientHistory | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadPatients = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const patientsData = await patientApi.getPatients();
      setPatients(patientsData);
    } catch (err) {
      console.error('Error loading patients:', err);
      setError(err instanceof Error ? err.message : 'Failed to load patients');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const loadPatient = useCallback(async (patientId: string) => {
    try {
      setIsLoading(true);
      setError(null);
      const patient = await patientApi.getPatient(patientId);
      setSelectedPatient(patient);
    } catch (err) {
      console.error('Error loading patient:', err);
      setError(err instanceof Error ? err.message : 'Failed to load patient');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const createPatient = useCallback(async (patientData: PatientFormData): Promise<Patient> => {
    try {
      setIsLoading(true);
      setError(null);
      
      // API expects patient data without createdAt/updatedAt (these are set server-side)
      const newPatient = await patientApi.createPatient({
        ...patientData
      });
      
      setPatients(prev => [...prev, newPatient]);
      return newPatient;
    } catch (err) {
      console.error('Error creating patient:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to create patient';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const updatePatient = useCallback(async (patientId: string, updates: Partial<Patient>) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const updatedPatient = await patientApi.updatePatient(patientId, {
        ...updates,
        updatedAt: new Date().toISOString()
      });
      
      setPatients(prev => 
        prev.map(patient => 
          patient.id === patientId ? updatedPatient : patient
        )
      );
      
      if (selectedPatient?.id === patientId) {
        setSelectedPatient(updatedPatient);
      }
    } catch (err) {
      console.error('Error updating patient:', err);
      setError(err instanceof Error ? err.message : 'Failed to update patient');
    } finally {
      setIsLoading(false);
    }
  }, [selectedPatient]);

  const deletePatient = useCallback(async (patientId: string) => {
    try {
      setIsLoading(true);
      setError(null);
      
      await patientApi.deletePatient(patientId);
      
      setPatients(prev => prev.filter(patient => patient.id !== patientId));
      
      if (selectedPatient?.id === patientId) {
        setSelectedPatient(null);
        setPatientHistory(null);
      }
    } catch (err) {
      console.error('Error deleting patient:', err);
      setError(err instanceof Error ? err.message : 'Failed to delete patient');
    } finally {
      setIsLoading(false);
    }
  }, [selectedPatient]);

  const loadPatientHistory = useCallback(async (patientId: string) => {
    try {
      setIsLoading(true);
      setError(null);
      const history = await historyApi.getPatientHistory(patientId);
      setPatientHistory(history);
    } catch (err) {
      console.error('Error loading patient history:', err);
      setError(err instanceof Error ? err.message : 'Failed to load patient history');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // Load patients on mount
  useEffect(() => {
    loadPatients();
  }, [loadPatients]);

  return {
    patients,
    selectedPatient,
    patientHistory,
    isLoading,
    error,
    loadPatients,
    loadPatient,
    createPatient,
    updatePatient,
    deletePatient,
    loadPatientHistory,
    clearError
  };
};
