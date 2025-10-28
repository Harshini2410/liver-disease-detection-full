import { useState, useCallback, useRef } from 'react';
import { DetectionResult, ProcessingStatus, DetectionRequest } from 'types/detection';
import { detectionApi } from 'services/api';
import { PROCESSING_STAGES } from 'utils/constants';

interface UseDetectionReturn {
  result: DetectionResult | null;
  status: ProcessingStatus | null;
  isProcessing: boolean;
  error: string | null;
  processImage: (request: DetectionRequest) => Promise<void>;
  clearResult: () => void;
  retry: () => void;
}

export const useDetection = (): UseDetectionReturn => {
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [status, setStatus] = useState<ProcessingStatus | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const currentResultId = useRef<string | null>(null);
  const pollingInterval = useRef<NodeJS.Timeout | null>(null);

  const startPolling = useCallback((resultId: string) => {
    if (pollingInterval.current) {
      clearInterval(pollingInterval.current);
    }

    pollingInterval.current = setInterval(async () => {
      try {
        const currentStatus = await detectionApi.getProcessingStatus(resultId);
        setStatus(currentStatus);

        if (currentStatus.status === 'completed') {
          const finalResult = await detectionApi.getResult(resultId);
          setResult(finalResult);
          setIsProcessing(false);
          if (pollingInterval.current) {
            clearInterval(pollingInterval.current);
            pollingInterval.current = null;
          }
        } else if (currentStatus.status === 'error') {
          setError('Processing failed. Please try again.');
          setIsProcessing(false);
          if (pollingInterval.current) {
            clearInterval(pollingInterval.current);
            pollingInterval.current = null;
          }
        }
      } catch (err) {
        console.error('Error polling status:', err);
        setError('Failed to check processing status.');
        setIsProcessing(false);
        if (pollingInterval.current) {
          clearInterval(pollingInterval.current);
          pollingInterval.current = null;
        }
      }
    }, 2000); // Poll every 2 seconds
  }, []);

  const processImage = useCallback(async (request: DetectionRequest) => {
    try {
      setIsProcessing(true);
      setError(null);
      setResult(null);
      setStatus(null);

      // Start with uploading status
      setStatus({
        status: 'uploading',
        progress: 0,
        message: 'Uploading image...',
        estimatedTime: 30
      });

      // Upload image and get result ID
      const resultId = await detectionApi.detectDisease(request);
      currentResultId.current = resultId;

      // Start polling for status updates
      startPolling(resultId);

    } catch (err) {
      console.error('Error processing image:', err);
      setError(err instanceof Error ? err.message : 'Failed to process image');
      setIsProcessing(false);
      setStatus(null);
    }
  }, [startPolling]);

  const clearResult = useCallback(() => {
    setResult(null);
    setStatus(null);
    setError(null);
    setIsProcessing(false);
    currentResultId.current = null;
    
    if (pollingInterval.current) {
      clearInterval(pollingInterval.current);
      pollingInterval.current = null;
    }
  }, []);

  const retry = useCallback(async () => {
    if (currentResultId.current) {
      try {
        setIsProcessing(true);
        setError(null);
        setStatus(null);
        startPolling(currentResultId.current);
      } catch (err) {
        setError('Failed to retry processing');
        setIsProcessing(false);
      }
    }
  }, [startPolling]);

  return {
    result,
    status,
    isProcessing,
    error,
    processImage,
    clearResult,
    retry
  };
};
