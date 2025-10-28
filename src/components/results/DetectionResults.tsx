import React, { useState } from 'react';
import { Download, Printer, Share2, FileText, AlertCircle } from 'lucide-react';
import { DetectionResult } from 'types/detection';
import AnnotatedImage from './AnnotatedImage';
import ConfidenceChart from './ConfidenceChart';
import { detectionApi } from 'services/api';
import { formatDate } from 'utils/formatters';

interface DetectionResultsProps {
  result: DetectionResult;
  className?: string;
}

const DetectionResults: React.FC<DetectionResultsProps> = ({ result, className = '' }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'analysis'>('overview');
  const [isDownloading, setIsDownloading] = useState<boolean>(false);

  const handleDownloadReport = async (format: 'pdf' | 'json' = 'pdf'): Promise<void> => {
    try {
      setIsDownloading(true);
      const blob = await detectionApi.downloadReport(result.resultId, format);

      if (!blob) {
        throw new Error('No data returned from report endpoint');
      }

      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `liver-analysis-report-${result.resultId}.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      // keep console error for debugging and show a user-friendly message
      // eslint-disable-next-line no-console
      console.error('Error downloading report:', err);
      alert('Failed to download report. Please try again.');
    } finally {
      setIsDownloading(false);
    }
  };

  const handlePrint = (): void => {
    window.print();
  };

  const handleShare = async (): Promise<void> => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'Liver Disease Detection Results',
          text: `Analysis completed on ${formatDate(result.timestamp)}`,
          url: window.location.href
        });
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error('Error sharing:', error);
      }
    } else if (navigator.clipboard) {
      await navigator.clipboard.writeText(window.location.href);
      alert('Link copied to clipboard');
    } else {
      alert('Sharing is not supported in this browser');
    }
  };

  type Tab = { id: 'overview' | 'analysis'; label: string; icon: React.ComponentType<any> };
  const tabs: Tab[] = [
    { id: 'overview', label: 'Overview', icon: FileText },
    { id: 'analysis', label: 'Analysis', icon: AlertCircle }
  ];

  return (
    <div className={`space-y-6 ${className}`}>
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Analysis Results</h2>
            <p className="text-gray-600 mt-1">Analysis completed on {formatDate(result.timestamp)}</p>
            <div className="flex items-center space-x-4 mt-2 text-sm text-gray-500">
              <span>Result ID: {result.resultId}</span>
              {result.patientId && <span>Patient ID: {result.patientId}</span>}
            </div>
          </div>

          <div className="flex items-center space-x-2 mt-4 md:mt-0">
            <button onClick={handlePrint} className="btn-outline flex items-center space-x-2">
              <Printer className="w-4 h-4" />
              <span>Print</span>
            </button>

            <button
              onClick={() => handleDownloadReport('pdf')}
              disabled={isDownloading}
              className="btn-primary flex items-center space-x-2"
            >
              {isDownloading ? (
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
              ) : (
                <Download className="w-4 h-4" />
              )}
              <span>Download PDF</span>
            </button>

            <button onClick={handleShare} className="btn-outline flex items-center space-x-2">
              <Share2 className="w-4 h-4" />
              <span>Share</span>
            </button>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6">
            {tabs.map((tab: Tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                    activeTab === tab.id
                      ? 'border-primary-500 text-primary-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <Icon className="w-4 h-4" />
                    <span>{tab.label}</span>
                  </div>
                </button>
              );
            })}
          </nav>
        </div>

        <div className="p-6">
          {activeTab === 'overview' && (
            <div className="space-y-6">
              <AnnotatedImage originalImage={result.originalImage} annotatedImage={result.annotatedImage} diseases={result.diseases} />
            </div>
          )}

          {activeTab === 'analysis' && (
            <div className="space-y-6">
              <ConfidenceChart diseases={result.diseases} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DetectionResults;
