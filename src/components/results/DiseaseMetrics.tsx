import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, Clock, Target, CheckCircle, AlertTriangle, XCircle } from 'lucide-react';
import { DetectionMetrics, Disease } from 'types/detection';
import { formatProcessingTime } from 'utils/formatters';

interface DiseaseMetricsProps {
  diseases: Disease[];
  metrics: DetectionMetrics;
  className?: string;
}

const DiseaseMetrics: React.FC<DiseaseMetricsProps> = ({
  diseases,
  metrics,
  className = ''
}) => {
  const getSeverityIcon = (severity: 'low' | 'medium' | 'high') => {
    switch (severity) {
      case 'high':
        return <XCircle className="w-5 h-5 text-red-500" />;
      case 'medium':
        return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
      case 'low':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      default:
        return <CheckCircle className="w-5 h-5 text-gray-500" />;
    }
  };

  const getSeverityColor = (severity: 'low' | 'medium' | 'high') => {
    switch (severity) {
      case 'high':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'medium':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'low':
        return 'text-green-600 bg-green-50 border-green-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const severityCounts = diseases.reduce((acc, disease) => {
    acc[disease.severity] = (acc[disease.severity] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const averageConfidence = diseases.length > 0 
    ? diseases.reduce((sum, disease) => sum + disease.confidence, 0) / diseases.length
    : 0;

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Overall Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="bg-white rounded-lg border border-gray-200 p-6"
        >
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <Target className="w-8 h-8 text-primary-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Total Detected</p>
              <p className="text-2xl font-bold text-gray-900">{diseases.length}</p>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="bg-white rounded-lg border border-gray-200 p-6"
        >
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <TrendingUp className="w-8 h-8 text-secondary-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Avg Confidence</p>
              <p className="text-2xl font-bold text-gray-900">
                {Math.round(averageConfidence * 100)}%
              </p>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="bg-white rounded-lg border border-gray-200 p-6"
        >
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <Clock className="w-8 h-8 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Processing Time</p>
              <p className="text-2xl font-bold text-gray-900">
                {formatProcessingTime(metrics.processingTime)}
              </p>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="bg-white rounded-lg border border-gray-200 p-6"
        >
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <CheckCircle className="w-8 h-8 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Model Accuracy</p>
              <p className="text-2xl font-bold text-gray-900">
                {Math.round(metrics.f1Score * 100)}%
              </p>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Severity Breakdown */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Severity Breakdown</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {(['high', 'medium', 'low'] as const).map((severity) => (
            <motion.div
              key={severity}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: severity === 'high' ? 0.1 : severity === 'medium' ? 0.2 : 0.3 }}
              className={`p-4 rounded-lg border ${getSeverityColor(severity)}`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  {getSeverityIcon(severity)}
                  <div>
                    <p className="font-medium capitalize">{severity} Severity</p>
                    <p className="text-sm opacity-75">
                      {severityCounts[severity] || 0} conditions detected
                    </p>
                  </div>
                </div>
                <div className="text-2xl font-bold">
                  {severityCounts[severity] || 0}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Model Performance Metrics */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Performance</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-3xl font-bold text-primary-600 mb-2">
              {Math.round(metrics.precision * 100)}%
            </div>
            <p className="text-sm font-medium text-gray-700">Precision</p>
            <p className="text-xs text-gray-500">True positives / (True positives + False positives)</p>
          </div>
          
          <div className="text-center">
            <div className="text-3xl font-bold text-secondary-600 mb-2">
              {Math.round(metrics.recall * 100)}%
            </div>
            <p className="text-sm font-medium text-gray-700">Recall</p>
            <p className="text-xs text-gray-500">True positives / (True positives + False negatives)</p>
          </div>
          
          <div className="text-center">
            <div className="text-3xl font-bold text-green-600 mb-2">
              {Math.round(metrics.f1Score * 100)}%
            </div>
            <p className="text-sm font-medium text-gray-700">F1-Score</p>
            <p className="text-xs text-gray-500">Harmonic mean of precision and recall</p>
          </div>
          
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-600 mb-2">
              {Math.round(metrics.mAP * 100)}%
            </div>
            <p className="text-sm font-medium text-gray-700">mAP</p>
            <p className="text-xs text-gray-500">Mean Average Precision</p>
          </div>
        </div>
      </div>

      {/* Individual Disease Details */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Detection Details</h3>
        <div className="space-y-3">
          {diseases.map((disease, index) => (
            <motion.div
              key={disease.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
            >
              <div className="flex items-center space-x-4">
                {getSeverityIcon(disease.severity)}
                <div>
                  <p className="font-medium text-gray-900 capitalize">{disease.name}</p>
                  <p className="text-sm text-gray-500">
                    Confidence: {Math.round(disease.confidence * 100)}%
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-4">
                <div className="text-right">
                  <p className="text-sm font-medium text-gray-700">Severity</p>
                  <p className={`text-sm capitalize ${
                    disease.severity === 'high' ? 'text-red-600' :
                    disease.severity === 'medium' ? 'text-yellow-600' :
                    'text-green-600'
                  }`}>
                    {disease.severity}
                  </p>
                </div>
                <div className="w-16 bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${
                      disease.confidence >= 0.8 ? 'bg-green-500' :
                      disease.confidence >= 0.6 ? 'bg-yellow-500' :
                      'bg-red-500'
                    }`}
                    style={{ width: `${disease.confidence * 100}%` }}
                  />
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default DiseaseMetrics;
