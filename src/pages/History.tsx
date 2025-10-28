import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Search, Filter, Calendar, FileText, Eye, Download, AlertCircle } from 'lucide-react';
import LoadingSpinner from 'components/common/LoadingSpinner';
import { AnalysisRecord } from 'types/patient';
import { formatDate, formatConfidence } from 'utils/formatters';

const History: React.FC = () => {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');
  const [filterSeverity, setFilterSeverity] = useState<string>('all');
  const [isLoading, setIsLoading] = useState(true);
  const [analyses, setAnalyses] = useState<AnalysisRecord[]>([]);

  // Mock data - replace with actual API call
  React.useEffect(() => {
    const mockAnalyses: AnalysisRecord[] = [
      {
        id: '1',
        patientId: 'P001',
        timestamp: '2024-01-15T10:30:00Z',
        diseases: ['fibrosis', 'steatosis'],
        confidence: 0.92,
        imageUrl: '/api/images/sample1.jpg',
        reportUrl: '/api/reports/1.pdf'
      },
      {
        id: '2',
        patientId: 'P002',
        timestamp: '2024-01-14T14:20:00Z',
        diseases: ['inflammation'],
        confidence: 0.87,
        imageUrl: '/api/images/sample2.jpg',
        reportUrl: '/api/reports/2.pdf'
      },
      {
        id: '3',
        patientId: 'P003',
        timestamp: '2024-01-13T09:15:00Z',
        diseases: ['ballooning', 'fibrosis'],
        confidence: 0.78,
        imageUrl: '/api/images/sample3.jpg',
        reportUrl: '/api/reports/3.pdf'
      }
    ];

    // Simulate API call
    setTimeout(() => {
      setAnalyses(mockAnalyses);
      setIsLoading(false);
    }, 1000);
  }, []);

  const filteredAnalyses = analyses.filter(analysis => {
    const matchesSearch = analysis.patientId.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         analysis.diseases.some(disease => disease.toLowerCase().includes(searchTerm.toLowerCase()));
    
    const matchesSeverity = filterSeverity === 'all' || 
                           (filterSeverity === 'high' && analysis.confidence >= 0.8) ||
                           (filterSeverity === 'medium' && analysis.confidence >= 0.6 && analysis.confidence < 0.8) ||
                           (filterSeverity === 'low' && analysis.confidence < 0.6);
    
    return matchesSearch && matchesSeverity;
  });

  const handleViewAnalysis = (analysis: AnalysisRecord) => {
    // Navigate to results page with analysis data
    navigate('/results', { state: { analysis } });
  };

  const handleDownloadReport = (analysis: AnalysisRecord) => {
    if (analysis.reportUrl) {
      // Trigger download
      window.open(analysis.reportUrl, '_blank');
    }
  };

  const getSeverityColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600 bg-green-100';
    if (confidence >= 0.6) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getSeverityLabel = (confidence: number) => {
    if (confidence >= 0.8) return 'High';
    if (confidence >= 0.6) return 'Medium';
    return 'Low';
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <LoadingSpinner size="lg" text="Loading analysis history..." />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            Analysis History
          </h1>
          <p className="text-lg text-gray-600">
            View and manage your previous liver disease analysis results.
          </p>
        </motion.div>

        {/* Search and Filter */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white rounded-lg border border-gray-200 p-6 mb-8"
        >
          <div className="flex flex-col md:flex-row gap-4">
            {/* Search */}
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                <input
                  type="text"
                  placeholder="Search by patient ID or disease..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="input-field pl-10"
                />
              </div>
            </div>

            {/* Filter */}
            <div className="md:w-48">
              <div className="relative">
                <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                <select
                  value={filterSeverity}
                  onChange={(e) => setFilterSeverity(e.target.value)}
                  className="input-field pl-10"
                >
                  <option value="all">All Severities</option>
                  <option value="high">High Confidence</option>
                  <option value="medium">Medium Confidence</option>
                  <option value="low">Low Confidence</option>
                </select>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Results Count */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-6"
        >
          <p className="text-gray-600">
            Showing {filteredAnalyses.length} of {analyses.length} analyses
          </p>
        </motion.div>

        {/* Analysis List */}
        <div className="space-y-4">
          {filteredAnalyses.length === 0 ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-white rounded-lg border border-gray-200 p-12 text-center"
            >
              <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">No analyses found</h3>
              <p className="text-gray-600 mb-6">
                {searchTerm || filterSeverity !== 'all' 
                  ? 'Try adjusting your search criteria.'
                  : 'Upload your first image to get started.'}
              </p>
              <button
                onClick={() => navigate('/upload')}
                className="btn-primary"
              >
                Upload Image
              </button>
            </motion.div>
          ) : (
            filteredAnalyses.map((analysis, index) => (
              <motion.div
                key={analysis.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 * index }}
                className="bg-white rounded-lg border border-gray-200 p-6 hover:shadow-lg transition-shadow duration-200"
              >
                <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-4 mb-3">
                      <div>
                        <h3 className="text-lg font-semibold text-gray-900">
                          Patient {analysis.patientId}
                        </h3>
                        <p className="text-sm text-gray-500 flex items-center">
                          <Calendar className="w-4 h-4 mr-1" />
                          {formatDate(analysis.timestamp)}
                        </p>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSeverityColor(analysis.confidence)}`}>
                          {getSeverityLabel(analysis.confidence)} Confidence
                        </span>
                        <span className="text-sm font-medium text-gray-700">
                          {formatConfidence(analysis.confidence)}
                        </span>
                      </div>
                    </div>

                    <div className="flex flex-wrap gap-2 mb-3">
                      {analysis.diseases.map((disease) => (
                        <span
                          key={disease}
                          className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-sm font-medium capitalize"
                        >
                          {disease}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="flex items-center space-x-3 mt-4 lg:mt-0">
                    <button
                      onClick={() => handleViewAnalysis(analysis)}
                      className="flex items-center space-x-2 px-4 py-2 text-primary-600 bg-primary-50 rounded-lg hover:bg-primary-100 transition-colors"
                    >
                      <Eye className="w-4 h-4" />
                      <span>View</span>
                    </button>
                    
                    {analysis.reportUrl && (
                      <button
                        onClick={() => handleDownloadReport(analysis)}
                        className="flex items-center space-x-2 px-4 py-2 text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
                      >
                        <Download className="w-4 h-4" />
                        <span>Download</span>
                      </button>
                    )}
                  </div>
                </div>
              </motion.div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default History;
