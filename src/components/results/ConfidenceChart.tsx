import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { motion } from 'framer-motion';
import { Disease } from 'types/detection';
import { DISEASE_TYPES, CONFIDENCE_LEVELS } from 'utils/constants';
import { formatConfidence, getConfidenceLevel } from 'utils/formatters';

interface ConfidenceChartProps {
  diseases: Disease[];
  className?: string;
}

const ConfidenceChart: React.FC<ConfidenceChartProps> = ({
  diseases,
  className = ''
}) => {
  // Prepare data for the chart
  const chartData = diseases.map(disease => {
    const diseaseType = DISEASE_TYPES.find(type => type.name === disease.name.toLowerCase());
    const confidenceLevel = getConfidenceLevel(disease.confidence);
    
    return {
      name: diseaseType?.displayName || disease.name,
      confidence: Math.round(disease.confidence * 100),
      color: diseaseType?.color || '#6B7280',
      level: confidenceLevel,
      severity: disease.severity
    };
  });

  // Sort by confidence level
  chartData.sort((a, b) => b.confidence - a.confidence);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-medium text-gray-900">{label}</p>
          <p className="text-sm text-gray-600">
            Confidence: <span className="font-medium">{data.confidence}%</span>
          </p>
          <p className="text-sm text-gray-600">
            Severity: <span className={`font-medium ${
              data.severity === 'high' ? 'text-red-600' :
              data.severity === 'medium' ? 'text-yellow-600' :
              'text-green-600'
            }`}>
              {data.severity}
            </span>
          </p>
        </div>
      );
    }
    return null;
  };

  const CustomBar = (props: any) => {
    const { fill, payload, ...rest } = props;
    return (
      <motion.rect
        {...rest}
        fill={payload.color}
        initial={{ height: 0 }}
        animate={{ height: rest.height }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      />
    );
  };

  return (
    <div className={`bg-white rounded-lg border border-gray-200 p-6 ${className}`}>
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">
          Detection Confidence Levels
        </h3>
        <p className="text-sm text-gray-600">
          Confidence scores for each detected liver condition
        </p>
      </div>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            margin={{
              top: 20,
              right: 30,
              left: 20,
              bottom: 5,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis 
              dataKey="name" 
              tick={{ fontSize: 12 }}
              angle={-45}
              textAnchor="end"
              height={80}
            />
            <YAxis 
              domain={[0, 100]}
              tick={{ fontSize: 12 }}
              label={{ value: 'Confidence (%)', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar 
              dataKey="confidence" 
              shape={<CustomBar />}
              radius={[4, 4, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Confidence Level Legend */}
      <div className="mt-6 pt-4 border-t border-gray-200">
        <h4 className="text-sm font-medium text-gray-900 mb-3">Confidence Levels</h4>
        <div className="flex flex-wrap gap-4">
          {Object.entries(CONFIDENCE_LEVELS).map(([level, config]) => (
            <div key={level} className="flex items-center space-x-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: config.color }}
              />
              <span className="text-sm text-gray-600">
                {config.label}: {config.threshold * 100}%+
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Summary Statistics */}
      <div className="mt-6 pt-4 border-t border-gray-200">
        <h4 className="text-sm font-medium text-gray-900 mb-3">Summary</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-gray-500">Total Detected:</span>
            <p className="font-medium text-gray-900">{diseases.length} conditions</p>
          </div>
          <div>
            <span className="text-gray-500">High Confidence:</span>
            <p className="font-medium text-green-600">
              {diseases.filter(d => getConfidenceLevel(d.confidence) === 'high').length}
            </p>
          </div>
          <div>
            <span className="text-gray-500">Medium Confidence:</span>
            <p className="font-medium text-yellow-600">
              {diseases.filter(d => getConfidenceLevel(d.confidence) === 'medium').length}
            </p>
          </div>
          <div>
            <span className="text-gray-500">Low Confidence:</span>
            <p className="font-medium text-red-600">
              {diseases.filter(d => getConfidenceLevel(d.confidence) === 'low').length}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConfidenceChart;
