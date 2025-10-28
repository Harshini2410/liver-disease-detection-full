import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Heart, Upload, BarChart3, Shield, Clock, Users, FileText, Zap } from 'lucide-react';
import LoadingSpinner from '@/components/common/LoadingSpinner';

const Home: React.FC = () => {
  const features = [
    {
      icon: Upload,
      title: 'Easy Image Upload',
      description: 'Drag and drop histopathology images or browse files with support for JPEG, PNG, and TIFF formats.',
      color: 'text-blue-600'
    },
    {
      icon: Zap,
      title: 'AI-Powered Analysis',
      description: 'Advanced deep learning models provide accurate detection of liver diseases with high confidence scores.',
      color: 'text-purple-600'
    },
    {
      icon: BarChart3,
      title: 'Detailed Reports',
      description: 'Comprehensive analysis reports with annotated images, confidence scores, and detailed metrics.',
      color: 'text-green-600'
    },
    {
      icon: Shield,
      title: 'Medical Grade',
      description: 'HIPAA compliant platform designed specifically for healthcare professionals and medical institutions.',
      color: 'text-red-600'
    }
  ];

  const stats = [
    { icon: Users, label: 'Healthcare Professionals', value: '500+' },
    { icon: FileText, label: 'Images Analyzed', value: '10,000+' },
    { icon: Clock, label: 'Average Processing Time', value: '< 30s' },
    { icon: Heart, label: 'Accuracy Rate', value: '95%+' }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero Section */}
      <section className="relative bg-gradient-to-br from-primary-600 to-secondary-600 text-white">
        <div className="absolute inset-0 bg-black opacity-10"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
          <div className="text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              <div className="flex items-center justify-center mb-6">
                <div className="flex items-center justify-center w-16 h-16 bg-white bg-opacity-20 rounded-full">
                  <Heart className="w-8 h-8 text-white" />
                </div>
              </div>
              
              <h1 className="text-4xl md:text-6xl font-bold mb-6">
                AI-Powered Liver Disease Detection
              </h1>
              
              <p className="text-xl md:text-2xl text-white text-opacity-90 mb-8 max-w-3xl mx-auto">
                Advanced histopathology image analysis for healthcare professionals. 
                Accurate, fast, and reliable liver disease detection using cutting-edge AI technology.
              </p>
              
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Link
                  to="/upload"
                  className="inline-flex items-center px-8 py-3 bg-white text-primary-600 font-semibold rounded-lg hover:bg-gray-100 transition-colors duration-200"
                >
                  <Upload className="w-5 h-5 mr-2" />
                  Start Analysis
                </Link>
                
                <Link
                  to="/history"
                  className="inline-flex items-center px-8 py-3 border-2 border-white text-white font-semibold rounded-lg hover:bg-white hover:text-primary-600 transition-colors duration-200"
                >
                  <FileText className="w-5 h-5 mr-2" />
                  View History
                </Link>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <motion.h2
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="text-3xl md:text-4xl font-bold text-gray-900 mb-4"
            >
              Why Choose LiverDetect?
            </motion.h2>
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="text-xl text-gray-600 max-w-2xl mx-auto"
            >
              Our platform combines state-of-the-art AI technology with medical expertise 
              to provide the most accurate liver disease detection available.
            </motion.p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="text-center p-6 rounded-lg hover:shadow-lg transition-shadow duration-300"
              >
                <div className="flex items-center justify-center w-16 h-16 bg-gray-100 rounded-full mx-auto mb-4">
                  <feature.icon className={`w-8 h-8 ${feature.color}`} />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">
                  {feature.title}
                </h3>
                <p className="text-gray-600">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-20 bg-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="text-center"
              >
                <div className="flex items-center justify-center w-16 h-16 bg-primary-600 rounded-full mx-auto mb-4">
                  <stat.icon className="w-8 h-8 text-white" />
                </div>
                <div className="text-3xl font-bold text-gray-900 mb-2">
                  {stat.value}
                </div>
                <div className="text-gray-600">
                  {stat.label}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-primary-600">
        <div className="max-w-4xl mx-auto text-center px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Ready to Get Started?
            </h2>
            <p className="text-xl text-white text-opacity-90 mb-8">
              Upload your first histopathology image and experience the power of AI-driven liver disease detection.
            </p>
            <Link
              to="/upload"
              className="inline-flex items-center px-8 py-3 bg-white text-primary-600 font-semibold rounded-lg hover:bg-gray-100 transition-colors duration-200"
            >
              <Upload className="w-5 h-5 mr-2" />
              Upload Image Now
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default Home;
