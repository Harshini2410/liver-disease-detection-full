# Liver Disease Detection System - Frontend

A modern, responsive web application for AI-powered liver disease detection using histopathology images. Built with React, TypeScript, and Tailwind CSS, this frontend provides a professional medical-grade interface for healthcare professionals.

## Features

### Medical-Grade Interface
- Clean, professional design optimized for healthcare professionals
- HIPAA compliant and accessible (WCAG 2.1 AA)
- Responsive design for tablets and desktops

### Image Upload & Processing
- Drag-and-drop image upload interface
- Support for JPEG, PNG, and TIFF formats
- Image preview with zoom, rotate, and basic editing
- Real-time upload progress and validation

### AI-Powered Analysis
- Real-time processing status updates
- Annotated image visualization with bounding boxes
- Confidence scores for each detected condition
- Multiple disease detection (Fibrosis, Steatosis, Inflammation, Ballooning)

### Comprehensive Results
- Interactive annotated images with disease overlays
- Confidence charts and metrics visualization
- Detailed performance metrics (Precision, Recall, F1-Score, mAP)
- Printable and downloadable reports

### Patient Management
- Analysis history tracking
- Patient data management
- Search and filter capabilities
- Export functionality

## Tech Stack

- **Frontend Framework**: React 18 with TypeScript
- **Styling**: Tailwind CSS with custom medical theme
- **Animations**: Framer Motion
- **Charts**: Recharts for data visualization
- **File Upload**: React Dropzone
- **Routing**: React Router v6
- **HTTP Client**: Axios
- **Icons**: Lucide React

## Getting Started

### Prerequisites

- Node.js 16+ and npm
- Backend API running (see backend documentation)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Harshini2410/liver-disease-detection.git
cd liver-disease-detection
```

2. Install dependencies:
```bash
npm install
```

3. Create environment file:
```bash
cp .env.example .env.local
```

4. Configure environment variables:
```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_APP_NAME=LiverDetect
```

5. Start the development server:
```bash
npm start
```

The application will be available at `http://localhost:3000`.

### Building for Production

```bash
npm run build
```

This creates an optimized production build in the `build` folder.

## Project Structure

```
src/
├── components/
│   ├── common/           # Reusable UI components
│   ├── upload/           # Image upload components
│   ├── results/          # Results visualization components
│   └── dashboard/        # Dashboard and analytics components
├── pages/                # Main application pages
├── services/             # API and utility services
├── types/                # TypeScript type definitions
├── utils/                # Utility functions and constants
├── hooks/                # Custom React hooks
└── styles/               # Global styles and CSS
```

## Key Components

### Image Upload
- **DragDropZone**: Drag-and-drop file upload with validation
- **ImageUpload**: Complete upload interface with preview and controls

### Results Visualization
- **DetectionResults**: Main results display with tabbed interface
- **AnnotatedImage**: Interactive image viewer with annotations
- **ConfidenceChart**: Confidence scores visualization
- **DiseaseMetrics**: Detailed metrics and performance data

### Common Components
- **Header**: Navigation with responsive mobile menu
- **Footer**: Site information and links
- **LoadingSpinner**: Animated loading indicators

## API Integration

The frontend integrates with a backend API for:

- Image upload and processing
- Detection results retrieval
- Patient data management
- Report generation and download

### Expected API Endpoints

```
POST /api/detect              # Upload image for analysis
GET  /api/results/:id         # Get detection results
GET  /api/results/:id/status  # Get processing status
GET  /api/patients            # Get patient list
POST /api/patients            # Create new patient
GET  /api/history             # Get analysis history
```

## Configuration

### Medical UI Theme

The application uses a custom medical-grade color scheme:

- **Primary**: Deep blue (#1E40AF) - Trust and professionalism
- **Secondary**: Teal (#0D9488) - Medical and clean
- **Disease Colors**:
  - Fibrosis: Blue (#3B82F6)
  - Steatosis: Green (#10B981)
  - Inflammation: Red (#EF4444)
  - Ballooning: Orange (#F59E0B)

### Accessibility Features

- WCAG 2.1 AA compliant color contrast
- Keyboard navigation support
- Screen reader friendly
- Focus management
- Reduced motion support

## Development

### Available Scripts

- `npm start` - Start development server
- `npm run build` - Build for production
- `npm test` - Run tests
- `npm run eject` - Eject from Create React App

### Code Style

The project uses:
- TypeScript for type safety
- ESLint for code linting
- Prettier for code formatting
- Tailwind CSS for styling

### Adding New Features

1. Create components in appropriate directories
2. Add TypeScript interfaces in `src/types/`
3. Update routing in `App.tsx`
4. Add API methods in `src/services/api.ts`
5. Update constants in `src/utils/constants.ts`

## Deployment

### Environment Variables

Configure these environment variables for production:

```env
REACT_APP_API_URL=https://your-api-domain.com
REACT_APP_APP_NAME=LiverDetect
REACT_APP_VERSION=1.0.0
```

### Build Optimization

The production build includes:
- Code splitting and lazy loading
- Image optimization
- CSS and JavaScript minification
- Tree shaking for smaller bundle size

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation wiki

## Acknowledgments

- Medical professionals who provided feedback
- Open source libraries and tools used
- AI research community for disease detection models
