import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import dashboardBg from '../assets/dashboard-bg.png';

const Dashboard = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      navigate('/processing', { state: { result: response.data } });
    } catch (error) {
      console.error('Upload error:', error);
      alert('Error processing image. Please try again.');
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen relative overflow-hidden">
      <div 
        className="absolute inset-0 bg-cover bg-center opacity-20"
        style={{ backgroundImage: `url(${dashboardBg})` }}
      />
      
      <div className="relative z-10 min-h-screen flex flex-col items-center justify-center p-4 md:p-8">
        <div className="text-center mb-8 md:mb-12">
          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold text-primary mb-4" style={{ fontFamily: 'cursive' }}>
            Plant Disease Detection
          </h1>
         <h1 className="text-2xl md:text-3xl font-bold text-secondary tamil-text">
            தாவர நோய் கண்டறிதல்
        </h1>
        </div> 

        <div className="w-full max-w-2xl">
          <div className="bg-white rounded-3xl shadow-2xl p-6 md:p-10 border-4 border-primary">
            <h2 className="text-2xl md:text-3xl font-bold text-center text-primary mb-6">
              Upload Leaf Image
            </h2>

            <div className="mb-6">
              <label className="flex flex-col items-center justify-center w-full h-48 md:h-64 border-4 border-dashed border-primary rounded-2xl cursor-pointer bg-green-50 hover:bg-green-100 transition-all">
                {preview ? (
                  <div className="relative w-full h-full p-4">
                    <img 
                      src={preview} 
                      alt="Preview" 
                      className="w-full h-full object-contain rounded-lg"
                    />
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    <svg className="w-16 h-16 md:w-20 md:h-20 mb-4 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    <p className="text-lg md:text-xl font-semibold text-primary mb-2">
                      Click to upload leaf image
                    </p>
                    <p className="text-sm text-gray-600">PNG, JPG up to 10MB</p>
                  </div>
                )}
                <input 
                  type="file" 
                  className="hidden" 
                  accept="image/*"
                  onChange={handleFileSelect}
                  disabled={loading}
                />
              </label>
            </div>

            <div className="flex flex-col sm:flex-row gap-4">
              {selectedFile && (
                <button
                  onClick={() => {
                    setSelectedFile(null);
                    setPreview(null);
                  }}
                  className="flex-1 px-6 py-3 bg-gray-200 text-gray-700 rounded-xl font-semibold hover:bg-gray-300 transition-all"
                  disabled={loading}
                >
                  Clear
                </button>
              )}
              <button
                onClick={handleUpload}
                disabled={!selectedFile || loading}
                className={`flex-1 px-6 py-4 rounded-xl font-bold text-lg transition-all ${
                  !selectedFile || loading
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-primary text-white hover:bg-green-700 shadow-lg hover:shadow-xl'
                }`}
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Processing...
                  </span>
                ) : (
                  'Analyze Disease'
                )}
              </button>
            </div>
          </div>
        </div>

        <p className="mt-8 text-center text-gray-600 text-sm md:text-base">
          Upload a clear image of the diseased leaf for accurate detection
        </p>
      </div>
    </div>
  );
};

export default Dashboard;