import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

const Processing = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const result = location.state?.result;

  if (!result) {
    navigate('/');
    return null;
  }

  const { processing_steps, disease_name, tamil_name, confidence, is_healthy } = result;

  return (
    <div className="min-h-screen bg-dark py-8 px-4">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl md:text-4xl font-bold text-white mb-2">
            Disease Detection Results
          </h1>
          <p className="text-gray-400">Automated Image Processing & Classification</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 mb-8">
          {processing_steps?.slice(0, 7).map((step, index) => (
            <div key={index} className="bg-gray-800 rounded-lg overflow-hidden shadow-lg border border-gray-700">
              <div className="bg-gray-900 px-4 py-2 flex items-center gap-2">
                <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center text-white font-bold text-sm">
                  {index + 1}
                </div>
                <h3 className="text-sm font-semibold text-white truncate">{step.title}</h3>
              </div>
              
              <div className="p-2">
                <img 
                  src={`data:image/png;base64,${step.image}`}
                  alt={step.title}
                  className="w-full h-48 object-contain bg-black rounded"
                />
              </div>
            </div>
          ))}
        </div>

        <div className={`max-w-2xl mx-auto rounded-2xl overflow-hidden shadow-2xl border-4 ${
          is_healthy ? 'border-green-500 bg-green-900' : 'border-red-500 bg-red-900'
        }`}>
          <div className="p-6 md:p-8">
            <div className="flex justify-center mb-4">
              <span className={`px-6 py-2 rounded-full font-bold text-lg ${
                is_healthy ? 'bg-green-500 text-white' : 'bg-red-500 text-white'
              }`}>
                {is_healthy ? '✓ HEALTHY' : '⚠ DISEASED'}
              </span>
            </div>

            <h2 className="text-2xl md:text-3xl font-bold text-white text-center mb-3">
              {disease_name}
            </h2>

            {tamil_name && (
              <h3 className="text-xl md:text-2xl font-bold text-accent text-center mb-6 tamil-text">
                {tamil_name}
              </h3>
            )}

            <div className="w-full h-px bg-white opacity-25 my-6"></div>

            <div className="text-center mb-6">
              <p className="text-gray-300 text-sm mb-2 uppercase tracking-wide">Confidence</p>
              <p className="text-4xl md:text-5xl font-bold text-accent">
                {confidence}%
              </p>
            </div>

            {!is_healthy && (
              <div className="flex justify-center">
                <button
                  onClick={() => navigate('/treatment', { state: { result } })}
                  className="px-8 py-4 bg-accent text-dark font-bold text-lg rounded-xl hover:bg-yellow-400 transition-all shadow-lg hover:shadow-xl transform hover:scale-105"
                >
                  View Treatment Options →
                </button>
              </div>
            )}

            {is_healthy && (
              <div className="bg-green-800 rounded-xl p-4 text-center">
                <p className="text-white text-lg">
                  ✓ No treatment needed. Continue regular care.
                </p>
              </div>
            )}
          </div>
        </div>

        <div className="flex justify-center mt-8">
          <button
            onClick={() => navigate('/')}
            className="px-6 py-3 bg-gray-700 text-white rounded-xl hover:bg-gray-600 transition-all"
          >
            ← Back to Dashboard
          </button>
        </div>
      </div>
    </div>
  );
};

export default Processing;