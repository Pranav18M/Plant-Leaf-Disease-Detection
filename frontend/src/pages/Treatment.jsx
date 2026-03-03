import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

const Treatment = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const result = location.state?.result;

  if (!result) {
    navigate('/');
    return null;
  }

  const { disease_name, tamil_name, confidence } = result;

  return (
    <div className="min-h-screen bg-dark py-8 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl md:text-4xl font-bold text-accent mb-4">
            Treatment Recommendations
          </h1>
          <div className="h-1 w-32 bg-primary mx-auto mb-6"></div>
          
          <h2 className="text-xl md:text-2xl font-bold text-white mb-2">
            {disease_name}
          </h2>
          {tamil_name && (
            <h3 className="text-lg md:text-xl font-bold text-accent mb-2 tamil-text">
              {tamil_name}
            </h3>
          )}
          <p className="text-blue-400">Confidence: {confidence}%</p>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div 
            onClick={() => navigate('/treatment-english', { state: { result } })}
            className="bg-gradient-to-br from-blue-900 to-blue-700 rounded-2xl p-8 cursor-pointer transform hover:scale-105 transition-all shadow-2xl border-2 border-blue-500 hover:border-blue-300"
          >
            <div className="text-center">
              <div className="w-20 h-20 bg-white rounded-full mx-auto mb-4 flex items-center justify-center">
                <span className="text-4xl">🇬🇧</span>
              </div>
              <h3 className="text-2xl font-bold text-white mb-2">English</h3>
              <p className="text-blue-200 mb-4">Chemical & Organic Treatments</p>
              <div className="inline-block px-4 py-2 bg-white text-blue-900 rounded-full font-semibold">
                View Details →
              </div>
            </div>
          </div>

          <div 
            onClick={() => navigate('/treatment-tamil', { state: { result } })}
            className="bg-gradient-to-br from-green-900 to-green-700 rounded-2xl p-8 cursor-pointer transform hover:scale-105 transition-all shadow-2xl border-2 border-green-500 hover:border-green-300"
          >
            <div className="text-center">
              <div className="w-20 h-20 bg-white rounded-full mx-auto mb-4 flex items-center justify-center tamil-text">
                <span className="text-3xl font-bold text-green-700">த</span>
              </div>
              <h3 className="text-2xl font-bold text-white mb-2 tamil-text">தமிழ்</h3>
              <p className="text-green-200 mb-4 tamil-text">இரசாயன & இயற்கை சிகிச்சை</p>
              <div className="inline-block px-4 py-2 bg-white text-green-900 rounded-full font-semibold tamil-text">
                விவரங்களைக் காண்க →
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-start gap-3">
            <div className="text-3xl">💡</div>
            <div>
              <h4 className="text-white font-bold mb-2">Select Your Preferred Language</h4>
              <p className="text-gray-400 text-sm">
                Choose English or Tamil to view detailed treatment recommendations.
              </p>
              <p className="text-gray-400 text-sm mt-2 tamil-text">
                விரிவான சிகிச்சை பரிந்துரைகளைப் பார்க்க ஆங்கிலம் அல்லது தமிழைத் தேர்ந்தெடுக்கவும்.
              </p>
            </div>
          </div>
        </div>

        <div className="flex justify-center mt-8">
          <button
            onClick={() => navigate(-1)}
            className="px-6 py-3 bg-gray-700 text-white rounded-xl hover:bg-gray-600 transition-all"
          >
            ← Back
          </button>
        </div>
      </div>
    </div>
  );
};

export default Treatment;