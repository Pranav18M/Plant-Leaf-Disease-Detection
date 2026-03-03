import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

const TreatmentTamil = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const result = location.state?.result;

  if (!result) {
    navigate('/');
    return null;
  }

  const { tamil_name, confidence, treatment_tamil } = result;

  return (
    <div className="min-h-screen bg-dark py-8 px-4">
      <div className="max-w-6xl mx-auto tamil-text">
        <div className="text-center mb-8">
          <h1 className="text-3xl md:text-4xl font-bold text-accent mb-2">
            சிகிச்சை பரிந்துரைகள்
          </h1>
          <div className="h-1 w-32 bg-primary mx-auto mb-4"></div>
          <h2 className="text-xl md:text-2xl font-bold text-white mb-2">
            {tamil_name}
          </h2>
          <p className="text-blue-400">நம்பகத்தன்மை: {confidence}%</p>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-gray-800 rounded-2xl p-6 border-t-4 border-blue-500">
            <h3 className="text-2xl font-bold text-blue-400 mb-6">
              🧪 இரசாயன சிகிச்சை
            </h3>
            
            <div className="space-y-6">
              {treatment_tamil?.chemical_treatments_tamil?.slice(0, 3).map((treatment, index) => (
                <div key={index} className="bg-gray-900 rounded-xl p-5 border border-gray-700">
                  <h4 className="font-bold text-white text-lg mb-3">
                    {index + 1}. {treatment.name}
                  </h4>
                  
                  <div className="space-y-2 text-sm">
                    <p className="text-gray-400">
                      💧 {treatment.dosage}
                    </p>
                    <p className="text-gray-400">
                      📅 {treatment.application}
                    </p>
                    <p className="text-orange-400 font-bold">
                      💰 {treatment.cost}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-gray-800 rounded-2xl p-6 border-t-4 border-green-500">
            <h3 className="text-2xl font-bold text-green-400 mb-6">
              🌿 இயற்கை சிகிச்சை
            </h3>
            
            <div className="space-y-6">
              {treatment_tamil?.organic_treatments_tamil?.slice(0, 3).map((treatment, index) => (
                <div key={index} className="bg-gray-900 rounded-xl p-5 border border-gray-700">
                  <h4 className="font-bold text-white text-lg mb-3">
                    {index + 1}. {treatment.name}
                  </h4>
                  
                  <div className="space-y-2 text-sm">
                    <p className="text-gray-400">
                      💧 {treatment.dosage}
                    </p>
                    <p className="text-gray-400">
                      📅 {treatment.application}
                    </p>
                    <p className="text-orange-400 font-bold">
                      💰 {treatment.cost}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <button
            onClick={() => navigate('/treatment', { state: { result } })}
            className="px-6 py-3 bg-gray-700 text-white rounded-xl hover:bg-gray-600 transition-all"
          >
            ← மொழி தேர்வுக்கு திரும்பு
          </button>
          <button
            onClick={() => navigate('/')}
            className="px-6 py-3 bg-primary text-white rounded-xl hover:bg-green-700 transition-all"
          >
            🏠 புதிய பகுப்பாய்வு
          </button>
        </div>
      </div>
    </div>
  );
};

export default TreatmentTamil;