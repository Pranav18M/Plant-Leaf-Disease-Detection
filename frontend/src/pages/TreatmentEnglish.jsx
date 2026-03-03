import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

const TreatmentEnglish = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const result = location.state?.result;

  if (!result) {
    navigate('/');
    return null;
  }

  const { disease_name, confidence, treatment_english } = result;

  return (
    <div className="min-h-screen bg-dark py-8 px-4">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl md:text-4xl font-bold text-accent mb-2">
            Treatment Recommendations
          </h1>
          <div className="h-1 w-32 bg-primary mx-auto mb-4"></div>
          <h2 className="text-xl md:text-2xl font-bold text-white mb-2">
            {disease_name}
          </h2>
          <p className="text-blue-400">Confidence: {confidence}%</p>
        </div>

        {treatment_english?.description && (
          <div className="bg-gray-800 rounded-xl p-6 mb-6 border border-gray-700">
            <p className="text-white text-center text-lg">
              📝 {treatment_english.description}
            </p>
          </div>
        )}

        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-gray-800 rounded-2xl p-6 border-t-4 border-blue-500">
            <h3 className="text-2xl font-bold text-blue-400 mb-6">
              🧪 Chemical Treatments
            </h3>
            
            <div className="space-y-6">
              {treatment_english?.chemical_treatments?.slice(0, 3).map((treatment, index) => (
                <div key={index} className="bg-gray-900 rounded-xl p-5 border border-gray-700">
                  <h4 className="font-bold text-white text-lg mb-3">
                    {index + 1}. {treatment.name}
                  </h4>
                  
                  <div className="space-y-2 text-sm">
                    <p className="text-gray-400">
                      <span className="font-semibold text-gray-300">💧 Dosage:</span> {treatment.dosage}
                    </p>
                    <p className="text-gray-400">
                      <span className="font-semibold text-gray-300">📅 Application:</span> {treatment.application}
                    </p>
                    <p className="text-orange-400 font-bold">
                      💰 Cost: {treatment.cost}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-gray-800 rounded-2xl p-6 border-t-4 border-green-500">
            <h3 className="text-2xl font-bold text-green-400 mb-6">
              🌿 Organic Treatments
            </h3>
            
            <div className="space-y-6">
              {treatment_english?.organic_treatments?.slice(0, 3).map((treatment, index) => (
                <div key={index} className="bg-gray-900 rounded-xl p-5 border border-gray-700">
                  <h4 className="font-bold text-white text-lg mb-3">
                    {index + 1}. {treatment.name}
                  </h4>
                  
                  <div className="space-y-2 text-sm">
                    <p className="text-gray-400">
                      <span className="font-semibold text-gray-300">💧 Dosage:</span> {treatment.dosage}
                    </p>
                    <p className="text-gray-400">
                      <span className="font-semibold text-gray-300">📅 Application:</span> {treatment.application}
                    </p>
                    <p className="text-orange-400 font-bold">
                      💰 Cost: {treatment.cost}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {treatment_english?.when_to_treat && (
          <div className="bg-red-900 border-2 border-red-500 rounded-xl p-6 text-center mb-8">
            <p className="text-red-200 font-bold text-lg">
              ⏰ {treatment_english.when_to_treat}
            </p>
          </div>
        )}

        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <button
            onClick={() => navigate('/treatment', { state: { result } })}
            className="px-6 py-3 bg-gray-700 text-white rounded-xl hover:bg-gray-600 transition-all"
          >
            ← Back
          </button>
          <button
            onClick={() => navigate('/')}
            className="px-6 py-3 bg-primary text-white rounded-xl hover:bg-green-700 transition-all"
          >
            🏠 New Diagnosis
          </button>
        </div>
      </div>
    </div>
  );
};

export default TreatmentEnglish;