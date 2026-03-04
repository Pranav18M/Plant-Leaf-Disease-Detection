import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import CostCalculator from '../components/CostCalculator';

const TreatmentTamil = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const result = location.state?.result;

  if (!result) {
    navigate('/');
    return null;
  }

  const { tamil_name, confidence, treatment_tamil, treatment_english } = result;

  return (
    <div className="min-h-screen bg-dark py-8 px-4">
      <div className="max-w-7xl mx-auto tamil-text">
        {/* Header */}
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

        {/* Main Grid: Treatments + Cost Calculator */}
        <div className="grid lg:grid-cols-3 gap-6 mb-8">
          {/* Chemical Treatments */}
          <div className="bg-gray-800 rounded-2xl p-6 border-t-4 border-blue-500">
            <h3 className="text-2xl font-bold text-blue-400 mb-6 flex items-center gap-2">
              🧪 இரசாயன சிகிச்சை
            </h3>
            
            <div className="space-y-6">
              {treatment_tamil?.chemical_treatments_tamil?.slice(0, 3).map((treatment, index) => (
                <div key={index} className="bg-gray-900 rounded-xl p-5 border border-gray-700 hover:border-blue-500 transition-all">
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
                    <div className="flex items-center justify-between">
                      <span className="text-red-400 text-xs line-through">உள்ளூர்: {treatment.cost}</span>
                      <span className="text-green-400 font-bold text-sm">
                        💰 ஆன்லைன்: ₹{(parseInt(treatment.cost.match(/\d+/)[0]) * 0.85).toFixed(0)}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Organic Treatments */}
          <div className="bg-gray-800 rounded-2xl p-6 border-t-4 border-green-500">
            <h3 className="text-2xl font-bold text-green-400 mb-6 flex items-center gap-2">
              🌿 இயற்கை சிகிச்சை
            </h3>
            
            <div className="space-y-6">
              {treatment_tamil?.organic_treatments_tamil?.slice(0, 3).map((treatment, index) => (
                <div key={index} className="bg-gray-900 rounded-xl p-5 border border-gray-700 hover:border-green-500 transition-all">
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
                    <div className="flex items-center justify-between">
                      <span className="text-red-400 text-xs line-through">உள்ளூர்: {treatment.cost}</span>
                      <span className="text-green-400 font-bold text-sm">
                        💰 ஆன்லைன்: ₹{(parseInt(treatment.cost.match(/\d+/)[0]) * 0.85).toFixed(0)}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Cost Calculator - Tamil Version */}
          <div className="lg:row-span-2">
            <CostCalculator 
              treatmentEnglish={treatment_english}
              treatmentTamil={treatment_tamil}
              language="tamil"
            />
          </div>
        </div>

        {/* Online Shopping Info - Tamil */}
        <div className="bg-gradient-to-r from-blue-900 to-green-900 rounded-xl p-6 mb-8 border-2 border-accent">
          <h3 className="text-white font-bold text-xl mb-4 text-center">
            🛒 ஆன்லைனில் வாங்கி அதிகம் சேமிக்கவும்!
          </h3>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-gray-900 bg-opacity-50 rounded-lg p-4 text-center">
              <p className="text-accent font-bold text-2xl mb-1">15%</p>
              <p className="text-gray-300 text-sm">அடிப்படை ஆன்லைன் தள்ளுபடி</p>
            </div>
            <div className="bg-gray-900 bg-opacity-50 rounded-lg p-4 text-center">
              <p className="text-accent font-bold text-2xl mb-1">20%</p>
              <p className="text-gray-300 text-sm">மொத்த தள்ளுபடி (10+ ஏக்கர்)</p>
            </div>
            <div className="bg-gray-900 bg-opacity-50 rounded-lg p-4 text-center">
              <p className="text-accent font-bold text-2xl mb-1">இலவசம்</p>
              <p className="text-gray-300 text-sm">விநியோகம் (3+ ஏக்கர்)</p>
            </div>
          </div>
          <p className="text-center text-blue-200 text-sm mt-4">
            💡 பரிந்துரை: அக்ரோஸ்டார், பிக்ஹாட், பிளிப்கார்ட் கிருஷி
          </p>
        </div>

        {/* Action Buttons */}
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