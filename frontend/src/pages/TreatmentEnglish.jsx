import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import CostCalculator from '../components/CostCalculator';

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
      <div className="max-w-7xl mx-auto">
        {/* Header */}
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

        {/* Description */}
        {treatment_english?.description && (
          <div className="bg-gray-800 rounded-xl p-6 mb-6 border border-gray-700">
            <p className="text-white text-center text-lg">
              📝 {treatment_english.description}
            </p>
          </div>
        )}

        {/* Main Grid: Treatments + Cost Calculator */}
        <div className="grid lg:grid-cols-3 gap-6 mb-8">
          {/* Chemical Treatments */}
          <div className="bg-gray-800 rounded-2xl p-6 border-t-4 border-blue-500">
            <h3 className="text-2xl font-bold text-blue-400 mb-6 flex items-center gap-2">
              🧪 Chemical Treatments
            </h3>
            
            <div className="space-y-6">
              {treatment_english?.chemical_treatments?.slice(0, 3)?.map((treatment, index) => {
                const match = treatment.cost?.match(/\d+/);
                const price = match ? parseInt(match[0]) : 0;

                return (
                  <div key={index} className="bg-gray-900 rounded-xl p-5 border border-gray-700 hover:border-blue-500 transition-all">
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
                      <div className="flex items-center justify-between">
                        <span className="text-red-400 text-xs line-through">Local: {treatment.cost}</span>
                        <span className="text-green-400 font-bold text-sm">
                          💰 Online: ₹{(price * 0.85).toFixed(0)}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Organic Treatments */}
          <div className="bg-gray-800 rounded-2xl p-6 border-t-4 border-green-500">
            <h3 className="text-2xl font-bold text-green-400 mb-6 flex items-center gap-2">
              🌿 Organic Treatments
            </h3>
            
            <div className="space-y-6">
              {treatment_english?.organic_treatments?.slice(0, 3)?.map((treatment, index) => {
                const match = treatment.cost?.match(/\d+/);
                const price = match ? parseInt(match[0]) : 0;

                return (
                  <div key={index} className="bg-gray-900 rounded-xl p-5 border border-gray-700 hover:border-green-500 transition-all">
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
                      <div className="flex items-center justify-between">
                        <span className="text-red-400 text-xs line-through">Local: {treatment.cost}</span>
                        <span className="text-green-400 font-bold text-sm">
                          💰 Online: ₹{(price * 0.85).toFixed(0)}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Cost Calculator - Online Pricing */}
          <div className="lg:row-span-2">
            <CostCalculator 
              treatmentEnglish={treatment_english}
              treatmentTamil={result.treatment_tamil}
              language="english"
            />
          </div>
        </div>

        {/* When to Treat */}
        {treatment_english?.when_to_treat && (
          <div className="bg-red-900 border-2 border-red-500 rounded-xl p-6 text-center mb-8">
            <p className="text-red-200 font-bold text-lg">
              ⏰ {treatment_english.when_to_treat}
            </p>
          </div>
        )}

        {/* Online Shopping Info */}
        <div className="bg-gradient-to-r from-blue-900 to-green-900 rounded-xl p-6 mb-8 border-2 border-accent">
          <h3 className="text-white font-bold text-xl mb-4 text-center">
            🛒 Buy Online & Save More!
          </h3>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-gray-900 bg-opacity-50 rounded-lg p-4 text-center">
              <p className="text-accent font-bold text-2xl mb-1">15%</p>
              <p className="text-gray-300 text-sm">Base Online Discount</p>
            </div>
            <div className="bg-gray-900 bg-opacity-50 rounded-lg p-4 text-center">
              <p className="text-accent font-bold text-2xl mb-1">20%</p>
              <p className="text-gray-300 text-sm">Bulk Discount (10+ acres)</p>
            </div>
            <div className="bg-gray-900 bg-opacity-50 rounded-lg p-4 text-center">
              <p className="text-accent font-bold text-2xl mb-1">FREE</p>
              <p className="text-gray-300 text-sm">Shipping (3+ acres)</p>
            </div>
          </div>
          <p className="text-center text-blue-200 text-sm mt-4">
            💡 Recommended: AgroStar, BigHaat, Flipkart Krishi
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <button
            onClick={() => navigate('/treatment', { state: { result } })}
            className="px-6 py-3 bg-gray-700 text-white rounded-xl hover:bg-gray-600 transition-all"
          >
            ← Back to Language Selection
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