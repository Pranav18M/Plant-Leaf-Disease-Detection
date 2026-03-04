import React, { useState } from 'react';

const CostCalculator = ({ treatmentEnglish, treatmentTamil, language = 'english' }) => {
  const [farmSize, setFarmSize] = useState(1);
  const [treatmentType, setTreatmentType] = useState('chemical');
  const [selectedTreatment, setSelectedTreatment] = useState(0);

  if (!treatmentEnglish) return null;

  const isEnglish = language === 'english';

  // Translations
  const translations = {
    english: {
      title: '💰 Cost Calculator',
      farmSize: 'Farm Size (Acres)',
      treatmentType: 'Treatment Type',
      chemical: '🧪 Chemical',
      organic: '🌿 Organic',
      selectTreatment: 'Select Treatment',
      localPrice: 'Local Shop Price:',
      onlinePrice: 'Online Price:',
      totalArea: 'Total Area:',
      acres: 'acres',
      subtotal: 'Subtotal (Local):',
      onlineSubtotal: 'Subtotal (Online):',
      onlineDiscount: 'Online Bulk Discount',
      finalCost: 'Final Online Cost:',
      youSave: 'You Save with Online',
      compareMsg: 'cheaper online for',
      priceComparison: '📊 Price Comparison (Online)',
      compareButton: 'Compare Chemical vs Organic',
      freeShipping: '✓ FREE Shipping',
      appDiscount: '+ Extra App Discount'
    },
    tamil: {
      title: '💰 செலவு கணக்கீடு',
      farmSize: 'பண்ணை அளவு (ஏக்கர்)',
      treatmentType: 'சிகிச்சை வகை',
      chemical: '🧪 இரசாயனம்',
      organic: '🌿 இயற்கை',
      selectTreatment: 'சிகிச்சையைத் தேர்ந்தெடுக்கவும்',
      localPrice: 'உள்ளூர் கடை விலை:',
      onlinePrice: 'ஆன்லைன் விலை:',
      totalArea: 'மொத்த பரப்பளவு:',
      acres: 'ஏக்கர்',
      subtotal: 'மொத்தம் (உள்ளூர்):',
      onlineSubtotal: 'மொத்தம் (ஆன்லைன்):',
      onlineDiscount: 'ஆன்லைன் மொத்த தள்ளுபடி',
      finalCost: 'இறுதி ஆன்லைன் செலவு:',
      youSave: 'ஆன்லைனில் நீங்கள் சேமிக்கிறீர்கள்',
      compareMsg: 'ஆன்லைனில் மலிவானது',
      priceComparison: '📊 விலை ஒப்பீடு (ஆன்லைன்)',
      compareButton: 'இரசாயன vs இயற்கை ஒப்பீடு',
      freeShipping: '✓ இலவச விநியோகம்',
      appDiscount: '+ கூடுதல் ஆப் தள்ளுபடி'
    }
  };

  const t = translations[language];

  const treatments = treatmentType === 'chemical' 
    ? (isEnglish ? treatmentEnglish.chemical_treatments : treatmentTamil.chemical_treatments_tamil)
    : (isEnglish ? treatmentEnglish.organic_treatments : treatmentTamil.organic_treatments_tamil);

  const currentTreatment = treatments[selectedTreatment];

  // Extract cost from string
  const extractCost = (costStr) => {
    const match = costStr.match(/₹(\d+)(?:-(\d+))?/);
    if (!match) return 0;
    const low = parseInt(match[1]);
    const high = match[2] ? parseInt(match[2]) : low;
    return (low + high) / 2;
  };

  const localBaseCost = extractCost(currentTreatment?.cost || '₹0');
  
  // Online prices are 12-18% cheaper than local
  const onlineDiscount = 0.15; // 15% cheaper online on base price
  const onlineBaseCost = localBaseCost * (1 - onlineDiscount);
  
  const dosagePerAcre = 2;
  const localTotalCost = localBaseCost * dosagePerAcre * farmSize;
  const onlineTotalCost = onlineBaseCost * dosagePerAcre * farmSize;
  
  // Online bulk discount (additional to base online discount)
  const bulkDiscount = farmSize >= 10 ? 0.20 : farmSize >= 5 ? 0.12 : farmSize >= 3 ? 0.08 : 0;
  const finalOnlineCost = onlineTotalCost * (1 - bulkDiscount);
  const totalSavings = localTotalCost - finalOnlineCost;
  const bulkSavings = onlineTotalCost - finalOnlineCost;

  // Free shipping threshold
  const freeShipping = farmSize >= 3;

  // Compare all treatments (online prices)
  const allCosts = treatments.map((t, idx) => ({
    name: t.name,
    localCost: extractCost(t.cost) * dosagePerAcre * farmSize,
    onlineCost: extractCost(t.cost) * (1 - onlineDiscount) * dosagePerAcre * farmSize * (1 - bulkDiscount),
    index: idx
  })).sort((a, b) => a.onlineCost - b.onlineCost);

  return (
    <div className={`bg-gray-800 rounded-2xl p-6 border-2 border-accent ${!isEnglish ? 'tamil-text' : ''}`}>
      <h3 className="text-2xl font-bold text-accent mb-6 flex items-center gap-2">
        {t.title}
      </h3>

      {/* Farm Size Input */}
      <div className="mb-6">
        <label className="block text-white font-semibold mb-2">
          {t.farmSize}
        </label>
        <input
          type="number"
          min="0.1"
          step="0.5"
          value={farmSize}
          onChange={(e) => setFarmSize(parseFloat(e.target.value) || 1)}
          className="w-full px-4 py-3 bg-gray-900 text-white rounded-xl border-2 border-gray-700 focus:border-accent focus:outline-none text-lg font-semibold"
        />
      </div>

      {/* Treatment Type Toggle */}
      <div className="mb-6">
        <label className="block text-white font-semibold mb-2">
          {t.treatmentType}
        </label>
        <div className="flex gap-2">
          <button
            onClick={() => { setTreatmentType('chemical'); setSelectedTreatment(0); }}
            className={`flex-1 px-4 py-3 rounded-xl font-semibold transition-all ${
              treatmentType === 'chemical'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            {t.chemical}
          </button>
          <button
            onClick={() => { setTreatmentType('organic'); setSelectedTreatment(0); }}
            className={`flex-1 px-4 py-3 rounded-xl font-semibold transition-all ${
              treatmentType === 'organic'
                ? 'bg-green-500 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            {t.organic}
          </button>
        </div>
      </div>

      {/* Treatment Selection */}
      <div className="mb-6">
        <label className="block text-white font-semibold mb-2">
          {t.selectTreatment}
        </label>
        <select
          value={selectedTreatment}
          onChange={(e) => setSelectedTreatment(parseInt(e.target.value))}
          className="w-full px-4 py-3 bg-gray-900 text-white rounded-xl border-2 border-gray-700 focus:border-accent focus:outline-none"
        >
          {treatments.map((treatment, idx) => (
            <option key={idx} value={idx}>
              {treatment.name}
            </option>
          ))}
        </select>
      </div>

      {/* Price Comparison: Local vs Online */}
      <div className="bg-gradient-to-r from-red-900 to-green-900 rounded-xl p-5 mb-6">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-red-200 text-xs mb-1">{t.localPrice}</p>
            <p className="text-white font-bold text-xl line-through">₹{(localBaseCost * dosagePerAcre).toFixed(0)}</p>
          </div>
          <div>
            <p className="text-green-200 text-xs mb-1">{t.onlinePrice}</p>
            <p className="text-green-400 font-bold text-xl">₹{(onlineBaseCost * dosagePerAcre).toFixed(0)}</p>
          </div>
        </div>
        <p className="text-center text-green-300 text-sm mt-2">
          💰 {((onlineDiscount) * 100).toFixed(0)}% {isEnglish ? 'cheaper online!' : 'ஆன்லைனில் மலிவானது!'}
        </p>
      </div>

      {/* Cost Breakdown */}
      <div className="bg-gray-900 rounded-xl p-5 mb-6 space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-gray-400">{t.totalArea}</span>
          <span className="text-white font-bold">{farmSize} {t.acres}</span>
        </div>
        <div className="h-px bg-gray-700"></div>
        
        {/* Local Shop Total */}
        <div className="flex justify-between items-center">
          <span className="text-red-300 text-sm line-through">{t.subtotal}</span>
          <span className="text-red-300 font-bold line-through">₹{localTotalCost.toFixed(0)}</span>
        </div>
        
        {/* Online Subtotal */}
        <div className="flex justify-between items-center">
          <span className="text-gray-400">{t.onlineSubtotal}</span>
          <span className="text-white font-bold">₹{onlineTotalCost.toFixed(0)}</span>
        </div>
        
        {/* Online Bulk Discount */}
        {bulkDiscount > 0 && (
          <>
            <div className="flex justify-between items-center text-green-400">
              <span>{t.onlineDiscount} ({(bulkDiscount * 100).toFixed(0)}%):</span>
              <span className="font-bold">-₹{bulkSavings.toFixed(0)}</span>
            </div>
            <div className="h-px bg-gray-700"></div>
          </>
        )}
        
        {/* Final Online Cost */}
        <div className="flex justify-between items-center text-lg pt-2">
          <span className="text-accent font-bold">{t.finalCost}</span>
          <span className="text-accent font-bold text-2xl">₹{finalOnlineCost.toFixed(0)}</span>
        </div>
      </div>

      {/* Total Savings Box */}
      <div className="bg-green-900 border-2 border-green-500 rounded-xl p-4 mb-6">
        <p className="text-green-200 text-center">
          <span className="text-2xl">🎉</span>
          <br />
          <strong className="text-green-400 text-xl">{t.youSave} ₹{totalSavings.toFixed(0)}</strong>
          <br />
          <span className="text-sm">{isEnglish ? 'vs Local Shop' : 'உள்ளூர் கடை விட'}</span>
        </p>
      </div>

      {/* Free Shipping & App Discount */}
      <div className="grid grid-cols-2 gap-3 mb-6">
        {freeShipping && (
          <div className="bg-blue-900 rounded-lg p-3 text-center border border-blue-500">
            <p className="text-blue-200 text-xs font-bold">{t.freeShipping}</p>
          </div>
        )}
        <div className="bg-purple-900 rounded-lg p-3 text-center border border-purple-500">
          <p className="text-purple-200 text-xs font-bold">{t.appDiscount}</p>
        </div>
      </div>

      {/* Price Comparison Table */}
      <div className="bg-gray-900 rounded-xl p-5 mb-6">
        <h4 className="text-white font-bold mb-3">{t.priceComparison}</h4>
        <div className="space-y-2">
          {allCosts.map((item, idx) => (
            <div 
              key={idx}
              className={`flex justify-between items-center p-3 rounded-lg transition-all ${
                item.index === selectedTreatment
                  ? 'bg-accent bg-opacity-20 border-2 border-accent'
                  : 'bg-gray-800 hover:bg-gray-700'
              }`}
            >
              <span className={`text-sm ${item.index === selectedTreatment ? 'text-accent font-bold' : 'text-gray-300'}`}>
                {idx === 0 && '🏆 '}{item.name}
              </span>
              <div className="text-right">
                <p className={`font-bold ${item.index === selectedTreatment ? 'text-accent' : 'text-white'}`}>
                  ₹{item.onlineCost.toFixed(0)}
                </p>
                <p className="text-xs text-gray-500 line-through">₹{item.localCost.toFixed(0)}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Compare Chemical vs Organic */}
      <button
        onClick={() => {
          const chemOnline = extractCost(treatmentEnglish.chemical_treatments[0].cost) * (1 - onlineDiscount) * dosagePerAcre * farmSize * (1 - bulkDiscount);
          const orgOnline = extractCost(treatmentEnglish.organic_treatments[0].cost) * (1 - onlineDiscount) * dosagePerAcre * farmSize * (1 - bulkDiscount);
          const diff = Math.abs(chemOnline - orgOnline);
          const cheaper = chemOnline < orgOnline ? (isEnglish ? 'Chemical' : 'இரசாயனம்') : (isEnglish ? 'Organic' : 'இயற்கை');
          alert(`💰 ${cheaper} ₹${diff.toFixed(0)} ${t.compareMsg} ${farmSize} ${t.acres}!`);
        }}
        className="w-full px-4 py-3 bg-gradient-to-r from-blue-500 to-green-500 text-white rounded-xl font-bold hover:shadow-lg transition-all"
      >
        {t.compareButton}
      </button>
    </div>
  );
};

export default CostCalculator;