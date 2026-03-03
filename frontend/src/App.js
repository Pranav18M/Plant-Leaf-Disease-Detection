import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Processing from './pages/Processing';
import Treatment from './pages/Treatment';
import TreatmentEnglish from './pages/TreatmentEnglish';
import TreatmentTamil from './pages/TreatmentTamil';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/processing" element={<Processing />} />
        <Route path="/treatment" element={<Treatment />} />
        <Route path="/treatment-english" element={<TreatmentEnglish />} />
        <Route path="/treatment-tamil" element={<TreatmentTamil />} />
      </Routes>
    </Router>
  );
}

export default App;