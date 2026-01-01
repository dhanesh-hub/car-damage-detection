import React, { useState } from 'react';
import axios from 'axios';

const BRANDS = {
  "Maruti Suzuki": "Budget", "Hyundai": "Mid-Range", "Tata Motors": "Mid-Range",
  "Mahindra": "Mid-Range", "Toyota": "Mid-Range", "Mercedes-Benz": "Luxury",
  "BMW": "Luxury", "Audi": "Luxury"
};

function App() {
  const [file, setFile] = useState(null);
  const [brand, setBrand] = useState("Maruti Suzuki");
  const [model, setModel] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleScan = async () => {
    if (!file) return alert("Please upload an image first!");
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("brand", brand);
    formData.append("model", model);

    try {
      // ✅ UPDATED URL: Added https:// and /scan endpoint
      const response = await axios.post(
        "https://car-damage-detection-production.up.railway.app/scan", 
        formData
      );
      setResult(response.data);
    } catch (err) {
      console.error(err);
      alert("Error connecting to AI Backend. Check browser console for details.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '40px', fontFamily: 'sans-serif', maxWidth: '800px', margin: 'auto' }}>
      <h1>🛡️ 2025 AI Damage Estimator</h1>
      <hr />
      
      <div style={{ margin: '20px 0', display: 'flex', gap: '10px' }}>
        <select value={brand} onChange={(e) => setBrand(e.target.value)} style={{ padding: '10px' }}>
          {Object.keys(BRANDS).map(b => <option key={b}>{b}</option>)}
        </select>
        <input 
          placeholder="Enter Model (e.g. Swift, S-Class)" 
          value={model} 
          onChange={(e) => setModel(e.target.value)}
          style={{ padding: '10px', flex: 1 }}
        />
      </div>

      <input type="file" onChange={(e) => setFile(e.target.files[0])} style={{ marginBottom: '20px' }} />
      
      <button 
        onClick={handleScan} 
        disabled={loading}
        style={{ width: '100%', padding: '15px', backgroundColor: '#d9534f', color: 'white', border: 'none', cursor: 'pointer', fontWeight: 'bold' }}
      >
        {loading ? "AI ANALYZING STRUCTURAL DAMAGE..." : "RUN HIGH-PRECISION SCAN (0.10)"}
      </button>

      {result && (
        <div style={{ marginTop: '30px', border: '1px solid #ddd', padding: '20px', borderRadius: '8px' }}>
          <h3>📋 Repair Estimate for {brand} {model}</h3>
          <p><strong>Damage Spots Found:</strong> {result.damage_found}</p>
          <div dangerouslySetInnerHTML={{ __html: result.bill_html }} />
        </div>
      )}
    </div>
  );
}

export default App;