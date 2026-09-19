// AgriYield AI Platform Client Application Logic

let cropMixChartInstance = null;
let historyChartInstance = null;
let currentUserRole = "farmer";

document.addEventListener("DOMContentLoaded", () => {
  loadInitialData();
  const cropSelect = document.getElementById("p-crop");
  if (cropSelect) {
    cropSelect.addEventListener("change", updateAutoSeason);
  }
  updateAutoSeason();
});

function classifyCropSeason(crop) {
  const c = (crop || "").toLowerCase().trim();
  const rabiCrops = ['wheat', 'barley', 'mustard', 'gram', 'chickpea', 'oats', 'pea', 'potato', 'linseed'];
  const zaidCrops = ['watermelon', 'muskmelon', 'cucumber', 'vegetable', 'fodder', 'sunflower', 'gourd'];
  
  if (rabiCrops.some(r => c.includes(r))) {
    return "Rabi";
  } else if (zaidCrops.some(z => c.includes(z))) {
    return "Zaid";
  } else {
    return "Kharif";
  }
}

function updateAutoSeason() {
  const cropEl = document.getElementById("p-crop");
  const seasonEl = document.getElementById("p-season");
  if (cropEl && seasonEl) {
    const crop = cropEl.value;
    const season = classifyCropSeason(crop);
    seasonEl.value = season;
  }
}

function handleLocationChange(loc) {
  if (!loc) return;
  const weatherInput = document.getElementById("weather-search-input");
  if (weatherInput) {
    weatherInput.value = loc;
  }
  loadWeather(loc);
}

function loadInitialData() {
  const loc = document.getElementById("p-location") ? document.getElementById("p-location").value : "Punjab, India";
  loadWeather(loc);
  loadSoilAnalysis();
  loadMarketData();
  loadSchemes();
}

function switchAuthView(viewName) {
  document.getElementById("auth-login-view").classList.remove("active");
  document.getElementById("auth-register-view").classList.remove("active");
  
  if (viewName === "login") {
    document.getElementById("auth-login-view").classList.add("active");
  } else if (viewName === "register") {
    document.getElementById("auth-register-view").classList.add("active");
  }
}

function handleAuthSubmit(event, type) {
  event.preventDefault();
  let name = "Ravi Kumar";
  let farm = "Green Acres";

  if (type === "register") {
    name = document.getElementById("reg-name").value || "Ravi Kumar";
    farm = document.getElementById("reg-farm").value || "Green Acres";
    currentUserRole = document.getElementById("reg-role").value || "farmer";
  }

  document.getElementById("user-display-name").textContent = name;
  document.getElementById("user-role-label").textContent = currentUserRole === "agronomist" ? "Agronomist" : "Farmer";
  document.getElementById("dash-farmer-name").textContent = name;
  document.getElementById("dash-agro-name").textContent = name;
  document.getElementById("dash-farm-name").textContent = farm;

  document.getElementById("auth-login-view").classList.remove("active");
  document.getElementById("auth-register-view").classList.remove("active");
  document.getElementById("main-app-wrapper").style.display = "flex";

  switchDashboardRoleView();
  renderCropMixChart();
}

function togglePersonaMode() {
  currentUserRole = currentUserRole === "farmer" ? "agronomist" : "farmer";
  document.getElementById("user-role-label").textContent = currentUserRole === "agronomist" ? "Agronomist" : "Farmer";
  switchDashboardRoleView();
}

function fillDemo(persona) {
  if (persona === 'farmer') {
    document.getElementById("login-email").value = "farmer@agriyield.ai";
    document.getElementById("login-pass").value = "farmer123";
    document.getElementById("user-display-name").textContent = "Ravi Kumar";
    currentUserRole = "farmer";
  } else {
    document.getElementById("login-email").value = "admin@agriyield.ai";
    document.getElementById("login-pass").value = "admin123";
    document.getElementById("user-display-name").textContent = "Dr. Sharma";
    currentUserRole = "agronomist";
  }
}

function logout() {
  document.getElementById("main-app-wrapper").style.display = "none";
  document.getElementById("auth-login-view").classList.add("active");
}

function switchDashboardRoleView() {
  const farmerDash = document.getElementById("view-dashboard-farmer");
  const agroDash = document.getElementById("view-dashboard-agronomist");

  if (currentUserRole === "agronomist") {
    farmerDash.classList.remove("active");
    agroDash.classList.add("active");
  } else {
    agroDash.classList.remove("active");
    farmerDash.classList.add("active");
  }
}

function switchNavTab(tabName) {
  const tabs = document.querySelectorAll(".nav-item");
  tabs.forEach(t => t.classList.remove("active"));

  const targetTab = document.querySelector(`.nav-item[data-tab="${tabName}"]`);
  if (targetTab) targetTab.classList.add("active");

  const views = document.querySelectorAll(".container.view-section");
  views.forEach(v => v.classList.remove("active"));

  if (tabName === "dashboard") {
    switchDashboardRoleView();
  } else {
    const targetView = document.getElementById(`view-${tabName}`);
    if (targetView) targetView.classList.add("active");
  }

  if (tabName === "history") {
    loadHistoricalTrends();
  }
}

async function handlePredictSubmit(event) {
  event.preventDefault();
  const payload = {
    crop: document.getElementById("p-crop").value,
    irrigation: document.getElementById("p-irrigation").value,
    location: document.getElementById("p-location").value,
    field_area: document.getElementById("p-area").value,
    season: document.getElementById("p-season").value,
    soil_ph: document.getElementById("p-ph").value,
    moisture: document.getElementById("p-moisture").value,
    n: document.getElementById("p-n").value,
    p: document.getElementById("p-p").value,
    k: document.getElementById("p-k").value,
    organic_matter: document.getElementById("p-om").value,
    algorithm: document.getElementById("p-algo") ? document.getElementById("p-algo").value : "Gradient Boosting"
  };

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const json = await res.json();

    if (json.success) {
      renderPredictOutput(json.data);
    }
  } catch (err) {
    console.error("Predict error:", err);
  }
}

function renderPredictOutput(d) {
  document.getElementById("predict-empty").style.display = "none";
  document.getElementById("predict-output").style.display = "block";

  document.getElementById("res-algo-badge").textContent = d.algorithm_used;
  document.getElementById("res-yield-ha").textContent = d.predicted_per_ha;
  document.getElementById("res-area").textContent = d.field_area_acres;
  document.getElementById("res-total-yield").textContent = `${d.total_yield_tonnes} tonnes`;

  document.getElementById("res-r2").textContent = d.r2_score;
  document.getElementById("res-accuracy").textContent = `${d.model_accuracy_pct}%`;
  document.getElementById("res-mae").textContent = d.mae;

  const impactContainer = document.getElementById("res-impact-container");
  impactContainer.innerHTML = d.impact_factors.map(f => `
    <div class="impact-item">
      <span><strong>${f.factor}</strong> (${f.status})</span>
      <span style="color: var(--primary-green); font-weight: 700;">+${f.pct}%</span>
    </div>
  `).join('');

  const insightsList = document.getElementById("res-insights-list");
  insightsList.innerHTML = d.agronomic_insights.map(i => `<li>${i}</li>`).join('');

  document.getElementById("dash-stat-pred").textContent = parseInt(document.getElementById("dash-stat-pred").textContent || "0") + 1;
  document.getElementById("dash-stat-yield").textContent = `${d.predicted_per_ha} t`;
}

async function handleSoilSubmit(event) {
  if (event) event.preventDefault();
  const payload = {
    soil_ph: document.getElementById("s-ph").value,
    moisture: document.getElementById("s-moisture").value,
    n: document.getElementById("s-n").value,
    p: document.getElementById("s-p").value,
    k: document.getElementById("s-k").value,
    organic_matter: document.getElementById("s-om").value,
    intended_crop: document.getElementById("s-crop").value
  };

  try {
    const res = await fetch("/api/soil-analysis", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const json = await res.json();
    if (json.success) {
      const d = json.data;
      document.getElementById("s-score").innerHTML = `${d.soil_health_score} <span style="font-size: 1rem; color: var(--text-muted);">/ 100</span>`;
      document.getElementById("s-rating").textContent = d.rating;

      const planList = document.getElementById("s-plan-list");
      planList.innerHTML = d.improvement_plan.map(p => `<li>${p}</li>`).join('');
      document.getElementById("dash-stat-soil").textContent = Math.round(d.soil_health_score);
    }
  } catch (err) {
    console.error("Soil analysis error:", err);
  }
}

function loadSoilAnalysis() {
  handleSoilSubmit(null);
}

async function handleWeatherSearch(event) {
  if (event) event.preventDefault();
  const loc = document.getElementById("weather-search-input").value || "Punjab, India";
  loadWeather(loc);
}

async function loadWeather(loc) {
  try {
    const res = await fetch(`/api/weather?location=${encodeURIComponent(loc)}`);
    const json = await res.json();
    if (json.success) {
      const d = json.data;
      document.getElementById("w-temp").textContent = `${d.current.temp_c}°C`;
      document.getElementById("w-condition").textContent = `${d.current.condition} · ${d.location}`;
      document.getElementById("w-humidity").textContent = `${d.current.humidity_pct}%`;
      document.getElementById("w-wind").textContent = `${d.current.wind_kmh} km/h`;
      document.getElementById("w-evap").textContent = `${d.current.evapotranspiration_mm} mm/d`;
      document.getElementById("w-advisory").textContent = d.agri_advisory;

      const fList = document.getElementById("w-forecast-list");
      fList.innerHTML = d.forecast.map(f => `
        <div style="display: flex; justify-content: space-between; padding: 8px 12px; background: var(--bg-sage); border-radius: 6px; font-size: 0.88rem;">
          <span><strong>${f.day}</strong> · ${f.condition}</span>
          <span style="color: var(--primary-green); font-weight: 600;">${f.high}°C / ${f.low}°C (${f.rain_prob}% rain)</span>
        </div>
      `).join('');
    }
  } catch (err) {
    console.error("Weather error:", err);
  }
}

async function handleHistorySearch(event) {
  if (event) event.preventDefault();
  loadHistoricalTrends();
}

async function loadHistoricalTrends() {
  const crop = document.getElementById("h-crop").value;
  const loc = document.getElementById("h-location").value;

  try {
    const res = await fetch(`/api/historical-trends?crop=${encodeURIComponent(crop)}&location=${encodeURIComponent(loc)}`);
    const json = await res.json();
    if (json.success) {
      renderHistoryChart(json.data.history);
    }
  } catch (err) {
    console.error("History error:", err);
  }
}

function renderHistoryChart(data) {
  const ctx = document.getElementById("historyChart").getContext("2d");
  if (historyChartInstance) historyChartInstance.destroy();

  const labels = data.map(d => d.year);
  const yields = data.map(d => d.yield_tonnes_ha);
  const benchmarks = data.map(d => d.regional_benchmark);

  historyChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Farm Yield (tonnes/ha)',
          data: yields,
          borderColor: '#15803d',
          backgroundColor: 'rgba(21, 128, 61, 0.1)',
          fill: true,
          tension: 0.3
        },
        {
          label: 'Regional Benchmark',
          data: benchmarks,
          borderColor: '#94a3b8',
          borderDash: [5, 5],
          tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom' } }
    }
  });
}

function renderCropMixChart() {
  const ctx = document.getElementById("cropMixChart").getContext("2d");
  if (cropMixChartInstance) cropMixChartInstance.destroy();

  cropMixChartInstance = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Wheat (60%)', 'Rice (25%)', 'Pulses (15%)'],
      datasets: [{
        data: [60, 25, 15],
        backgroundColor: ['#15803d', '#0284c7', '#c2410c']
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: 'right' } }
    }
  });
}

async function loadMarketData() {
  const crop = document.getElementById("m-crop-select") ? document.getElementById("m-crop-select").value : "Wheat";
  try {
    const res = await fetch(`/api/market-intelligence?crop=${encodeURIComponent(crop)}`);
    const json = await res.json();
    if (json.success) {
      const d = json.data;
      document.getElementById("m-price").textContent = `₹${d.mandi_price_inr.toLocaleString()} / ${d.unit}`;
      document.getElementById("m-trend").textContent = `↑ ${d.trend_30d} 30-day price trend (${d.active_buyers} active buyers)`;

      const mandiList = document.getElementById("m-mandi-list");
      mandiList.innerHTML = d.nearby_mandis.map(m => `
        <div style="display: flex; justify-content: space-between; padding: 10px; background: var(--bg-sage); border-radius: 8px; font-size: 0.88rem;">
          <span><strong>${m.name}</strong> (${m.dist_km} km)</span>
          <strong style="color: var(--primary-green);">₹${m.price} / qtl</strong>
        </div>
      `).join('');

      const tipsList = document.getElementById("m-tips-list");
      tipsList.innerHTML = d.post_harvest_tips.map(t => `
        <div style="background: #f8faf6; padding: 12px; border-radius: 8px; font-size: 0.88rem; border-left: 4px solid var(--primary-green);">
          ${t}
        </div>
      `).join('');
    }
  } catch (err) {
    console.error("Market error:", err);
  }
}

async function loadSchemes() {
  try {
    const res = await fetch("/api/schemes");
    const json = await res.json();
    if (json.success) {
      const grid = document.getElementById("schemes-grid");
      grid.innerHTML = json.data.map(s => `
        <div class="card">
          <span class="user-badge" style="display: inline-block; font-size: 0.75rem; margin-bottom: 8px;">${s.category}</span>
          <h3 style="font-size: 1.1rem; margin-bottom: 8px;">${s.title}</h3>
          <p style="font-size: 0.88rem; color: #334155; margin-bottom: 12px;"><strong>Benefit:</strong> ${s.benefit}</p>
          <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 16px;">
            <div><strong>Eligibility:</strong> ${s.eligibility}</div>
            <div><strong>Documents:</strong> ${s.documents}</div>
          </div>
          <a href="${s.link}" target="_blank" class="btn-outline" style="display: inline-block; text-decoration: none;">Apply on Official Portal ↗</a>
        </div>
      `).join('');
    }
  } catch (err) {
    console.error("Schemes error:", err);
  }
}

function toggleChatDrawer() {
  const drawer = document.getElementById("chat-drawer");
  drawer.classList.toggle("hidden");
}

function handleChatKeyPress(event) {
  if (event.key === "Enter") {
    sendChatMessage();
  }
}

async function sendChatMessage() {
  const input = document.getElementById("chat-input-field");
  const msgText = input.value.trim();
  if (!msgText) return;

  const msgsContainer = document.getElementById("chat-messages");
  
  const userDiv = document.createElement("div");
  userDiv.className = "msg user";
  userDiv.textContent = msgText;
  msgsContainer.appendChild(userDiv);
  input.value = "";
  msgsContainer.scrollTop = msgsContainer.scrollHeight;

  try {
    const res = await fetch("/api/chat-guide", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msgText })
    });
    const json = await res.json();
    
    const botDiv = document.createElement("div");
    botDiv.className = "msg bot";
    botDiv.innerHTML = json.reply;
    msgsContainer.appendChild(botDiv);
    msgsContainer.scrollTop = msgsContainer.scrollHeight;
  } catch (err) {
    console.error("Chat error:", err);
  }
}
